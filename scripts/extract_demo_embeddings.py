"""Featurize training demonstrations with a trained policy.

Loads the checkpoint's own training dataset, runs the policy's embedding
extraction over every dataset sample, and writes the canonical trajectory
layout (the same layout ``eval_save_rollouts.py`` writes for rollouts):

    <output_dir>/
        episodes/metadata.yaml          source block + per-demo {index, length,
                                        success, buffer_episode_idx, split}
        policy_embeddings/<layer>.npz   (total_samples, d) float32, key "embeddings"
        observations/state.npz          low-dim policies only: the obs frame the
                                        policy acts from (window frame To-1), flattened
        actions.npz                     low-dim policies only: the action executed at
                                        that frame (oa_step_convention-aware)

Alignment contract: one embedding row per POLICY-DATASET SAMPLE (a sliding
training window), ordered demo-by-demo as the DataLoader emits them with
``shuffle=False``. Episode ``length`` counts dataset samples per demo, NOT raw
HDF5 timesteps. Per-demo sample counts are derived from the sampler's
``indices`` digitized into the replay buffer's ``episode_ends`` — demos masked
out of the split contribute no samples and are dropped, so the row/episode
correspondence is exact (we refuse to write on any mismatch).

Demo and rollout embeddings use the same machinery
(``diffusion_policy/common/policy_embeddings.py``) for the same layer name, so
both live in one representation space — the prerequisite for data-support
retrieval in the main repo.

Usage (from the repo root, in the submodule venv):
    python scripts/extract_demo_embeddings.py \
        --checkpoint data/outputs/train/<run>/checkpoints/latest.ckpt \
        --output-dir /tmp/demo_features --layer bottleneck_plan_t0
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from diffusion_policy.common.policy_embeddings import PolicyEmbeddingExtractor
from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_policy_from_checkpoint,
)
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("extract_demo_embeddings")


def resolve_checkpoint(path: pathlib.Path) -> pathlib.Path:
    """Resolve a .ckpt file, or pick the best/latest one from a train dir."""
    if path.is_file():
        return path
    checkpoint_dir = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"no checkpoint at {path}")
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    try:
        return get_best_checkpoint(checkpoints)
    except ValueError:
        latest = checkpoint_dir / "latest.ckpt"
        if latest.exists():
            return latest
        raise FileNotFoundError(f"no usable checkpoint in {checkpoint_dir}")


def episode_sample_counts(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Per-demo dataset-sample counts, in DataLoader iteration order.

    The replay buffer holds every demo in the HDF5, but the sampler only
    emits indices for this split's demos. Each ``sampler.indices`` row is
    ``(buffer_start, buffer_end, sample_start, sample_end)``; digitizing
    ``buffer_start`` into the buffer's ``episode_ends`` bins yields the demo
    index per sample. Returns ``(counts, buffer_episode_idx)`` for the demos
    that contribute at least one sample.
    """
    rb = getattr(dataset, "replay_buffer", None)
    sampler = getattr(dataset, "sampler", None)
    if rb is None or sampler is None or not hasattr(sampler, "indices"):
        raise RuntimeError(
            "cannot determine per-demo sample counts: dataset lacks "
            "replay_buffer.episode_ends or sampler.indices"
        )
    ends = np.asarray(rb.episode_ends[:], dtype=np.int64)
    indices = np.asarray(sampler.indices)
    if indices.ndim != 2 or indices.shape[1] < 1:
        raise RuntimeError(
            f"sampler.indices has unexpected shape {indices.shape}; "
            "expected (N, >=1) with buffer_start in column 0"
        )
    buf_start = indices[:, 0].astype(np.int64)
    # side='right': a buf_start equal to an end belongs to the NEXT demo —
    # episode_ends marks each demo's past-the-end position.
    demo_idx = np.searchsorted(ends, buf_start, side="right")
    counts = np.bincount(demo_idx, minlength=len(ends))
    contributing = np.nonzero(counts > 0)[0]
    return counts[contributing].astype(np.int64), contributing.astype(np.int64)


def split_obs_batch(obs, n_obs_steps: int, obs_is_dict: bool):
    """Convert a dataset batch's obs to the extractor's input format."""
    if obs_is_dict:
        keys = list(obs.keys())
        batch_size = obs[keys[0]].shape[0]
        return [
            {k: obs[k][i, :n_obs_steps].detach().cpu().numpy().astype(np.float32)
             for k in keys}
            for i in range(batch_size)
        ]
    return obs[:, :n_obs_steps].detach().cpu().numpy().astype(np.float32)


def extract_split(
    dataset,
    extractor: PolicyEmbeddingExtractor,
    n_obs_steps: int,
    batch_size: int,
    device: torch.device,
    batch_seed_offset: int,
    state_frame: Optional[int],
    action_frame: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
    """Embed every sample of one dataset split, preserving DataLoader order."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    embs: List[np.ndarray] = []
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    n_batches = 0
    for batch in tqdm.tqdm(loader, desc="batches"):
        obs_input = split_obs_batch(batch["obs"], n_obs_steps, extractor.obs_is_dict)
        action_np = None
        if extractor.needs_actions or action_frame is not None:
            action_np = batch["action"].detach().cpu().numpy().astype(np.float32)
        embs.append(
            extractor.extract(
                obs_input,
                action_np if extractor.needs_actions else None,
                batch_seed=batch_seed_offset + n_batches,
            )
        )
        if state_frame is not None:
            obs_np = np.asarray(obs_input, dtype=np.float32)
            states.append(obs_np[:, state_frame].reshape(obs_np.shape[0], -1))
        if action_frame is not None:
            actions.append(action_np[:, action_frame])
        n_batches += 1
    return (
        np.concatenate(embs, axis=0),
        np.concatenate(states, axis=0) if states else None,
        np.concatenate(actions, axis=0) if actions else None,
        n_batches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True,
                        help=".ckpt file or train dir (best/latest is picked)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", default="bottleneck_plan_t0",
                        help="see diffusion_policy/common/policy_embeddings.py")
    parser.add_argument("--dataset-path", default=None,
                        help="override the checkpoint's stored training dataset path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-validation", action="store_true",
                        help="also embed the validation split (appended after train)")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    checkpoint = resolve_checkpoint(pathlib.Path(args.checkpoint))
    device = torch.device(args.device)

    logger.info("loading checkpoint %s", checkpoint)
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)
    extractor = PolicyEmbeddingExtractor(policy, args.layer, device)
    n_obs_steps = int(policy.n_obs_steps)

    if args.dataset_path is not None and "dataset_path" in cfg.task.dataset:
        cfg.task.dataset.dataset_path = args.dataset_path
    dataset_path = str(cfg.task.dataset.get("dataset_path", ""))
    logger.info("instantiating training dataset (%s)", dataset_path or "stored path")
    train_set = hydra.utils.instantiate(cfg.task.dataset)

    splits = [("train", train_set)]
    if args.include_validation:
        val_set = train_set.get_validation_dataset()
        if len(val_set) > 0:
            splits.append(("val", val_set))
        else:
            logger.info("validation split is empty; skipping")

    # Low-dim policies: also store the state the policy acts from and the
    # action executed there, for state/state-action retrieval features.
    is_lowdim = isinstance(policy, DiffusionUnetLowdimPolicy)
    state_frame = n_obs_steps - 1 if is_lowdim else None
    action_frame = None
    if is_lowdim:
        action_frame = (
            n_obs_steps - 1 if getattr(policy, "oa_step_convention", False) else n_obs_steps
        )

    all_embs, all_states, all_actions = [], [], []
    episodes = []
    seed_offset = 0
    for split_name, ds in splits:
        counts, buffer_idx = episode_sample_counts(ds)
        if int(counts.sum()) != len(ds):
            raise RuntimeError(
                f"split {split_name!r}: per-demo counts sum to {int(counts.sum())} "
                f"but the dataset has {len(ds)} samples; refusing to write a "
                "misaligned layout"
            )
        logger.info(
            "split %s: %d samples across %d demos", split_name, len(ds), len(counts)
        )
        embs, states, actions, n_batches = extract_split(
            ds, extractor, n_obs_steps, args.batch_size, device,
            batch_seed_offset=seed_offset,
            state_frame=state_frame, action_frame=action_frame,
        )
        seed_offset += n_batches
        if embs.shape[0] != int(counts.sum()):
            raise RuntimeError(
                f"split {split_name!r}: extracted {embs.shape[0]} embeddings for "
                f"{int(counts.sum())} samples; refusing to write a misaligned layout"
            )
        all_embs.append(embs)
        if states is not None:
            all_states.append(states)
        if actions is not None:
            all_actions.append(actions)
        for n, buf_i in zip(counts, buffer_idx):
            episodes.append({
                "index": len(episodes),
                "length": int(n),
                "success": True,  # demonstrations are successful by construction
                "buffer_episode_idx": int(buf_i),
                "split": split_name,
            })

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    total = sum(ep["length"] for ep in episodes)
    assert embeddings.shape[0] == total, (embeddings.shape, total)

    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": {
            "kind": "demos",
            "dataset": dataset_path,
            "checkpoint": str(checkpoint),
            "layer": args.layer,
        },
        "episodes": episodes,
    }
    with open(output_dir / "episodes" / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    emb_dir = output_dir / "policy_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(emb_dir / f"{args.layer}.npz", embeddings=embeddings)
    logger.info("wrote %s embeddings for %d demos", embeddings.shape, len(episodes))

    if all_states:
        states = np.concatenate(all_states, axis=0).astype(np.float32)
        assert states.shape[0] == total, (states.shape, total)
        (output_dir / "observations").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "observations" / "state.npz", observations=states
        )
    if all_actions:
        actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        assert actions.shape[0] == total, (actions.shape, total)
        np.savez_compressed(output_dir / "actions.npz", actions=actions)
    logger.info("done: %s", output_dir)


if __name__ == "__main__":
    main()
