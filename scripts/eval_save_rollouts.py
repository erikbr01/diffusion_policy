"""Evaluate a diffusion-policy checkpoint and write the canonical rollout layout.

Runs N evaluation episodes with the task's env runner (robomimic / mimicgen /
robocasa), records per-episode length + success, extracts per-timestep policy
embeddings from the diffusion U-Net, and writes:

    <output_dir>/
        episodes/metadata.yaml          episodes: [{index, length, success}]
        policy_embeddings/<layer>.npz   (total_steps, d) float32, key "embeddings"
        media/ep{idx:04d}_{success|failure}.mp4   (with --num-vis != 0)

Per-timestep alignment: the low-dim env runners record one row per policy
inference call (one action chunk of ``n_action_steps`` executed steps), storing
the observation window and the full predicted action plan. Episode lengths in
``metadata.yaml`` count those rows, and one embedding is extracted per row, so
embeddings align 1:1 with the episode sample ranges.

Embedding layer names are documented in
``diffusion_policy/common/policy_embeddings.py`` (the shared extraction
machinery). The default ``bottleneck_plan_t0`` conditions the UNet bottleneck
on the actual predicted plan at the final denoising step.

Usage (from the repo root, in the sim conda env):
    python scripts/eval_save_rollouts.py \
        --checkpoint data/outputs/train/<run>/checkpoints/latest.ckpt \
        --output-dir /tmp/rollouts --num-episodes 50 --num-vis 5
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import pickle
import shutil
import sys
from typing import List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
import tqdm
import yaml

from diffusion_policy.common.policy_embeddings import (
    DEFAULT_LAYER,
    PolicyEmbeddingExtractor,
)
from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_policy_from_checkpoint,
)
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)

logger = logging.getLogger("eval_save_rollouts")


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Policy-embedding extraction (shared machinery in common/policy_embeddings)
# ---------------------------------------------------------------------------

def _episode_records(pkl_path: pathlib.Path) -> list:
    """Load one episode's per-inference-step records (DataFrame or list of dicts)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if hasattr(data, "iloc"):
        return [data.iloc[t] for t in range(len(data))]
    return list(data)


def extract_embeddings(
    policy,
    episodes_dir: pathlib.Path,
    layer: str,
    lengths: List[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract one embedding per recorded inference step across all episodes."""
    extractor = PolicyEmbeddingExtractor(policy, layer, device)

    pkls = sorted(episodes_dir.glob("ep*.pkl"))
    if len(pkls) != len(lengths):
        raise RuntimeError(
            f"{len(pkls)} episode pickles but metadata lists {len(lengths)} episodes"
        )

    all_obs, all_actions = [], []
    for pkl_path, ep_len in zip(pkls, lengths):
        records = _episode_records(pkl_path)
        if len(records) != ep_len:
            raise RuntimeError(
                f"{pkl_path.name} has {len(records)} records but metadata says {ep_len}"
            )
        for row in records:
            if extractor.obs_is_dict:
                all_obs.append(
                    {k: np.asarray(v, dtype=np.float32) for k, v in row["obs"].items()}
                )
            else:
                obs = np.asarray(row["obs"], dtype=np.float32)
                all_obs.append(obs[np.newaxis, :] if obs.ndim == 1 else obs)
            if extractor.needs_actions:
                all_actions.append(np.asarray(row["action"], dtype=np.float32))

    action_arr = np.stack(all_actions, axis=0) if extractor.needs_actions else None
    n_total = len(all_obs)
    logger.info("extracting layer=%s for %d timesteps", layer, n_total)

    results = []
    for start in tqdm.tqdm(range(0, n_total, batch_size), desc="embeddings"):
        results.append(
            extractor.extract(
                all_obs[start:start + batch_size],
                action_arr[start:start + batch_size]
                if extractor.needs_actions
                else None,
                batch_seed=start,
            )
        )
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Canonical rollout layout
# ---------------------------------------------------------------------------

def _read_native_metadata(episodes_dir: pathlib.Path) -> Tuple[List[int], List[bool]]:
    with open(episodes_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    lengths = [int(x) for x in meta["episode_lengths"]]
    successes = [bool(x) for x in meta.get("episode_successes", [False] * len(lengths))]
    return lengths, successes


def _write_canonical_metadata(
    episodes_dir: pathlib.Path, lengths: List[int], successes: List[bool]
) -> None:
    metadata = {
        "episodes": [
            {"index": i, "length": int(n), "success": bool(s)}
            for i, (n, s) in enumerate(zip(lengths, successes))
        ]
    }
    with open(episodes_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def _rename_media(media_dir: pathlib.Path) -> None:
    """Rename runner mp4s (ep0000_succ.mp4) to the canonical ep0000_success.mp4."""
    if not media_dir.is_dir():
        return
    for path in sorted(media_dir.glob("*.mp4")):
        new_name = path.name.replace("_succ.", "_success.").replace("_fail.", "_failure.")
        if new_name != path.name:
            path.rename(media_dir / new_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True,
                        help=".ckpt file, or a train dir containing checkpoints/.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100000,
                        help="Test start seed; episode i uses seed + i.")
    parser.add_argument("--layer", default=DEFAULT_LAYER,
                        help="Policy-embedding layer name (see module docstring).")
    parser.add_argument("--num-vis", type=int, default=0,
                        help="Episodes to record as mp4 (-1 = all).")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Parallel envs; default keeps the checkpoint config value.")
    parser.add_argument("--dataset-path", default=None,
                        help="Override the env runner's dataset path (env metadata).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for embedding extraction.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    output_dir = pathlib.Path(args.output_dir)
    for sub in ("episodes", "media", "policy_embeddings"):
        if (output_dir / sub).exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{output_dir / sub} already exists; pass --overwrite to replace"
                )
            shutil.rmtree(output_dir / sub)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(pathlib.Path(args.checkpoint))
    device = torch.device(args.device)
    logger.info("loading checkpoint %s", checkpoint)
    policy, cfg = get_policy_from_checkpoint(checkpoint, device=device)

    num_vis = args.num_episodes if args.num_vis < 0 else min(args.num_vis, args.num_episodes)
    runner_cfg = cfg.task.env_runner
    runner_cfg.n_train = 0
    runner_cfg.n_train_vis = 0
    runner_cfg.n_test = args.num_episodes
    runner_cfg.n_test_vis = num_vis
    runner_cfg.test_start_seed = args.seed
    if args.n_envs is not None:
        runner_cfg.n_envs = args.n_envs
    if isinstance(policy, DiffusionUnetHybridImagePolicy):
        # The image runner's episode saving runs sequentially and records a
        # video for every episode (asserts n_envs == 1, n_test_vis == n_test).
        if args.n_envs not in (None, 1) or num_vis != args.num_episodes:
            logger.info(
                "image policy: forcing n_envs=1 and per-episode videos "
                "(requested n_envs=%s, num_vis=%d)", args.n_envs, num_vis
            )
        runner_cfg.n_envs = 1
        runner_cfg.n_test_vis = args.num_episodes
    if args.dataset_path is not None and "dataset_path" in runner_cfg:
        runner_cfg.dataset_path = args.dataset_path

    env_runner = hydra.utils.instantiate(
        runner_cfg, output_dir=str(output_dir), save_episodes=True
    )
    logger.info("running %d episodes (seed=%d)", args.num_episodes, args.seed)
    runner_log = env_runner.run(policy)

    json_log = {k: v for k, v in runner_log.items() if isinstance(v, (int, float, str, bool))}
    with open(output_dir / "eval_log.json", "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

    episodes_dir = output_dir / "episodes"
    lengths, successes = _read_native_metadata(episodes_dir)
    embeddings = extract_embeddings(
        policy, episodes_dir, args.layer, lengths, args.batch_size, device
    )
    if embeddings.shape[0] != sum(lengths):
        raise RuntimeError(
            f"{embeddings.shape[0]} embeddings but episode lengths sum to {sum(lengths)}"
        )

    emb_dir = output_dir / "policy_embeddings"
    emb_dir.mkdir(exist_ok=True)
    np.savez_compressed(emb_dir / f"{args.layer}.npz", embeddings=embeddings)
    _write_canonical_metadata(episodes_dir, lengths, successes)
    _rename_media(output_dir / "media")
    logger.info(
        "done: %d episodes, %d/%d successes, embeddings %s",
        len(lengths), sum(successes), len(successes), embeddings.shape,
    )


if __name__ == "__main__":
    main()
