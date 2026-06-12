"""Evaluate a diffusion-policy checkpoint and write the canonical rollout layout.

Runs N evaluation episodes with the task's env runner (robomimic / mimicgen /
robocasa), records per-episode length + success, extracts per-timestep policy
embeddings from the diffusion U-Net, and writes:

    <output_dir>/
        episodes/metadata.yaml          episodes: [{index, length, success}]
        policy_embeddings/<layer>.npz   (total_steps, d) float32, key "embeddings"
        observations/state.npz          (total_steps, d_obs) float32, key "observations"
        actions.npz                     (total_steps, d_action) float32, key "actions"
        media/ep{idx:04d}_{success|failure}.mp4   (with --num-vis != 0)

``observations/state.npz`` is the low-dim state at the moment of each recorded
inference step: for low-dim policies the LAST frame of the recorded (To, d_obs)
obs window, flattened; for image policies the low-dim ('agent_pos'-style) keys
of the obs dict — i.e. the non-image entries, last frame each — concatenated in
sorted key order. ``actions.npz`` is the FIRST action the runner executes from
each step's predicted plan: the runners step the env with the policy's
``action`` output, which is ``action_pred[To-1 : To-1+n_action_steps]`` for
image policies and ``action_pred[start : start+n_action_steps]`` with
``start = To-1`` (``oa_step_convention``) or ``To`` for low-dim policies, minus
the runner's first ``n_latency_steps``; we store the first row of that slice.

Per-timestep alignment: the low-dim env runners record one row per policy
inference call (one action chunk of ``n_action_steps`` executed steps), storing
the observation window and the full predicted action plan. Episode lengths in
``metadata.yaml`` count those rows, and one embedding is extracted per row, so
embeddings align 1:1 with the episode sample ranges.

Embedding layer names are documented in
``diffusion_policy/common/policy_embeddings.py`` (the shared extraction
machinery). The default ``bottleneck_plan_t0`` conditions the UNet bottleneck
on the actual predicted plan at the final denoising step. ``--layer
obs_encoder`` skips the UNet pass and stores the policy's conditioning vector
(obs-encoder features for image policies, the flattened raw obs window for
low-dim policies).

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
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
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
# Raw state / executed-action extraction
# ---------------------------------------------------------------------------

def executed_action_index(policy, n_latency_steps: int) -> int:
    """Index into the recorded action plan of the first executed action.

    Mirrors each policy's ``predict_action`` slicing (``action =
    action_pred[start : start + n_action_steps]``) plus the low-dim runner's
    latency cut (``action[:, n_latency_steps:]``); the image runner executes
    the slice as-is.
    """
    if isinstance(policy, (DiffusionUnetHybridImagePolicy, DiffusionTransformerHybridImagePolicy)):
        return policy.n_obs_steps - 1
    if getattr(policy, "pred_action_steps_only", False):
        return n_latency_steps
    start = policy.n_obs_steps - (1 if getattr(policy, "oa_step_convention", False) else 0)
    return start + n_latency_steps


def _state_vector(row_obs, obs_is_dict: bool) -> np.ndarray:
    """Flattened low-dim state for one recorded step (see module docstring)."""
    if obs_is_dict:
        lowdim_keys = sorted(k for k, v in row_obs.items() if np.asarray(v).ndim == 2)
        if not lowdim_keys:
            raise RuntimeError(f"no low-dim obs keys among {sorted(row_obs)}")
        return np.concatenate(
            [np.asarray(row_obs[k], dtype=np.float32)[-1].ravel() for k in lowdim_keys]
        )
    obs = np.asarray(row_obs, dtype=np.float32)
    return obs.ravel() if obs.ndim == 1 else obs[-1].ravel()


def extract_states_and_actions(
    policy,
    episodes_dir: pathlib.Path,
    lengths: List[int],
    n_latency_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per recorded step: the flattened low-dim state + the first executed action."""
    obs_is_dict = isinstance(
        policy, (DiffusionUnetHybridImagePolicy, DiffusionTransformerHybridImagePolicy)
    )
    exec_idx = executed_action_index(policy, n_latency_steps)

    pkls = sorted(episodes_dir.glob("ep*.pkl"))
    if len(pkls) != len(lengths):
        raise RuntimeError(
            f"{len(pkls)} episode pickles but metadata lists {len(lengths)} episodes"
        )

    states, actions = [], []
    for pkl_path, ep_len in zip(pkls, lengths):
        records = _episode_records(pkl_path)
        if len(records) != ep_len:
            raise RuntimeError(
                f"{pkl_path.name} has {len(records)} records but metadata says {ep_len}"
            )
        for row in records:
            states.append(_state_vector(row["obs"], obs_is_dict))
            plan = np.asarray(row["action"], dtype=np.float32)
            actions.append(plan[exec_idx])
    return (
        np.stack(states, axis=0).astype(np.float32),
        np.stack(actions, axis=0).astype(np.float32),
    )


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
    parser.add_argument("--runner-target", default=None,
                        help="Override cfg.task.env_runner._target_ (e.g. the robocasa365 "
                             "runner for the released multitask checkpoint, whose embedded "
                             "target points at the benchmark fork's runner).")
    parser.add_argument("--runner-set", action="append", default=[],
                        help="key=value override applied to the env runner cfg (repeatable), "
                             "e.g. --runner-set env_name=PickPlaceCounterToCabinet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for embedding extraction.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    output_dir = pathlib.Path(args.output_dir)
    for sub in ("episodes", "media", "policy_embeddings", "observations", "actions.npz"):
        if (output_dir / sub).exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{output_dir / sub} already exists; pass --overwrite to replace"
                )
            if (output_dir / sub).is_dir():
                shutil.rmtree(output_dir / sub)
            else:
                (output_dir / sub).unlink()
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
    if isinstance(policy, (DiffusionUnetHybridImagePolicy, DiffusionTransformerHybridImagePolicy)):
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
    if args.runner_target is not None:
        runner_cfg._target_ = args.runner_target
    for override in args.runner_set:
        key, _, value = override.partition("=")
        from omegaconf import OmegaConf as _OC
        _OC.update(runner_cfg, key, yaml.safe_load(value), force_add=True)

    env_runner = hydra.utils.instantiate(
        runner_cfg, output_dir=str(output_dir), save_episodes=True
    )
    logger.info("running %d episodes (seed=%d)", args.num_episodes, args.seed)
    import inspect

    if "policy_cfg" in inspect.signature(env_runner.run).parameters:
        # The robocasa runner re-instantiates the policy in eval subprocesses.
        runner_log = env_runner.run(policy, policy_cfg=cfg.policy)
    else:
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

    n_latency_steps = int(runner_cfg.get("n_latency_steps", 0) or 0)
    states, exec_actions = extract_states_and_actions(
        policy, episodes_dir, lengths, n_latency_steps
    )
    if states.shape[0] != sum(lengths):
        raise RuntimeError(
            f"{states.shape[0]} states but episode lengths sum to {sum(lengths)}"
        )
    obs_dir = output_dir / "observations"
    obs_dir.mkdir(exist_ok=True)
    np.savez_compressed(obs_dir / "state.npz", observations=states)
    np.savez_compressed(output_dir / "actions.npz", actions=exec_actions)
    logger.info("wrote state %s and executed actions %s", states.shape, exec_actions.shape)

    _write_canonical_metadata(episodes_dir, lengths, successes)
    _rename_media(output_dir / "media")
    logger.info(
        "done: %d episodes, %d/%d successes, embeddings %s",
        len(lengths), sum(successes), len(successes), embeddings.shape,
    )


if __name__ == "__main__":
    main()
