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

Embedding layer names follow ``{bottleneck|decoder|encoder}[_{plan|plan8|exec}][_t{T}]``:

    hook        bottleneck = model.mid_modules[-1], decoder = up_modules[-1][1],
                encoder = down_modules[-1][1]; output is global-avg-pooled.
    action      UNet action input: omitted = random noise, plan = full predicted
                plan, exec = action[0] tiled, plan8 = first 8 steps zero-padded.
    t           diffusion timestep: omitted = average over all inference steps,
                _t{T} = single timestep T (0 = final denoising step).

The default ``bottleneck_plan_t0`` conditions the UNet bottleneck on the actual
predicted plan at the final denoising step.

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
import re
import shutil
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hydra
import numpy as np
import torch
import tqdm
import yaml

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.trak_util import (
    get_best_checkpoint,
    get_policy_from_checkpoint,
)
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
    DiffusionUnetHybridImagePolicy,
)
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

logger = logging.getLogger("eval_save_rollouts")

DEFAULT_LAYER = "bottleneck_plan_t0"

_LAYER_RE = re.compile(
    r"^(?P<hook>bottleneck|decoder|encoder)"
    r"(?:_(?P<action>plan8|plan|exec))?"
    r"(?:_t(?P<t>\d+))?$"
)


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
# Policy-embedding extraction (diffusion U-Net, low-dim policies)
# ---------------------------------------------------------------------------

def _parse_layer(layer: str) -> Tuple[str, Optional[str], Optional[int]]:
    """Parse a layer name into (hook, action, t_single)."""
    m = _LAYER_RE.match(layer)
    if m is None:
        raise ValueError(
            f"Unknown layer {layer!r}. Format: "
            f"{{bottleneck|decoder|encoder}}[_{{plan|plan8|exec}}][_t{{T}}]"
        )
    t = m.group("t")
    return m.group("hook"), m.group("action"), None if t is None else int(t)


def _get_hook_module(model, hook: str):
    """Return the U-Net module to hook for a given hook name."""
    if hook == "bottleneck":
        return model.mid_modules[-1]
    if hook == "decoder":
        return model.up_modules[-1][1]
    if hook == "encoder":
        return model.down_modules[-1][1]
    raise ValueError(f"Unknown hook: {hook}")


def _build_noise_schedule(n_infer: int, device: torch.device):
    betas = torch.linspace(1e-4, 0.02, n_infer, device=device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    return alphas_cumprod.sqrt(), (1.0 - alphas_cumprod).sqrt()


def _build_action_batch(
    action_arr: Optional[np.ndarray],
    action_type: Optional[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return clean actions (B, horizon, action_dim), or None for random noise."""
    if action_type is None:
        return None
    assert action_arr is not None
    B, horizon, action_dim = action_arr.shape
    if action_type == "plan":
        clean = action_arr
    elif action_type == "exec":
        clean = np.tile(action_arr[:, 0:1, :], (1, horizon, 1))
    elif action_type == "plan8":
        clean = np.concatenate(
            [action_arr[:, :8, :], np.zeros((B, horizon - 8, action_dim), dtype=np.float32)],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    return torch.as_tensor(clean, dtype=torch.float32, device=device)


def _extract_batch(
    policy,
    global_cond_b: torch.Tensor,
    action_batch: Optional[np.ndarray],
    action_type: Optional[str],
    hook_name: str,
    noise_step_indices: torch.Tensor,
    sqrt_alpha: torch.Tensor,
    sqrt_one_minus_alpha: torch.Tensor,
    device: torch.device,
    batch_seed: int,
) -> np.ndarray:
    """Return a (B, D) embedding for one batch of rollout timesteps.

    ``global_cond_b`` is the (B, cond_dim) UNet conditioning: flattened raw
    observations for low-dim policies, obs-encoder features for image
    policies — mirroring each policy's own conditioning path.
    """
    model = policy.model
    B = len(global_cond_b)
    N = len(noise_step_indices)

    global_cond = global_cond_b.to(device).repeat_interleave(N, dim=0)

    rng = torch.Generator(device=device)
    rng.manual_seed(batch_seed)
    base_noise = torch.randn(
        B, policy.horizon, policy.action_dim, generator=rng, device=device
    ).repeat_interleave(N, dim=0)

    t_indices = noise_step_indices.repeat(B)
    a = sqrt_alpha[t_indices].view(B * N, 1, 1)
    s = sqrt_one_minus_alpha[t_indices].view(B * N, 1, 1)

    clean_actions = _build_action_batch(action_batch, action_type, device)
    if clean_actions is not None:
        noisy_actions = a * clean_actions.repeat_interleave(N, dim=0) + s * base_noise
    else:
        noisy_actions = s * base_noise

    captured = []
    hook_module = _get_hook_module(model, hook_name)

    def _hook(module, inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        captured.append(feat.detach().float().mean(dim=-1).cpu())  # global avg pool

    handle = hook_module.register_forward_hook(_hook)
    with torch.no_grad():
        model(noisy_actions, t_indices, global_cond=global_cond)
    handle.remove()

    return captured[0].reshape(B, N, -1).mean(dim=1).numpy().astype(np.float32)


def _episode_records(pkl_path: pathlib.Path) -> list:
    """Load one episode's per-inference-step records (DataFrame or list of dicts)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if hasattr(data, "iloc"):
        return [data.iloc[t] for t in range(len(data))]
    return list(data)


def _lowdim_global_cond(policy, obs_batch: list, device: torch.device) -> torch.Tensor:
    """Flattened raw obs window, as in the low-dim policy's conditioning."""
    obs_arr = np.stack(obs_batch, axis=0)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    return obs_t.reshape(len(obs_batch), -1)


def _image_global_cond(policy, obs_batch: list, device: torch.device) -> torch.Tensor:
    """Obs-encoder features, mirroring DiffusionUnetHybridImagePolicy.predict_action."""
    keys = obs_batch[0].keys()
    obs_dict = {
        k: torch.as_tensor(
            np.stack([o[k] for o in obs_batch]), dtype=torch.float32, device=device
        )
        for k in keys
    }
    with torch.no_grad():
        obs_dict = policy._center_crop_external(obs_dict)
        nobs = policy.normalizer.normalize(obs_dict)
        To = policy.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = policy.obs_encoder(this_nobs)
        return nobs_features.reshape(len(obs_batch), -1)


def extract_embeddings(
    policy,
    episodes_dir: pathlib.Path,
    layer: str,
    lengths: List[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract one embedding per recorded inference step across all episodes."""
    if isinstance(policy, DiffusionUnetLowdimPolicy):
        make_global_cond = _lowdim_global_cond
        obs_is_dict = False
    elif isinstance(policy, DiffusionUnetHybridImagePolicy):
        make_global_cond = _image_global_cond
        obs_is_dict = True
    else:
        raise NotImplementedError(
            f"Policy embeddings are supported for DiffusionUnetLowdimPolicy and "
            f"DiffusionUnetHybridImagePolicy, got {type(policy).__name__}."
        )
    hook_name, action_type, t_single = _parse_layer(layer)
    needs_actions = action_type is not None

    n_infer = policy.num_inference_steps
    if t_single is not None:
        noise_step_indices = torch.tensor([t_single], dtype=torch.long, device=device)
    else:
        noise_step_indices = torch.linspace(
            0, n_infer - 1, n_infer, dtype=torch.long, device=device
        )

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
            if obs_is_dict:
                all_obs.append(
                    {k: np.asarray(v, dtype=np.float32) for k, v in row["obs"].items()}
                )
            else:
                obs = np.asarray(row["obs"], dtype=np.float32)
                all_obs.append(obs[np.newaxis, :] if obs.ndim == 1 else obs)
            if needs_actions:
                all_actions.append(np.asarray(row["action"], dtype=np.float32))

    action_arr = np.stack(all_actions, axis=0) if needs_actions else None
    n_total = len(all_obs)
    logger.info("extracting layer=%s for %d timesteps", layer, n_total)

    sqrt_alpha, sqrt_one_minus_alpha = _build_noise_schedule(n_infer, device)
    results = []
    for start in tqdm.tqdm(range(0, n_total, batch_size), desc="embeddings"):
        global_cond_b = make_global_cond(
            policy, all_obs[start:start + batch_size], device
        )
        results.append(
            _extract_batch(
                policy,
                global_cond_b,
                action_arr[start:start + batch_size] if needs_actions else None,
                action_type,
                hook_name,
                noise_step_indices,
                sqrt_alpha,
                sqrt_one_minus_alpha,
                device,
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
