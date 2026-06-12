"""Offline policy-embedding extraction over a robo_dataset TrajectoryStore.

For every recorded timestep of every episode, build the obs window the policy
would have seen (images resized to the checkpoint's shape_meta resolution,
uint8 HWC → float CHW in [0, 1], lowdim as-is) plus the action plan
(``actions[t : t+horizon]``, end-padded), and extract one embedding via the
same machinery ``scripts/eval_save_rollouts.py`` uses
(``diffusion_policy.common.policy_embeddings``).

Supported stores: anything ``robo_dataset.store_factory.open_store`` opens
(VideoStore dirs, robomimic-format HDF5, raw DROID session trees, ...).
"""

from __future__ import annotations

import logging
import pathlib
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm

from diffusion_policy.common.policy_embeddings import PolicyEmbeddingExtractor
from diffusion_policy.real_world.droid.obs_processing import (
    ObsSpec,
    episode_action_plans,
    episode_obs_windows,
    obs_spec_from_cfg,
    resize_frames,
    resolve_key_map,
)
from diffusion_policy.real_world.droid.rollout_layout import (
    media_path,
    write_episode_mp4,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store opening / inspection
# ---------------------------------------------------------------------------

def open_droid_store(path: str, **droid_kwargs):
    """Open a trajectory store: VideoStore dir, robomimic HDF5, or raw DROID.

    Raw DROID session trees (dirs containing ``trajectory.h5`` files) are not
    auto-detected by robo_dataset; ``droid_kwargs`` (cameras, camera_source,
    image_size, ...) are forwarded to DroidStore for that case.
    """
    from robo_dataset.store_factory import detect_format, open_store

    p = pathlib.Path(path)
    try:
        fmt = detect_format(str(p))
    except ValueError:
        if p.is_dir() and any(p.rglob("trajectory.h5")):
            logger.info("opening raw DROID session tree at %s", p)
            return open_store(str(p), format="droid", **droid_kwargs)
        raise
    logger.info("opening %s store at %s", fmt, p)
    return open_store(str(p), format=fmt)


def episode_successes(store) -> List[Optional[bool]]:
    """Per-episode success flags, or None when the store has no signal.

    Uses the ``reward`` key (droid convention: success flag on the last step)
    or, for robomimic HDF5 files, the per-demo ``rewards`` dataset.
    """
    n = store.n_episodes
    if "reward" in store.keys:
        return [
            bool(
                np.asarray(
                    store.get_sequence("reward", int(store.episode_ends[i]) - 1,
                                       int(store.episode_ends[i]))
                ).reshape(-1)[-1] > 0.5
            )
            for i in range(n)
        ]

    from robo_dataset.backends.hdf5 import RobomimicHDF5Store

    if isinstance(store, RobomimicHDF5Store):
        import h5py

        out: List[Optional[bool]] = []
        with h5py.File(store.path, "r") as f:
            for i in range(n):
                demo = f[f"data/demo_{i}"]
                if "rewards" in demo:
                    out.append(bool(demo["rewards"][-1] > 0.5))
                else:
                    out.append(None)
        return out

    return [None] * n


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _load_episode_arrays(
    store,
    ep_idx: int,
    spec: ObsSpec,
    key_map: Dict[str, str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Episode arrays keyed by checkpoint obs key, images at target resolution."""
    start = store.episode_start(ep_idx)
    end = int(store.episode_ends[ep_idx])
    arrays: Dict[str, np.ndarray] = {}
    for key, (c, h, w) in spec.rgb_shapes.items():
        frames = np.asarray(store.get_sequence(key_map[key], start, end))
        if frames.dtype != np.uint8:
            raise ValueError(
                f"rgb key {key_map[key]!r} has dtype {frames.dtype}, expected uint8"
            )
        arrays[key] = resize_frames(frames, (h, w))
    for key in spec.lowdim_dims:
        arrays[key] = np.asarray(
            store.get_sequence(key_map[key], start, end), dtype=np.float32
        )
    actions = np.asarray(
        store.get_sequence("action", start, end), dtype=np.float32
    )
    return arrays, actions


def extract_store_embeddings(
    policy,
    cfg,
    store,
    layer: str,
    device: torch.device,
    batch_size: int = 64,
    obs_key_overrides: Optional[Dict[str, str]] = None,
    episode_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Extract one embedding per recorded timestep.

    Returns ``(embeddings (total_steps, d) float32, episode lengths)`` with
    rows ordered episode-by-episode, timestep-by-timestep.
    """
    extractor = PolicyEmbeddingExtractor(policy, layer, device)
    if not extractor.obs_is_dict:
        raise NotImplementedError(
            "store extraction currently supports image policies "
            "(DiffusionUnetHybridImagePolicy); the droid checkpoints are image-based."
        )

    spec = obs_spec_from_cfg(cfg)
    key_map = resolve_key_map(spec, store.keys, obs_key_overrides)
    logger.info(
        "obs map (checkpoint <- store): %s | n_obs_steps=%d horizon=%d",
        key_map, spec.n_obs_steps, spec.horizon,
    )

    action_dim = int(store.frame_shape("action")[0])
    if action_dim != policy.action_dim:
        raise ValueError(
            f"store action dim {action_dim} != checkpoint action dim "
            f"{policy.action_dim}"
        )

    if episode_indices is None:
        episode_indices = range(store.n_episodes)

    lengths: List[int] = []
    chunks: List[np.ndarray] = []
    global_step = 0
    for ep_idx in episode_indices:
        arrays, actions = _load_episode_arrays(store, ep_idx, spec, key_map)
        T = actions.shape[0]
        for start in tqdm.tqdm(
            range(0, T, batch_size), desc=f"ep{ep_idx:04d}", leave=False
        ):
            timesteps = range(start, min(start + batch_size, T))
            obs_batch = episode_obs_windows(arrays, spec, timesteps)
            action_batch = (
                episode_action_plans(actions, spec.horizon, timesteps)
                if extractor.needs_actions
                else None
            )
            chunks.append(
                extractor.extract(
                    obs_batch, action_batch, batch_seed=global_step + start
                )
            )
        lengths.append(T)
        global_step += T
        logger.info("episode %d: %d timesteps embedded", ep_idx, T)

    embeddings = np.concatenate(chunks, axis=0)
    return embeddings, lengths


# ---------------------------------------------------------------------------
# Media export
# ---------------------------------------------------------------------------

def export_media(
    store,
    output_dir: pathlib.Path,
    camera_key: str,
    successes: Sequence[Optional[bool]],
    episode_indices: Sequence[int],
    fps: Optional[float] = None,
) -> int:
    """Write media/ep{i:04d}_{tag}.mp4 per episode from a store camera key.

    VideoStore episodes are copied byte-for-byte from the per-episode MP4s
    (free); other stores are re-encoded with PyAV.
    """
    if camera_key not in store.keys:
        raise KeyError(
            f"camera key {camera_key!r} not in store (have: {sorted(store.keys)})"
        )
    fps = fps if fps is not None else float(getattr(store, "_fps", 10.0))
    n_written = 0
    for out_idx, ep_idx in enumerate(episode_indices):
        dst = media_path(output_dir, out_idx, successes[out_idx])
        src = (
            pathlib.Path(store.path) / "videos" / camera_key
            / f"episode_{ep_idx:06d}.mp4"
        ) if getattr(store, "path", None) else None
        if src is not None and src.is_file():
            shutil.copyfile(src, dst)
        else:
            start = store.episode_start(ep_idx)
            end = int(store.episode_ends[ep_idx])
            frames = np.asarray(store.get_sequence(camera_key, start, end))
            write_episode_mp4(dst, frames.astype(np.uint8), fps=fps)
        n_written += 1
    return n_written
