"""Observation assembly for DROID diffusion-policy inference and extraction.

Maps DROID observation sources (the live env, robo_dataset stores) onto a
checkpoint's ``shape_meta`` obs keys, applying the same image preprocessing
the droid training data used (cv2 INTER_AREA resize to the target resolution,
uint8 HWC → float32 CHW in [0, 1] — see robo_dataset's DroidStore and
``diffusion_policy/dataset/robo_store_dataset.py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Naming differs between the droid stores / training data (left column) and
# the live env (values include the env aliases): a checkpoint key resolves to
# the first available alias.
KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hand_camera_image": ("hand_camera_image", "wrist_image"),
    "wrist_image": ("wrist_image", "hand_camera_image"),
    "joint_positions": ("joint_positions", "joint_position"),
    "joint_position": ("joint_position", "joint_positions"),
}


@dataclass(frozen=True)
class ObsSpec:
    """Obs/action layout of a diffusion checkpoint, read from its cfg."""

    rgb_shapes: Dict[str, Tuple[int, int, int]]  # key -> (C, H, W)
    lowdim_dims: Dict[str, int]  # key -> D
    n_obs_steps: int
    horizon: int
    action_dim: int

    @property
    def obs_keys(self) -> List[str]:
        return list(self.rgb_shapes) + list(self.lowdim_dims)


def obs_spec_from_shape_meta(
    shape_meta: Dict, n_obs_steps: int, horizon: int
) -> ObsSpec:
    """Build an ObsSpec from a shape_meta dict (cfg or JSON form)."""
    rgb_shapes: Dict[str, Tuple[int, int, int]] = {}
    lowdim_dims: Dict[str, int] = {}
    for key, attr in shape_meta["obs"].items():
        shape = tuple(int(x) for x in attr["shape"])
        if dict(attr).get("type", "low_dim") == "rgb":
            if len(shape) != 3:
                raise ValueError(f"rgb obs {key!r} has non-CHW shape {shape}")
            rgb_shapes[key] = shape
        else:
            if len(shape) != 1:
                raise ValueError(f"low_dim obs {key!r} has non-1D shape {shape}")
            lowdim_dims[key] = shape[0]
    return ObsSpec(
        rgb_shapes=rgb_shapes,
        lowdim_dims=lowdim_dims,
        n_obs_steps=int(n_obs_steps),
        horizon=int(horizon),
        action_dim=int(shape_meta["action"]["shape"][0]),
    )


def obs_spec_from_cfg(cfg) -> ObsSpec:
    """Build an ObsSpec from a checkpoint cfg (``cfg.task.shape_meta``)."""
    shape_meta = cfg.task.shape_meta if "task" in cfg else cfg.shape_meta
    return obs_spec_from_shape_meta(
        shape_meta, n_obs_steps=cfg.n_obs_steps, horizon=cfg.horizon
    )


def resolve_key_map(
    spec: ObsSpec,
    available_keys: Iterable[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Map each checkpoint obs key to a source key.

    Tries the override, then the key itself, then known aliases. Raises with
    the available key list when a checkpoint key cannot be resolved.
    """
    available = set(available_keys)
    overrides = overrides or {}
    key_map: Dict[str, str] = {}
    for key in spec.obs_keys:
        candidates = (overrides.get(key),) if key in overrides else (
            key, *KEY_ALIASES.get(key, ()))
        source = next((c for c in candidates if c in available), None)
        if source is None:
            raise KeyError(
                f"Checkpoint obs key {key!r} not found (tried {candidates}); "
                f"available keys: {sorted(available)}. Map it explicitly via "
                f"--obs-keys {key}=<source_key>."
            )
        key_map[key] = source
    return key_map


def resize_frames(frames: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Resize (T, H, W, 3) or (H, W, 3) uint8 frames to (h, w), INTER_AREA.

    INTER_AREA matches the droid → training-store conversion (DroidStore).
    """
    h, w = hw
    if frames.shape[-3:-1] == (h, w):
        return frames
    if frames.ndim == 3:
        return cv2.resize(frames, (w, h), interpolation=cv2.INTER_AREA)
    return np.stack(
        [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]
    )


def frames_to_policy_format(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 → (T, 3, H, W) float32 in [0, 1]."""
    return np.moveaxis(frames.astype(np.float32) / 255.0, -1, 1)


class ObsWindowBuffer:
    """Streaming obs-window assembly for live inference.

    Push raw env observations; ``window()`` returns the checkpoint-format obs
    dict ``{key: (To, ...)}`` with rgb keys as float32 CHW in [0, 1], front-
    padded by repeating the oldest frame (matching the training sampler's
    pad_before behaviour at episode starts).
    """

    def __init__(self, spec: ObsSpec, key_map: Dict[str, str]) -> None:
        self._spec = spec
        self._key_map = key_map
        self._frames: deque = deque(maxlen=spec.n_obs_steps)

    def reset(self) -> None:
        self._frames.clear()

    def push(self, env_obs: Dict[str, np.ndarray]) -> None:
        frame: Dict[str, np.ndarray] = {}
        for key, (c, h, w) in self._spec.rgb_shapes.items():
            img = np.asarray(env_obs[self._key_map[key]])
            frame[key] = frames_to_policy_format(
                resize_frames(img.astype(np.uint8), (h, w))[None]
            )[0]
        for key, dim in self._spec.lowdim_dims.items():
            vec = np.atleast_1d(
                np.asarray(env_obs[self._key_map[key]], dtype=np.float32)
            )
            if vec.shape[0] != dim:
                raise ValueError(
                    f"obs {key!r}: got dim {vec.shape[0]}, checkpoint expects {dim}"
                )
            frame[key] = vec
        self._frames.append(frame)

    def window(self) -> Dict[str, np.ndarray]:
        if not self._frames:
            raise RuntimeError("push() an observation before window()")
        frames = list(self._frames)
        while len(frames) < self._spec.n_obs_steps:
            frames.insert(0, frames[0])
        return {
            key: np.stack([f[key] for f in frames]) for key in frames[0]
        }


def episode_obs_windows(
    ep_arrays: Dict[str, np.ndarray],
    spec: ObsSpec,
    timesteps: Sequence[int],
) -> List[Dict[str, np.ndarray]]:
    """Per-timestep obs windows from full-episode checkpoint-keyed arrays.

    ``ep_arrays``: rgb keys as (T, H, W, 3) uint8 at the checkpoint
    resolution, lowdim keys as (T, D) float32. For each timestep t, the
    window covers frames [t - To + 1, t], front-padded by repeating frame 0.
    """
    To = spec.n_obs_steps
    out: List[Dict[str, np.ndarray]] = []
    for t in timesteps:
        idx = np.clip(np.arange(t - To + 1, t + 1), 0, None)
        obs: Dict[str, np.ndarray] = {}
        for key in spec.rgb_shapes:
            obs[key] = frames_to_policy_format(ep_arrays[key][idx])
        for key in spec.lowdim_dims:
            obs[key] = ep_arrays[key][idx].astype(np.float32)
        out.append(obs)
    return out


def episode_action_plans(
    actions: np.ndarray, horizon: int, timesteps: Sequence[int]
) -> np.ndarray:
    """(B, horizon, Da) action plans: actions[t : t+horizon], end-padded by
    repeating the last action (matching the training sampler's pad_after)."""
    T = actions.shape[0]
    idx = np.minimum(
        np.asarray(timesteps)[:, None] + np.arange(horizon)[None, :], T - 1
    )
    return actions[idx].astype(np.float32)
