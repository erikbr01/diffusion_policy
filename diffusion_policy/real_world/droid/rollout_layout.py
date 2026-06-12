"""Writers for the policy_graphs canonical rollout layout.

::

    <output_dir>/
        episodes/metadata.yaml          episodes: [{index, length, success}]
        policy_embeddings/<layer>.npz   key "embeddings", (total_steps, d) float32
        media/ep{idx:04d}_{success|failure|unknown}.mp4

Success may be unknown for real-robot episodes — ``success: null`` in the
metadata and an ``unknown`` media tag.
"""

from __future__ import annotations

import fractions
import pathlib
from typing import List, Optional, Sequence

import numpy as np
import yaml


def success_tag(success: Optional[bool]) -> str:
    if success is None:
        return "unknown"
    return "success" if success else "failure"


def write_metadata(
    output_dir: pathlib.Path,
    lengths: Sequence[int],
    successes: Sequence[Optional[bool]],
) -> pathlib.Path:
    """Write episodes/metadata.yaml; success entries may be None (null)."""
    if len(lengths) != len(successes):
        raise ValueError("lengths and successes must have equal length")
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "episodes": [
            {
                "index": i,
                "length": int(n),
                "success": None if s is None else bool(s),
            }
            for i, (n, s) in enumerate(zip(lengths, successes))
        ]
    }
    path = episodes_dir / "metadata.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return path


def write_embeddings(
    output_dir: pathlib.Path, layer: str, embeddings: np.ndarray
) -> pathlib.Path:
    emb_dir = output_dir / "policy_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    path = emb_dir / f"{layer}.npz"
    np.savez_compressed(path, embeddings=embeddings.astype(np.float32))
    return path


def media_path(
    output_dir: pathlib.Path, episode_idx: int, success: Optional[bool]
) -> pathlib.Path:
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    return media_dir / f"ep{episode_idx:04d}_{success_tag(success)}.mp4"


def write_episode_mp4(
    path: pathlib.Path,
    frames: np.ndarray,
    fps: float = 10.0,
    crf: int = 23,
) -> pathlib.Path:
    """Encode (T, H, W, 3) uint8 RGB frames to H.264 MP4 via PyAV."""
    import av

    frames = np.ascontiguousarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T, H, W, 3) frames, got {frames.shape}")
    T, H, W = frames.shape[:3]
    # yuv420p needs even dimensions; crop a pixel if necessary.
    H -= H % 2
    W -= W % 2

    container = av.open(str(path), "w")
    rate = fractions.Fraction(fps).limit_denominator(1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "fast"}
    for arr in frames[:, :H, :W]:
        for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path
