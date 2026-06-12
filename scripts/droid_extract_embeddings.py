"""Offline policy-embedding extraction for recorded DROID episodes.

Reads recorded episodes from a robo_dataset store (VideoStore directory,
robomimic-format HDF5, or a raw DROID session tree), extracts one policy
embedding per recorded timestep with a trained droid diffusion checkpoint,
and writes the canonical policy_graphs rollout layout::

    <output_dir>/
        episodes/metadata.yaml          episodes: [{index, length, success}]
        policy_embeddings/<layer>.npz   (total_steps, d) float32
        media/ep{idx:04d}_{tag}.mp4     (with --camera-key)

Usage (in this repo's venv)::

    python scripts/droid_extract_embeddings.py \\
        --store /path/to/kendama.video \\
        --checkpoint data/outputs/train/<run>/checkpoints/latest.ckpt \\
        --output-dir /path/to/rollouts --camera-key exterior_image_1_left

Obs keys are mapped from the checkpoint's shape_meta to store keys
automatically; override with ``--obs-key <ckpt_key>=<store_key>``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from diffusion_policy.common.trak_util import get_policy_from_checkpoint
from diffusion_policy.real_world.droid.rollout_layout import (
    write_embeddings,
    write_metadata,
)
from diffusion_policy.real_world.droid.store_embeddings import (
    episode_successes,
    export_media,
    extract_store_embeddings,
    open_droid_store,
)

logger = logging.getLogger("droid_extract_embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True,
                        help="VideoStore dir, robomimic HDF5, or raw DROID session tree.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", default="bottleneck_plan_t0")
    parser.add_argument("--camera-key", default=None,
                        help="Store camera key to export per-episode mp4s from.")
    parser.add_argument("--obs-key", action="append", default=[],
                        metavar="CKPT_KEY=STORE_KEY",
                        help="Override the checkpoint->store obs-key mapping.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
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

    overrides = {}
    for spec in args.obs_key:
        ckpt_key, _, store_key = spec.partition("=")
        if not store_key:
            raise ValueError(f"--obs-key expects CKPT_KEY=STORE_KEY, got {spec!r}")
        overrides[ckpt_key] = store_key

    device = torch.device(args.device)
    logger.info("loading checkpoint %s", args.checkpoint)
    policy, cfg = get_policy_from_checkpoint(pathlib.Path(args.checkpoint), device=device)
    store = open_droid_store(args.store)

    n_episodes = store.n_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    episode_indices = list(range(n_episodes))

    embeddings, lengths = extract_store_embeddings(
        policy,
        cfg,
        store,
        layer=args.layer,
        device=device,
        batch_size=args.batch_size,
        obs_key_overrides=overrides or None,
        episode_indices=episode_indices,
    )
    successes = [episode_successes(store)[i] for i in episode_indices]

    write_metadata(output_dir, lengths, successes)
    write_embeddings(output_dir, args.layer, embeddings)
    if args.camera_key:
        n_videos = export_media(
            store, output_dir, args.camera_key, successes, episode_indices
        )
        logger.info("wrote %d videos", n_videos)
    logger.info(
        "done: %d episodes, embeddings %s -> %s",
        len(lengths), embeddings.shape, output_dir,
    )


if __name__ == "__main__":
    main()
