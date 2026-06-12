"""Collect DROID real-robot rollouts in the canonical policy_graphs layout.

Runs on the robot workstation (requires the DROID stack for real rollouts;
``--dry-run`` exercises the full loop with synthetic observations anywhere).
Rolls out a diffusion checkpoint served locally or over HTTP/WebSocket,
records each episode as a DROID-format ``trajectory.h5``, and writes::

    <output_dir>/
        episodes/metadata.yaml            episodes: [{index, length, success}]
        recordings/ep{idx:04d}/trajectory.h5
        policy_embeddings/<layer>.npz     (with --extract-embeddings, local ckpt)

Success flags are unknown at collection time (annotate metadata.yaml after
the session, or rely on downstream tooling). Embeddings can also be added
later with ``scripts/droid_extract_embeddings.py --store <output_dir>/recordings``.

Usage::

    python scripts/droid_collect_rollouts.py \\
        --policy http://localhost:8000 --num-episodes 10 \\
        --output-dir /data/rollouts/kendama_session \\
        --instruction "pick up the kendama"
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion_policy.real_world.droid.env import DROIDInferenceEnv
from diffusion_policy.real_world.droid.policy_client import (
    HttpPolicy,
    LocalPolicy,
    WebSocketPolicy,
)
from diffusion_policy.real_world.droid.rollout_layout import write_metadata
from diffusion_policy.real_world.droid.runner import (
    DROIDInferenceRunner,
    InferenceMode,
)

logger = logging.getLogger("droid_collect_rollouts")


def _build_policy(spec: str, device: str):
    """``http://...`` | ``ws://...`` | a local checkpoint path."""
    if spec.startswith("http://") or spec.startswith("https://"):
        return HttpPolicy(spec)
    if spec.startswith("ws://") or spec.startswith("wss://"):
        return WebSocketPolicy(spec)
    return LocalPolicy(pathlib.Path(spec), device=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", required=True,
                        help="http(s)://host:port, ws(s)://host:port, or a local .ckpt path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--mode", choices=[m.value for m in InferenceMode],
                        default=InferenceMode.ASYNC_CHUNK.value)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--max-timesteps", type=int, default=600)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--extract-embeddings", action="store_true",
                        help="Extract embeddings after collection (local checkpoint only).")
    parser.add_argument("--layer", default="bottleneck_plan_t0")
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic observations instead of robot hardware.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    output_dir = pathlib.Path(args.output_dir)
    recordings = output_dir / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)

    env = DROIDInferenceEnv(dry_run=args.dry_run)
    policy = _build_policy(args.policy, args.device)
    runner = DROIDInferenceRunner(
        env,
        policy,
        mode=InferenceMode(args.mode),
        open_loop_horizon=args.open_loop_horizon,
        max_timesteps=args.max_timesteps,
        control_hz=args.control_hz,
    )

    lengths, successes = [], []
    for ep in range(args.num_episodes):
        record = runner.run_episode(instruction=args.instruction, episode_idx=ep)
        episode_data = env.pop_episode_data()
        env.save_droid_trajectory(
            recordings / f"ep{ep:04d}" / "trajectory.h5", episode_data, record.success
        )
        lengths.append(record.n_steps)
        successes.append(record.success)
        logger.info(
            "episode %d: %d steps, %d chunks, %.1f ms avg latency",
            ep, record.n_steps, record.n_new_chunks, record.avg_chunk_latency_ms,
        )
    write_metadata(output_dir, lengths, successes)

    if args.extract_embeddings:
        if not isinstance(policy, LocalPolicy):
            raise SystemExit(
                "--extract-embeddings needs a local checkpoint; for served "
                "policies run scripts/droid_extract_embeddings.py afterwards."
            )
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).parent / "droid_extract_embeddings.py"),
                "--store", str(recordings),
                "--checkpoint", args.policy,
                "--output-dir", str(output_dir),
                "--layer", args.layer,
                "--overwrite",
            ],
            check=True,
        )
    logger.info("session written to %s", output_dir)


if __name__ == "__main__":
    main()
