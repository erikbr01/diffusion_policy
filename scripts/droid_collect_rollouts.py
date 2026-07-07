"""Collect DROID real-robot rollouts in the canonical policy_graphs layout.

Runs on the robot workstation in ``zed_env_spacemouse`` (requires the DROID
stack; ``--dry-run`` exercises the full loop anywhere with synthetic obs).
Talks to the diffusion-policy HTTP server running on the GPU box.

Output layout::

    <output_dir>/
        episodes/metadata.yaml                  per-episode index, length, success
        recordings/ep_YYYYMMDDTHHMMSS/
            trajectory.h5                       robot state + actions (DROID format)
            wrist_image.mp4                     CRF-18 H.264 at control Hz
            exterior_image_1_left.mp4
            embeddings_<layer>.npz              (n_steps, D) float32, this episode only

Embeddings are collected online by the policy server (``--embed-layer``).
MP4s are saved for offline re-extraction of any other layer via
``scripts/droid_extract_embeddings.py``.

Usage::

    # GPU box
    python -m diffusion_policy.real_world.droid.policy_server \\
        --checkpoint /path/to/latest.ckpt --device cuda:0 \\
        --host 0.0.0.0 --port 5001 --embed-layer bottleneck_plan_t0

    # Robot workstation
    conda activate zed_env_spacemouse
    python scripts/droid_collect_rollouts.py \\
        --policy http://<gpu_box_ip>:5001 --num-episodes 20 \\
        --output-dir /data/rollouts/set_table_jun22 \\
        --instruction "prepare plate - milk right" \\
        --mode async_chunk --control-hz 15 --max-timesteps 600 \\
        --embed-layer bottleneck_plan_t0 --max-joint-velocity 1.0
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import logging
import os
import pathlib
import select
import signal
import sys
import termios
import threading
import tty

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from diffusion_policy.real_world.droid.env import DROIDInferenceEnv
from diffusion_policy.real_world.droid.policy_client import (
    HttpPolicy,
    LocalPolicy,
    WebSocketPolicy,
)

from diffusion_policy.real_world.droid.runner import (
    DROIDInferenceRunner,
    InferenceMode,
)

logger = logging.getLogger("droid_collect_rollouts")


_SAVED_TERM_ATTRS = None


def _save_terminal() -> None:
    global _SAVED_TERM_ATTRS
    if sys.stdin.isatty():
        try:
            _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
            atexit.register(_restore_terminal)
        except Exception:
            pass


def _restore_terminal() -> None:
    if _SAVED_TERM_ATTRS is not None:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
        except Exception:
            pass


def _quit() -> None:
    """Restore terminal then send SIGINT to the main thread."""
    _restore_terminal()
    os.kill(os.getpid(), signal.SIGINT)


def _wait_for_space(prompt: str) -> None:
    """Block until the user presses space. q / Ctrl-C quits. Terminal in raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt, flush=True)
    try:
        tty.setraw(fd)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch == " ":
                    return
                if ch in ("q", "\x03"):  # q or Ctrl-C
                    _quit()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _space_stop_listener(stop_event: threading.Event) -> None:
    """Watch for space / q / Ctrl-C in raw mode; exits cleanly when stop_event is set."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                if ch == " ":
                    print("\r[stop requested]", flush=True)
                    stop_event.set()
                    return
                if ch in ("q", "\x03"):  # q or Ctrl-C
                    stop_event.set()
                    _quit()
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _prompt_outcome(episode_idx: int) -> tuple:
    """Prompt for episode outcome. Returns (success: bool|None, note: str, retry: bool)."""
    print(f"\nepisode {episode_idx:04d} outcome — [s]uccess / [f]ailure / [u]nknown / [r]etry")
    while True:
        sys.stdout.write("outcome> ")
        sys.stdout.flush()
        line = sys.stdin.readline().strip()
        if not line:
            continue
        key, _, note = line.partition(" ")
        key = key.lower()
        if key == "s":
            return True, note, False
        if key == "f":
            return False, note, False
        if key == "u":
            return None, note, False
        if key == "r":
            return None, "", True
        if key == "q":
            raise SystemExit(0)
        print("  enter s, f, u, or r (q to quit)")


# droid_robot_ik_solver.RobotIKSolver.max_joint_delta — the per-control-step joint
# position delta (rad) a normalized joint_velocity command of magnitude 1.0 produces
# (relative_max_joint_delta = [0.2]*7, see robot_ik_solver.py in the droid stack).
# Used to reconstruct the commanded *target position* from a velocity command so it
# can be plotted in the same units as actual proprio. If that constant changes on the
# robot workstation, update it here too.
_DROID_JOINT_VELOCITY_DELTA_SCALE = 0.2


def _commanded_target_position(actual_positions: np.ndarray, commanded_velocity: np.ndarray) -> np.ndarray:
    """Reconstruct the per-step commanded joint *target position* (rad) from a
    joint_velocity command, mirroring franka/robot.py's create_action_dict:
    joint_delta = joint_velocity_to_delta(command); target = current_position + joint_delta.
    """
    cmd = commanded_velocity.copy()
    peak = np.abs(cmd).max(axis=1, keepdims=True)
    renorm = peak > 1
    cmd = np.where(renorm, cmd / np.where(peak == 0, 1, peak), cmd)
    delta = cmd * _DROID_JOINT_VELOCITY_DELTA_SCALE
    return actual_positions + delta


def _build_joint_tracking_figure(episode_data: list, action_space: str):
    """Build a per-joint commanded-target-vs-actual-position figure for one episode.

    Both traces are plotted in the same units (rad) on one axis:
      - 'joint_position' actions already *are* the absolute commanded target.
      - 'joint_velocity' actions are converted to the implied one-step-ahead
        target position (see ``_commanded_target_position``), then plotted
        one timestep later — i.e. compared against where the joint actually
        was once that command should have taken effect. A persistent gap
        between the two traces is exactly the overshoot/tracking lag we're
        trying to diagnose.

    Returns None (and logs why) if there's nothing sensible to plot.
    """
    if not episode_data or "cartesian" in action_space:
        logger.info("skipping joint-tracking plot (no data or non-joint action space)")
        return None

    import matplotlib.pyplot as plt

    positions = np.stack([step["obs"]["joint_position"] for step in episode_data])  # (T, 7)
    commanded = np.stack([step["action"][:7] for step in episode_data])  # (T, 7)

    if action_space == "joint_position":
        target = commanded
        target_x = np.arange(len(target))
    else:
        target = _commanded_target_position(positions, commanded)[:-1]  # last has no "next" to land on
        target_x = np.arange(1, len(positions))

    fig, axes = plt.subplots(7, 1, figsize=(10, 16), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(positions[:, j], color="tab:blue", label="actual position (rad)")
        ax.plot(target_x, target[:, j], color="tab:orange", linestyle="--", label="commanded target (rad)")
        ax.set_ylabel(f"joint {j}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"commanded target vs. actual position — {action_space}")
    fig.tight_layout()
    return fig


def _build_policy(spec: str, device: str):
    """``http://...`` | ``ws://...`` | a local checkpoint path."""
    if spec.startswith("http://") or spec.startswith("https://"):
        return HttpPolicy(spec)
    if spec.startswith("ws://") or spec.startswith("wss://"):
        return WebSocketPolicy(spec)
    return LocalPolicy(pathlib.Path(spec), device=device)


def main() -> None:
    _save_terminal()
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
    parser.add_argument("--open-loop-horizon", type=int, default=4,
                        help="Steady-state open-loop steps between policy queries.")
    parser.add_argument("--warmup-horizon", type=int, default=16,
                        help="Open-loop steps per chunk during episode warmup.")
    parser.add_argument("--warmup-steps", type=int, default=15,
                        help="Number of steps to stay in warmup mode (default 1s at 15Hz).")
    parser.add_argument("--max-timesteps", type=int, default=600)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--action-space", default="joint_velocity",
                        choices=["joint_velocity", "cartesian_velocity", "joint_position"],
                        help="Action space the policy was trained with.")
    parser.add_argument("--position-gain", type=float, default=1.0,
                        help="Gain for joint_position→velocity conversion (joint_position mode only).")
    parser.add_argument("--max-joint-velocity", type=float, default=None,
                        help="If the largest |joint velocity| in a command exceeds this (rad/s), "
                             "scale the whole velocity vector down to preserve direction "
                             "(Cartesian path unchanged, just slower) rather than clipping "
                             "components independently. Applies to joint_velocity and "
                             "joint_position (converted) actions. Unset disables scaling.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embed-layer", default="bottleneck_plan_t0",
                        help="Embedding layer to extract online via the policy server "
                             "(pass to --embed-layer on the server). Set to '' to disable.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic observations instead of robot hardware.")
    parser.add_argument("--plot-actions", action="store_true",
                        help="Save a per-joint commanded-vs-actual plot (joint_tracking.png) "
                             "in each episode's recording dir after it finishes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    output_dir = pathlib.Path(args.output_dir)
    recordings = output_dir / "recordings"
    recordings.mkdir(parents=True, exist_ok=True)

    env = DROIDInferenceEnv(
        dry_run=args.dry_run,
        action_space=args.action_space,
        control_hz=args.control_hz,
        position_gain=args.position_gain,
        max_joint_velocity=args.max_joint_velocity,
    )
    policy = _build_policy(args.policy, args.device)
    runner = DROIDInferenceRunner(
        env,
        policy,
        mode=InferenceMode(args.mode),
        open_loop_horizon=args.open_loop_horizon,
        warmup_steps=args.warmup_steps,
        warmup_horizon=args.warmup_horizon,
        max_timesteps=args.max_timesteps,
        control_hz=args.control_hz,
    )

    use_tty = sys.stdin.isatty()
    lengths, successes = [], []
    ep = 0
    while ep < args.num_episodes:
        if use_tty:
            _wait_for_space(
                f"\nepisode {ep:04d} / {args.num_episodes} — SPACE to start, q to quit"
            )

        stop_event = threading.Event()
        if use_tty:
            print("  running — press SPACE to stop early", flush=True)
            listener = threading.Thread(
                target=_space_stop_listener, args=(stop_event,), daemon=True
            )
            listener.start()

        record = runner.run_episode(
            instruction=args.instruction, episode_idx=ep, stop_event=stop_event
        )
        stop_event.set()  # signal listener to exit if episode ended naturally
        if use_tty:
            listener.join(timeout=0.3)  # wait for raw-mode cleanup before prompt

        episode_data = env.pop_episode_data()
        env.reset()  # reset immediately so robot goes home while user labels

        tracking_fig = None
        if args.plot_actions:
            import matplotlib.pyplot as plt
            tracking_fig = _build_joint_tracking_figure(episode_data, args.action_space)
            if tracking_fig is not None:
                plt.show()  # blocks until closed — inspect before labeling the outcome

        if use_tty:
            success, note, retry = _prompt_outcome(ep)
        else:
            success, note, retry = None, "", False

        if retry:
            if tracking_fig is not None:
                import matplotlib.pyplot as plt
                plt.close(tracking_fig)
            logger.info("episode %d: retrying (not saved)", ep)
            continue

        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_dir_name = f"ep_{ts}"
        ep_dir = recordings / ep_dir_name
        env.save_droid_trajectory(
            ep_dir / "trajectory.h5",
            episode_data,
            success,
            fps=args.control_hz,
            instruction=args.instruction,
        )
        if record.embeddings is not None and args.embed_layer:
            emb_path = ep_dir / f"embeddings_{args.embed_layer}.npz"
            np.savez_compressed(emb_path, embeddings=record.embeddings)
        if tracking_fig is not None:
            import matplotlib.pyplot as plt
            tracking_fig.savefig(ep_dir / "joint_tracking.png", dpi=120)
            plt.close(tracking_fig)
            logger.info("wrote joint-tracking plot to %s", ep_dir / "joint_tracking.png")
        import yaml
        with open(ep_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump({
                "index": ep,
                "length": record.n_steps,
                "success": None if success is None else bool(success),
                "note": note,
                "args": vars(args),
            }, f, sort_keys=False)
        lengths.append(record.n_steps)
        successes.append(success)
        logger.info(
            "episode %d: %d steps, %d chunks, %.1f ms avg latency — %s%s",
            ep, record.n_steps, record.n_new_chunks, record.avg_chunk_latency_ms,
            "success" if success else ("failure" if success is False else "unknown"),
            f" ({note})" if note else "",
        )
        ep += 1

    labeled = [s for s in successes if s is not None]
    if labeled:
        logger.info(
            "session complete: %d / %d successful (%.0f%%) — %d unlabeled | written to %s",
            sum(labeled), len(labeled), 100 * sum(labeled) / len(labeled),
            len(successes) - len(labeled), output_dir,
        )
    else:
        logger.info("session complete: %d episodes, no labels — written to %s", len(successes), output_dir)


if __name__ == "__main__":
    main()
