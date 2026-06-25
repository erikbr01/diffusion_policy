"""DROID real-robot inference environment.

Wraps ``droid.robot_env.RobotEnv`` with:
  - observation preprocessing (BGR→RGB / BGRA→RGB, state assembly)
  - per-step episode recording for the rollout writers
  - dry_run mode for smoke-testing the control loop without hardware

The ``droid`` package is imported lazily (inside methods) so this module can
be imported anywhere; hardware code paths raise a clear error when the DROID
robot stack is absent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DROID_HINT = (
    "requires the DROID robot stack ('droid' package + ZED/Franka drivers), "
    "present on the robot workstation"
)


def _import_droid_module(module: str):
    """Import a droid submodule, with ``~/src_droid`` as a path fallback."""
    src = Path("~/src_droid").expanduser()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"importing {module!r} failed — this {_DROID_HINT}") from e


class DROIDInferenceEnv:
    """Inference wrapper around ``droid.robot_env.RobotEnv``.

    Parameters
    ----------
    action_space : str
        'joint_velocity', 'cartesian_velocity', or 'joint_position'.
        'joint_position' uses a velocity wrapper: at each step the absolute
        target is converted to joint_velocity via
        ``(target - current_pos) * control_hz`` and sent to the robot.
    control_hz : float
        Control frequency in Hz. Used by the joint_position velocity wrapper.
    wrist_serial, ext1_serial, ext2_serial : str
        ZED serial numbers of the wrist / exterior cameras.
    dry_run : bool
        Skip hardware init entirely; get_obs() returns random tensors and
        step() is a no-op. For smoke-testing the runner without a robot.
    record_data : bool
        Accumulate per-step {obs, action} dicts for pop_episode_data().
    """

    # Camera serial numbers matching the lab rig (from droid/misc/parameters.py).
    DEFAULT_WRIST_SERIAL = "14313307"
    DEFAULT_EXT1_SERIAL = "36716034"
    DEFAULT_EXT2_SERIAL = "37617599"

    def __init__(
        self,
        action_space: str = "joint_velocity",
        control_hz: float = 15.0,
        position_gain: float = 1.0,
        wrist_serial: str = DEFAULT_WRIST_SERIAL,
        ext1_serial: str = DEFAULT_EXT1_SERIAL,
        ext2_serial: str = DEFAULT_EXT2_SERIAL,
        dry_run: bool = False,
        record_data: bool = True,
    ) -> None:
        if action_space not in ("joint_velocity", "cartesian_velocity", "joint_position"):
            raise ValueError(f"Unknown action_space: {action_space!r}")

        self._action_space_name = action_space
        self._control_hz = control_hz
        self._position_gain = position_gain
        self.wrist_serial = wrist_serial
        self.ext1_serial = ext1_serial
        self.ext2_serial = ext2_serial
        self.dry_run = dry_run
        self.record_data = record_data

        # 7-DOF Franka: cartesian = 6+1=7, joint = 7+1=8. Matches RobotEnv.DoF.
        self.action_dim = 7 if "cartesian" in action_space else 8

        self._robot_env = None
        self._episode_data: List[Dict] = []
        self._last_obs: Optional[Dict] = None

        if not dry_run:
            self._init_robot()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @property
    def _robot_action_space(self) -> str:
        # joint_position is implemented as a velocity wrapper on top of joint_velocity
        return "joint_velocity" if self._action_space_name == "joint_position" else self._action_space_name

    def _init_robot(self) -> None:
        robot_env_mod = _import_droid_module("droid.robot_env")

        camera_kwargs = {
            self.wrist_serial: {"image": True},
            self.ext1_serial: {"image": True},
            self.ext2_serial: {"image": True},
        }
        self._robot_env = robot_env_mod.RobotEnv(
            action_space=self._robot_action_space,
            camera_kwargs=camera_kwargs,
            do_reset=True,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict:
        self._episode_data = []
        if self.dry_run:
            obs = self._dummy_obs()
        else:
            self._robot_env.reset()
            obs = self.get_obs()
        self._last_obs = obs
        return obs

    def _to_robot_action(self, action: np.ndarray, current_obs: Dict) -> np.ndarray:
        """Convert policy action to the robot's native action space."""
        if self._action_space_name != "joint_position":
            return action
        current_joints = current_obs.get("joint_position", np.zeros(7))
        velocity = (action[:7] - current_joints) * self._control_hz * self._position_gain
        return np.concatenate([velocity, action[7:8]], axis=0)

    def step(self, action: np.ndarray):
        if self.dry_run:
            obs = self._dummy_obs()
        else:
            robot_action = self._to_robot_action(action, self._last_obs or {})
            self._robot_env.update_robot(
                robot_action,
                action_space=self._robot_action_space,
                gripper_action_space="position",
                blocking=False,
            )
            obs = self.get_obs()

        if self.record_data:
            self._episode_data.append(
                {
                    "timestep": len(self._episode_data),
                    "obs": {
                        k: v.copy() if hasattr(v, "copy") else v
                        for k, v in obs.items()
                    },
                    "action": np.asarray(action, dtype=np.float32).copy(),
                }
            )
        self._last_obs = obs
        return obs, 0.0, False, {}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_obs(self) -> Dict:
        """Return preprocessed observation dict ready for policy consumption."""
        if self.dry_run:
            return self._dummy_obs()
        raw = self._robot_env.get_observation()
        return self._extract_observation(raw)

    def _extract_observation(self, raw: Dict) -> Dict:
        image_obs = raw.get("image", {})
        robot_state = raw.get("robot_state", {})

        obs: Dict[str, np.ndarray] = {
            "joint_position": np.array(
                robot_state.get("joint_positions", np.zeros(7)), dtype=np.float32
            ),
            "gripper_position": np.atleast_1d(
                np.array(robot_state.get("gripper_position", 0.0), dtype=np.float32)
            ),
            "cartesian_position": np.array(
                robot_state.get("cartesian_position", np.zeros(6)), dtype=np.float32
            ),
        }

        for serial, key in (
            (self.wrist_serial, "wrist_image"),
            (self.ext1_serial, "exterior_image_1_left"),
            (self.ext2_serial, "exterior_image_2_left"),
        ):
            img = image_obs.get(f"{serial}_left")
            if img is not None:
                obs[key] = self._preprocess_image(img)

        return obs

    @staticmethod
    def _preprocess_image(img: np.ndarray) -> np.ndarray:
        # Only convert colour space — do NOT resize here. Policy backends are
        # responsible for resizing to their model's expected resolution.
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1].copy()  # BGR → RGB
        return img

    def _dummy_obs(self) -> Dict:
        rng = np.random.default_rng()
        return {
            "joint_position": rng.random(7).astype(np.float32),
            "gripper_position": rng.random(1).astype(np.float32),
            "cartesian_position": rng.random(6).astype(np.float32),
            "wrist_image": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "exterior_image_1_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
            "exterior_image_2_left": rng.integers(0, 255, (180, 320, 3), dtype=np.uint8),
        }

    # ------------------------------------------------------------------
    # Episode recording
    # ------------------------------------------------------------------

    def pop_episode_data(self) -> List[Dict]:
        """Return the recorded per-step data of the current episode and clear it."""
        data, self._episode_data = self._episode_data, []
        return data

    def save_droid_trajectory(
        self,
        path: Path | str,
        episode_data: List[Dict],
        success: Optional[bool] = None,
        fps: float = 15.0,
        instruction: str = "",
    ) -> Path:
        """Save an episode as DROID-format trajectory.h5 via TrajectoryWriter.

        Produces a file readable by robo_dataset's DroidStore for later
        re-ingestion. Robot-workstation only (uses the droid package).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer_mod = _import_droid_module("droid.trajectory_utils.trajectory_writer")
        writer = writer_mod.TrajectoryWriter(
            str(path),
            metadata={"success": bool(success), "instruction": instruction},
            save_images=False,
        )
        image_keys = [
            k for k in ("wrist_image", "exterior_image_1_left", "exterior_image_2_left")
            if episode_data and k in episode_data[0]["obs"]
        ]
        image_buffers: Dict[str, list] = {k: [] for k in image_keys}
        for step in episode_data:
            obs = step["obs"]
            action = step["action"]
            for k in image_keys:
                image_buffers[k].append(obs[k])
            timestep = {
                "observation": {
                    "robot_state": {
                        "joint_positions": obs.get("joint_position", np.zeros(7)),
                        "cartesian_position": obs.get("cartesian_position", np.zeros(6)),
                        "gripper_position": float(
                            np.atleast_1d(obs.get("gripper_position", np.zeros(1)))[0]
                        ),
                    },
                    "controller_info": {
                        "movement_enabled": True,
                        "success": False,
                        "failure": False,
                    },
                },
                "action": {
                    "joint_velocity": action[:7] if self._action_space_name != "joint_position" else np.zeros(7),
                    "joint_position": action[:7] if self._action_space_name == "joint_position" else np.zeros(7),
                    "gripper_position": float(action[7:8][0]) if len(action) > 7 else 0.0,
                    "cartesian_velocity": np.zeros(6),
                },
            }
            writer.write_timestep(timestep)
        writer.close(metadata={"success": bool(success)})

        # Save per-camera frames as H.264 MP4s alongside trajectory.h5 so
        # offline embedding extraction can reconstruct any layer later.
        import imageio
        for k, frames in image_buffers.items():
            mp4_path = path.parent / f"{k}.mp4"
            writer_mp4 = imageio.get_writer(
                str(mp4_path), fps=int(fps), macro_block_size=1,
                output_params=["-crf", "18", "-preset", "fast"],
            )
            for frame in frames:
                writer_mp4.append_data(frame)
            writer_mp4.close()
        return path

    # ------------------------------------------------------------------

    def close(self) -> None:
        pass  # ZED cameras / RobotEnv have no explicit close in current droid versions
