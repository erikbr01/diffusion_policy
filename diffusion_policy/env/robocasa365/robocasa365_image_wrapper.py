"""Gym wrapper exposing RoboCasa365 (robocasa 1.0 / robosuite 1.5) envs to
diffusion-policy image runners.

Port of robocasa-benchmark/diffusion_policy's adapted RobomimicImageWrapper:
the underlying env is a robocasa365 *gymnasium* env (``gym.make("robocasa/<Task>")``,
lerobot-style obs keys), observations are remapped through the checkpoint's
``shape_meta`` (``lerobot_keys``), the task language is embedded with the
``LangEncoder`` from robocasa-benchmark/robomimic (the exact robomimic
flavor the released checkpoints were trained with — its custom ResNet
obs-net structure is also why that fork, mirrored at
erikbr01/robocasa_robomimic, must be installed in this venv instead of our
robomimic), and actions go through ``robocasa.utils.env_utils.convert_action``.
"""

from typing import Optional

import numpy as np
import gym
from gym import spaces

from robomimic.macros import LANG_EMB_KEY
from robomimic.utils.lang_utils import LangEncoder
from robocasa.utils.env_utils import convert_action


def _process_frame(frame: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32 in [0, 1]."""
    frame = np.asarray(frame).astype(np.float32) / 255.0
    return np.moveaxis(frame, -1, 0)


class Robocasa365ImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key="robot0_agentview_left_image",
    ):
        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.lang = None
        self.lang_emb = None
        self.lang_encoder = LangEncoder("cpu")

        action_shape = shape_meta["action"]["shape"]
        self.action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)

        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            if key == LANG_EMB_KEY:
                min_value, max_value = -100, 100
            elif key.endswith("image"):
                min_value, max_value = 0, 1
            else:
                min_value, max_value = -1, 1
            observation_space[key] = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
        self.observation_space = observation_space

    def process_obs(self, raw_obs):
        """Remap lerobot-style keys to policy keys; normalize images to CHW [0,1]."""
        policy_obs = {}
        for policy_key, value in self.shape_meta["obs"].items():
            raw_keys = value.get("lerobot_keys", None)
            if raw_keys is None:
                continue
            entry = raw_obs[raw_keys[0]]
            if value.get("type", "low_dim") == "rgb":
                entry = _process_frame(entry)
            policy_obs[policy_key] = entry
        return policy_obs

    def get_observation(self, raw_obs):
        obs_proc = self.process_obs(raw_obs)
        assert self.lang is not None
        obs_proc[LANG_EMB_KEY] = self.lang_emb
        self.render_cache = obs_proc[self.render_obs_key]
        return {key: obs_proc[key] for key in self.observation_space.keys()}

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        raw_obs, info = self.env.reset()
        self.lang = raw_obs["annotation.human.task_description"]
        self.lang_emb = self.lang_encoder.get_lang_emb(self.lang).numpy()
        return self.get_observation(raw_obs)

    def step(self, action):
        raw_obs, reward, done, truncated, info = self.env.step(convert_action(action))
        obs = self.get_observation(raw_obs)
        # robocasa365 success lives in info; latch it for _is_success.
        self._last_info = info
        return obs, reward, done, info

    def _render_frame(self, mode="rgb_array"):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        img = np.moveaxis(self.render_cache, 0, -1)
        return (img * 255).astype(np.uint8)

    def _is_success(self):
        info = getattr(self, "_last_info", None) or {}
        return bool(info.get("success", False))

    def render(self, mode="rgb_array"):
        return self._render_frame(mode)
