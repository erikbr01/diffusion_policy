"""RoboCasa365 image runner for pretrained multitask diffusion policies.

Same episode-saving contract as ``robomimic_image_runner`` (sequential
episodes, per-inference-call ``episodes/ep####_{succ|fail}.pkl`` records +
``metadata.yaml`` + per-episode videos, consumed by
``scripts/eval_save_rollouts.py`` for offline embedding extraction), but the
environment is a RoboCasa365 gymnasium env (``gym.make("robocasa/<Task>")``,
robosuite 1.5) wrapped by :class:`Robocasa365ImageWrapper` — no robomimic
dataset/env_meta involved. Built for the released
``robocasa/robocasa365_checkpoints`` diffusion transformer hybrid policy;
actions are delta OSC converted by the wrapper (``abs_action`` unsupported).

Run inside the robocasa365 DP venv
(``/mnt/ssdB/erik/policy_graphs/venvs/robocasa365_dp_venv``).
"""

import collections
import math
import os
import pathlib
import pickle
import time
import uuid

import dill
import numpy as np
import pandas as pd
import torch
import tqdm
import wandb
import yaml

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robocasa365.robocasa365_image_wrapper import Robocasa365ImageWrapper
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy


def create_env(env_name, split, seed=None):
    import gymnasium
    import robocasa  # noqa: F401 — registers robocasa/* envs

    return gymnasium.make(f"robocasa/{env_name}", split=split, seed=seed)


class Robocasa365ImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        shape_meta: dict,
        env_name="PickPlaceCounterToCabinet",
        split="pretrain",
        env_seed=None,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=500,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="robot0_agentview_left_image",
        fps=10,
        crf=22,
        past_action=False,
        tqdm_interval_sec=5.0,
        save_episodes=False,
        # accepted-and-ignored knobs so checkpoint cfgs instantiate cleanly
        dataset_path=None,
        n_train=0,
        n_train_vis=0,
        train_start_idx=0,
        abs_action=False,
        n_envs=None,
        env_kwargs=None,
    ):
        super().__init__(output_dir)
        if env_kwargs:
            env_name = env_kwargs.get("env_name", env_name)
            split = env_kwargs.get("split", split)
            env_seed = env_kwargs.get("seed", env_seed)
        assert not abs_action, "robocasa365 uses delta OSC actions (converted in the wrapper)"

        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        def env_fn():
            robocasa_env = create_env(env_name=env_name, split=split, seed=env_seed)
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    Robocasa365ImageWrapper(
                        env=robocasa_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn]
        env_seeds = []
        env_prefixs = []
        env_init_fn_dills = []
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, i=i, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    if save_episodes:
                        filename = pathlib.Path(output_dir).joinpath("media", f"ep{i:04d}.mp4")
                    else:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", f"{uuid.uuid4()}_{time.time()}_{os.getpid()}.mp4"
                        )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, Robocasa365ImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        self.env = SyncVectorEnv(env_fns)
        self.episode_obs_keys = list(shape_meta["obs"].keys())
        self.env_name = env_name
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.abs_action = False
        self.tqdm_interval_sec = tqdm_interval_sec

        self.save_episodes = save_episodes
        self.episode_dir = None
        if save_episodes:
            assert n_test > 0 and n_test == n_test_vis
            self.episode_dir = pathlib.Path(output_dir) / "episodes"
            self.episode_dir.mkdir()
            self.media_dir = pathlib.Path(output_dir) / "media"

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        n_inits = len(self.env_init_fn_dills)
        all_video_paths = [None] * n_inits
        all_successes = [None] * n_inits

        if self.save_episodes:
            total_timesteps = 0
            episode_lengths = []
            episode_successes = []

        for chunk_idx in range(n_inits):
            env.call_each("run_dill_function", args_list=[(self.env_init_fn_dills[chunk_idx],)])

            if self.save_episodes:
                timestep = 0
                episode_data = []

            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {self.env_name} {chunk_idx + 1}/{n_inits}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            success = False
            while not done:
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict["past_action"] = past_action[:, -(self.n_obs_steps - 1) :].astype(np.float32)

                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())

                if self.save_episodes:
                    assert isinstance(
                        policy, (DiffusionUnetHybridImagePolicy, DiffusionTransformerHybridImagePolicy)
                    )
                    result = {
                        "idx": total_timesteps,
                        "episode": chunk_idx,
                        "timestep": timestep,
                        "obs": {
                            k: v.copy().astype(np.float32)[0]
                            for k, v in np_obs_dict.items()
                            if k in self.episode_obs_keys
                        },
                        "img": env.call("_render_frame", "rgb_array")[0],
                    }
                    if "action_pred" in np_action_dict:
                        result["action"] = np_action_dict["action_pred"].copy().astype(np.float32)[0]
                    else:
                        result["action"] = np_action_dict["action"].copy().astype(np.float32)[0]
                    episode_data.append(result)

                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                obs, reward, done, info = env.step(action)
                done = np.all(done)
                success = bool(env.call("_is_success")[0])
                if self.save_episodes and success:
                    done = True

                past_action = action
                pbar.update(action.shape[1])
                if self.save_episodes:
                    timestep += action.shape[1]
                    total_timesteps += 1
            pbar.close()

            all_video_paths[chunk_idx] = env.render()[0]
            all_successes[chunk_idx] = success

            if self.save_episodes:
                postfix = "succ" if success else "fail"
                episode_lengths.append(len(episode_data))
                episode_successes.append(success)
                episode_data = pd.DataFrame(episode_data)
                episode_data["reward"] = reward[0]
                episode_data["success"] = success
                with open(self.episode_dir / f"ep{chunk_idx:04d}_{postfix}.pkl", "wb") as f:
                    pickle.dump(episode_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                del episode_data

        _ = env.reset()

        log_data = {}
        per_prefix = collections.defaultdict(list)
        for i in range(n_inits):
            prefix = self.env_prefixs[i]
            per_prefix[prefix].append(float(all_successes[i]))
            log_data[prefix + f"sim_success_{self.env_seeds[i]}"] = float(all_successes[i])
            if all_video_paths[i] is not None:
                log_data[prefix + f"sim_video_{self.env_seeds[i]}"] = wandb.Video(all_video_paths[i])
        for prefix, values in per_prefix.items():
            log_data[prefix + "mean_score"] = float(np.mean(values))

        if self.save_episodes:
            episode_files = sorted(self.episode_dir.glob("ep*.pkl"))
            media_files = sorted(self.media_dir.glob("*.mp4"))
            assert len(episode_files) == len(media_files)
            for episode_file, media_file in zip(episode_files, media_files):
                media_file.rename(self.media_dir / f"{episode_file.stem}{media_file.suffix}")
            metadata = {
                "length": int(sum(episode_lengths)),
                "episode_lengths": [int(x) for x in episode_lengths],
                "episode_successes": [bool(x) for x in episode_successes],
            }
            with open(self.episode_dir / "metadata.yaml", "w") as f:
                yaml.safe_dump(metadata, f)

        return log_data
