"""Generate low-dim diffusion-policy experiment configs for MimicGen core tasks.

Reads each core MimicGen HDF5 (robomimic-format, delta OSC 7-DoF actions,
low-dim obs keys ``object`` / ``robot0_eef_pos`` / ``robot0_eef_quat`` /
``robot0_gripper_qpos``) and emits a self-contained experiment config at
``configs/low_dim/<task>_mimicgen_lowdim/diffusion_policy_cnn/config.yaml``
following the ``square_mimicgen_lowdim`` pattern:

- ``obs_dim`` from the dataset's actual object-key dimension (+ 3 + 4 + 2),
- ``max_steps`` ≈ 2x the longest demo (rounded up to 50, min 400),
- dataset paths point at the given HDF5 (overridden at launch anyway by
  policy_graphs' ``dataset_override_keys``).

Usage::

    .venv/bin/python scripts/generate_mimicgen_lowdim_configs.py \
        /mnt/ssdB/.../mimicgen/core/*.hdf5
"""

import argparse
import json
import math
from pathlib import Path

import h5py
import yaml

REPO = Path(__file__).resolve().parents[1]
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ROBOT_DIM = 3 + 4 + 2  # eef_pos + eef_quat + gripper_qpos


def inspect_dataset(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demos = list(f["data"])
        first = f[f"data/{demos[0]}/obs"]
        missing = [k for k in OBS_KEYS if k not in first]
        if missing:
            raise ValueError(f"{path}: missing low-dim obs keys {missing}")
        control_delta = env_args["env_kwargs"]["controller_configs"]["control_delta"]
        if not control_delta:
            raise ValueError(f"{path}: expected delta OSC actions (control_delta=true)")
        object_dim = first["object"].shape[1]
        action_dim = f[f"data/{demos[0]}/actions"].shape[1]
        if action_dim != 7:
            raise ValueError(f"{path}: expected 7-DoF delta actions, got {action_dim}")
        max_len = max(f[f"data/{d}/actions"].shape[0] for d in demos)
    return {
        "env_name": env_args["env_name"],
        "obs_dim": object_dim + ROBOT_DIM,
        "object_dim": object_dim,
        "max_steps": max(400, int(math.ceil(2 * max_len / 50.0)) * 50),
        "num_demos": len(demos),
    }


def build_config(task: str, dataset_path: Path, info: dict) -> dict:
    obs_dim = info["obs_dim"]
    name = f"{task}_mimicgen_lowdim"
    return {
        "defaults": ["../../_shared/diffusion_unet_lowdim_square_arch@_here_", "_self_"],
        "obs_dim": obs_dim,
        "action_dim": 7,
        "policy": {
            "obs_dim": obs_dim,
            "action_dim": 7,
            "model": {"global_cond_dim": obs_dim * 2, "input_dim": 7},
        },
        "exp_name": f"mimicgen_{task}_lowdim",
        "logging": {
            "name": f"train_diffusion_unet_lowdim_{name}",
            "tags": ["train_diffusion_unet_lowdim", name, "mimicgen"],
        },
        "multi_run": {
            "run_dir": f"data/outputs/mimicgen_{task}_lowdim/train_diffusion_unet_lowdim",
            "wandb_name_base": f"mimicgen_{task}_lowdim_diffusion_unet_lowdim",
        },
        "task": {
            "abs_action": False,
            "action_dim": 7,
            "dataset": {
                "_target_": "diffusion_policy.dataset.mimicgen_replay_lowdim_dataset.MimicgenReplayLowdimDataset",
                "abs_action": False,
                "dataset_path": str(dataset_path),
                "horizon": 16,
                "max_train_episodes": None,
                "obs_keys": OBS_KEYS,
                "pad_after": 7,
                "pad_before": 1,
                "seed": 42,
                "use_legacy_normalizer": False,
                "val_ratio": 0.02,
            },
            "dataset_path": str(dataset_path),
            "dataset_type": "gen",
            "env_runner": {
                "_target_": "diffusion_policy.env_runner.mimicgen_lowdim_runner.MimicgenLowdimRunner",
                "abs_action": False,
                "crf": 22,
                "dataset_path": str(dataset_path),
                "fps": 10,
                "max_steps": info["max_steps"],
                "n_action_steps": 8,
                "n_envs": 28,
                "n_latency_steps": 0,
                "n_obs_steps": 2,
                "n_test": 50,
                "n_test_vis": 4,
                "n_train": 6,
                "n_train_vis": 2,
                "obs_keys": OBS_KEYS,
                "past_action": False,
                "render_hw": [128, 128],
                "test_start_seed": 100000,
                "train_start_idx": 0,
            },
            "keypoint_dim": 3,
            "name": name,
            "obs_dim": obs_dim,
            "obs_keys": OBS_KEYS,
            "task_name": f"{task}_mimicgen",
        },
        "task_name": name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="core MimicGen hdf5 files (<task>_dN.hdf5)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing config dirs")
    args = parser.parse_args()

    header = (
        "# MimicGen {env} low-dim diffusion — generated by\n"
        "# scripts/generate_mimicgen_lowdim_configs.py from {src}\n"
        "# (obs_dim {obs_dim}: object {object_dim} + eef 7 + gripper 2; {num_demos} demos;\n"
        "# max_steps {max_steps} ~= 2x longest demo). Delta OSC actions — abs_action stays false.\n"
    )
    for dataset_path in args.datasets:
        task = dataset_path.stem  # e.g. stack_d0
        info = inspect_dataset(dataset_path)
        out_dir = REPO / "configs" / "low_dim" / f"{task}_mimicgen_lowdim" / "diffusion_policy_cnn"
        out_path = out_dir / "config.yaml"
        if out_path.exists() and not args.force:
            print(f"{task}: exists, skipping ({out_path})")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        config = build_config(task, dataset_path, info)
        text = header.format(env=info["env_name"], src=dataset_path, **info) + yaml.safe_dump(
            config, sort_keys=False, default_flow_style=False
        )
        out_path.write_text(text)
        print(
            f"{task}: {info['env_name']} obs_dim={info['obs_dim']} "
            f"max_steps={info['max_steps']} demos={info['num_demos']} -> {out_path.relative_to(REPO)}"
        )


if __name__ == "__main__":
    main()
