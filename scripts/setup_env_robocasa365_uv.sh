#!/usr/bin/env bash
# Build the RoboCasa365 diffusion-policy eval venv (uv, no conda).
#
# Evaluates the released robocasa/robocasa365_checkpoints diffusion
# transformer hybrid policy with this repo's eval_save_rollouts.py
# (--runner-target diffusion_policy.env_runner.robocasa365_image_runner.Robocasa365ImageRunner
#  --layer obs_encoder). Needs the policy_graphs checkout layout:
#   third_party/diffusion_policy (this repo), third_party/robocasa365,
#   third_party/robomimic (lang_utils CLIP fork).
#
# Usage:  VENV=/mnt/ssdB/erik/policy_graphs/venvs/robocasa365_dp_venv \
#         scripts/setup_env_robocasa365_uv.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
THIRD_PARTY="$(cd "$HERE/.." && pwd)"
VENV="${VENV:-$HERE/.venv-robocasa365}"

uv venv --python 3.10 "$VENV" --allow-existing
PY="$VENV/bin/python"

# Torch first (with its CUDA libs — do NOT --no-deps torch).
uv pip install --python "$PY" "torch==2.7.1" "torchvision==0.22.1"

# Simulator stack: robocasa365 + robosuite master. --no-deps on the editables
# keeps their pyproject pins from fighting; the runtime deps follow explicitly.
uv pip install --python "$PY" --no-deps \
    -e "$THIRD_PARTY/robocasa365" \
    "robosuite @ git+https://github.com/ARISE-Initiative/robosuite@master" \
    "robomimic @ git+https://github.com/erikbr01/robocasa_robomimic@main" \
    -e "$HERE"

# Runtime deps (found the hard way; see docs/model_fleet.md in policy_graphs):
# - mujoco must be ==3.3.1 and numpy ==2.2.5 (robocasa365 asserts both)
uv pip install --python "$PY" \
    "mujoco==3.3.1" "numpy==2.2.5" "gymnasium>=0.29" lxml scipy numba pynput \
    pyyaml pillow "diffusers==0.11.1" einops dill hydra-core omegaconf zarr numcodecs \
    wandb imageio imageio-ffmpeg av termcolor h5py pandas tqdm matplotlib \
    imagecodecs "huggingface-hub==0.25.2" "transformers==4.45.2" opencv-python threadpoolctl

# old gym 0.21 for this repo's gym_util wrappers (legacy pip recipe)
uv pip install --python "$PY" "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"
"$VENV/bin/pip" install --quiet --no-build-isolation "gym==0.21.0"

# pytorch3d: no wheel for this torch/python; CPU-only source build suffices
# (only pytorch3d.transforms is used, which is pure torch).
PYTORCH3D_NO_CUDA=1 FORCE_CUDA=0 MAX_JOBS="${MAX_JOBS:-12}" \
uv pip install --python "$PY" --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

"$PY" - << 'EOF'
import torch, robocasa, robosuite, robomimic, gymnasium, diffusion_policy
import pytorch3d.transforms
from robomimic.utils.lang_utils import get_lang_emb
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.env_runner.robocasa365_image_runner import Robocasa365ImageRunner
print("robocasa365 DP venv OK | torch", torch.__version__, "| cuda", torch.cuda.is_available())
EOF
echo "Done: $VENV"
