#!/usr/bin/env bash
# Create the RoboCasa training environment with uv (no conda).
#
# RoboCasa needs robosuite >=1.5 (PandaOmron, composite controllers), which is
# incompatible with the mimicgen/robomimic stack's robosuite 1.4.1 — hence a
# separate venv from the one built by scripts/setup_env_uv.sh.
#
# Usage, from the diffusion_policy repo root inside a policy_graphs checkout:
#
#   scripts/setup_env_robocasa_uv.sh [VENV_DIR]
#
# Pass the adapter the resulting interpreter:
#   ++env.python_path=<VENV_DIR>/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${1:-.venv-robocasa}"
ROBOSUITE_REF="85abee2"  # ARISE-Initiative/robosuite master, robosuite 1.5.2
ROBOSUITE_SRC="${ROBOSUITE_SRC:-$VENV-src/robosuite}"

uv venv --clear --python 3.11 "$VENV"
PY="$VENV/bin/python"
uv pip install --python "$PY" "setuptools==65.5.0" "wheel==0.38.4" "pip<24"

# robosuite 1.5 from a pinned ARISE master commit
if [ ! -d "$ROBOSUITE_SRC" ]; then
    git clone https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_SRC"
fi
git -C "$ROBOSUITE_SRC" fetch -q && git -C "$ROBOSUITE_SRC" checkout -q "$ROBOSUITE_REF"
uv pip install --python "$PY" -e "$ROBOSUITE_SRC"

# robocasa + the 0.5-era robomimic it needs (the policy_graphs fork submodule)
"$VENV/bin/pip" install --quiet -e ../robocasa --no-deps
"$VENV/bin/pip" install --quiet -e ../robomimic --no-deps

# training stack (py3.11-compatible subset; the main requirements.txt pins a
# py3.9 stack with robosuite 1.4.1 and cannot be installed here verbatim)
uv pip install --python "$PY" \
    "torch==2.4.0" torchvision "numpy>=2,<3" numba scipy mujoco gymnasium \
    "hydra-core==1.3.2" "omegaconf==2.3.0" "diffusers==0.11.1" \
    "huggingface-hub==0.25.2" "zarr<3" einops wandb av scikit-image \
    numpydantic transformers pandas pyarrow fastparquet imagecodecs \
    threadpoolctl tqdm dill h5py imageio imageio-ffmpeg
"$VENV/bin/pip" install --quiet --no-build-isolation "gym==0.21.0"
uv pip install --python "$PY" -e . --no-deps
FORCE_CUDA=0 "$VENV/bin/pip" install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

echo "done: $($PY --version) at $(pwd)/$VENV"
echo "robocasa assets: python -m robocasa.scripts.download_kitchen_assets (once)"
