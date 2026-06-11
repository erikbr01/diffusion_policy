#!/usr/bin/env bash
# Create the training environment with uv (no conda).
#
# Usage, from the diffusion_policy repo root (third_party/diffusion_policy
# inside a policy_graphs checkout):
#
#   scripts/setup_env_uv.sh
#
# Installs the pinned training stack plus the simulator packages from the
# sibling policy_graphs submodules (robomimic, mimicgen, robocasa).
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --clear --python 3.9 .venv
# gym 0.21 needs legacy build tooling
uv pip install --python .venv/bin/python "setuptools==65.5.0" "wheel==0.38.4" "pip<24"
uv pip install --python .venv/bin/python -r requirements.txt
# gym 0.21 cannot build in isolation (broken extras_require with new setuptools)
.venv/bin/pip install --quiet --no-build-isolation "gym==0.21.0"
uv pip install --python .venv/bin/python -e . --no-deps

# Simulators from the policy_graphs submodules when run inside the main repo.
for sim in ../robomimic ../mimicgen ../robocasa ../robo_dataset; do
    if [ -d "$sim" ]; then
        echo "installing $(basename "$sim") from $sim"
        .venv/bin/pip install --quiet -e "$sim" --no-deps
    fi
done

echo "done: $(.venv/bin/python --version) at $(pwd)/.venv"

# pytorch3d has no linux wheel for this torch/python combo; CPU-only build is
# sufficient (only pytorch3d.transforms is used, which is pure torch).
FORCE_CUDA=0 .venv/bin/pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
