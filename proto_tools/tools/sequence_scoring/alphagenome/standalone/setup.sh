#!/bin/bash
# Setup script for AlphaGenome standalone environment
set -euo pipefail
source standalone_helpers.sh

proto_check_gated_hf_repo "google/alphagenome-all-folds" "https://huggingface.co/google/alphagenome-all-folds" "README.md"

echo "Setting up AlphaGenome standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Resolve CUDA constraint (backward compat with ALPHAGENOME_CUDA_TOOLKIT_VERSION)
CUDA_CONSTRAINT="${ALPHAGENOME_CUDA_TOOLKIT_CONSTRAINT:-${ALPHAGENOME_CUDA_TOOLKIT_VERSION:-}}"
proto_install_cuda_toolkit "$CUDA_CONSTRAINT"

# Collapse CUDNN to a single source: delete the CUDNN shipped by the mamba
# cuda-toolkit so JAX's pip-bundled nvidia-cudnn-cu12 (under
# site-packages/nvidia/cudnn/lib) is the only CUDNN visible at runtime.
# Without this, LD_LIBRARY_PATH=${VENV_PATH}/cuda_env/lib (env_vars.txt)
# exposes the toolkit's CUDNN first and mixes sublibs across versions
# → CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED. XLA only needs nvcc/ptxas
# from the toolkit, not its CUDNN.
rm -f "${VENV_PATH}/cuda_env/lib/libcudnn"*

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

JAX_SPEC="${ALPHAGENOME_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"
proto_install_jax ALPHAGENOME

REPO_DIR="${VENV_PATH}/src/alphagenome_research"
ALPHAGENOME_COMMIT="b2046c6ae00a7e6d60fec071d6e4d19454a71f8e"
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning alphagenome_research..."
  git clone https://github.com/google-deepmind/alphagenome_research.git "$REPO_DIR"
else
  echo "alphagenome_research already cloned."
fi
echo "Checking out pinned alphagenome_research commit ${ALPHAGENOME_COMMIT}..."
git -C "$REPO_DIR" fetch origin "$ALPHAGENOME_COMMIT"
git -C "$REPO_DIR" checkout "$ALPHAGENOME_COMMIT"
echo "Installing alphagenome_research from local clone..."
uv pip install "$REPO_DIR"

# alphagenome_research leaves JAX unconstrained and may upgrade jax/jaxlib to
# versions incompatible with the selected CUDA plugin. Re-apply the selected
# JAX range to keep GPU wheels aligned with host driver/CUDA.
echo "Re-applying JAX compatibility spec after alphagenome_research install..."
uv pip install --upgrade "${JAX_SPEC}"

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
rm -f "$SITE_PACKAGES/sitecustomize.py"

echo "AlphaGenome setup complete!"
