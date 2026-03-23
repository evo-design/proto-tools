#!/bin/bash
# Setup script for AlphaGenome standalone environment
set -euo pipefail
source standalone_helpers.sh

bpt_check_gated_hf_repo "google/alphagenome-all-folds" "https://huggingface.co/google/alphagenome-all-folds" "README.md"

echo "Setting up AlphaGenome standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Resolve CUDA constraint (backward compat with ALPHAGENOME_CUDA_TOOLKIT_VERSION)
CUDA_CONSTRAINT="${ALPHAGENOME_CUDA_TOOLKIT_CONSTRAINT:-${ALPHAGENOME_CUDA_TOOLKIT_VERSION:-}}"
bpt_install_cuda_toolkit "$CUDA_CONSTRAINT"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

JAX_SPEC="${ALPHAGENOME_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"
bpt_install_jax ALPHAGENOME

REPO_DIR="${VENV_PATH}/src/alphagenome_research"
if [ -d "$REPO_DIR" ]; then
  echo "Updating existing alphagenome_research clone..."
  git -C "$REPO_DIR" pull origin main
else
  echo "Cloning alphagenome_research..."
  git clone https://github.com/google-deepmind/alphagenome_research.git "$REPO_DIR"
fi
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
