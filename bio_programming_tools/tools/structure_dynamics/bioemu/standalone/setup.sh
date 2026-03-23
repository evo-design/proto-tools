#!/bin/bash
# Setup script for BioEmu standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up BioEmu standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_cuda_toolkit "${BIOEMU_CUDA_TOOLKIT_CONSTRAINT:-}"
bpt_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# ============================================================================
# Pre-build ColabFold venv with compatible JAX + CUDA
# ============================================================================
# BioEmu normally creates ~/.bioemu_colabfold on first run via its bundled
# colabfold_setup/setup.sh. We pre-build it here instead so we can:
#   1. Keep it inside the tool env (via BIOEMU_COLABFOLD_DIR)
#   2. Use sed instead of patch (patch fails with chown on setgid filesystems)
#   3. Install JAX with compatible CUDA versions for the detected driver
COLABFOLD_DIR="${BIOEMU_COLABFOLD_DIR:-${VENV_PATH}/colabfold_env}"

echo "Building ColabFold venv at ${COLABFOLD_DIR}..."
rm -rf "$COLABFOLD_DIR"

# Create venv and install colabfold (mirrors bioemu's colabfold_setup/setup.sh)
python -m venv --without-pip "$COLABFOLD_DIR"
uv pip install --python "$COLABFOLD_DIR/bin/python" 'colabfold[alphafold-minus-jax]==1.5.4'

# Install JAX with driver-compatible CUDA libs
# ColabFold 1.5.4 uses dm-haiku 0.0.11 which requires JAX <0.5 (uses removed
# jax.interpreters.xla.xe internals). Pin to 0.4.35 as upstream BioEmu does.
JAX_SPEC="jax[cuda12]==0.4.35"
echo "Installing JAX in ColabFold venv: ${JAX_SPEC}"
uv pip install --python "$COLABFOLD_DIR/bin/python" --force-reinstall \
    "${JAX_SPEC}" \
    "numpy==1.26.4"

# Apply bioemu's patches using sed (not patch, which fails with chown on setgid filesystems)
COLABFOLD_SITE_PACKAGES="$COLABFOLD_DIR/lib/python3.*/site-packages"

echo "Patching colabfold installation..."
# modules.py: add representations_evo to the return dict (line 146)
sed -i "s/ret = {'representations':representations}/ret = {'representations':representations, 'representations_evo': representations}/" \
    ${COLABFOLD_SITE_PACKAGES}/alphafold/model/modules.py

# batch.py: save single and pair evo representations after the existing save lines
# Note: colabfold's batch.py has no spaces after commas in files.get() calls
sed -i '/np.save(files.get("single_repr","npy")/a\                np.save(files.get("single_repr_evo","npy"),result["representations_evo"]["single"])' \
    ${COLABFOLD_SITE_PACKAGES}/colabfold/batch.py
sed -i '/np.save(files.get("pair_repr","npy")/a\                np.save(files.get("pair_repr_evo","npy"),result["representations_evo"]["pair"])' \
    ${COLABFOLD_SITE_PACKAGES}/colabfold/batch.py

touch "$COLABFOLD_DIR/.COLABFOLD_PATCHED"
echo "ColabFold venv ready at ${COLABFOLD_DIR}"

echo "BioEmu setup complete!"
