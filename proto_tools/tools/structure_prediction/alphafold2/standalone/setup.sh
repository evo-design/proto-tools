#!/bin/bash
# Setup script for AlphaFold2 (ColabDesign) standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AlphaFold2 (ColabDesign) standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit "${ALPHAFOLD2_CUDA_TOOLKIT_CONSTRAINT:-}"
proto_install_jax ALPHAFOLD2

echo "Installing default ColabDesign and dependencies..."
uv pip install "colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@gamma"
uv pip install biopython ipython

echo "Installing Germinal ColabDesign fork (gradient backend with alpha/bias and framework contacts)..."
GERMINAL_DIR="${TOOL_VENV_PATH:-$VIRTUAL_ENV}/data/colabdesign_germinal"
mkdir -p "$GERMINAL_DIR"
uv pip install --target "$GERMINAL_DIR" "colabdesign @ git+https://github.com/SantiagoMille/germinal.git#subdirectory=colabdesign"

# Download AF2 parameters (~3.5GB)
proto_resolve_weights_dir alphafold2
WEIGHTS_DIR="${WEIGHTS_DIR}/params"
mkdir -p "$WEIGHTS_DIR"
PARAMS_DIR="$WEIGHTS_DIR"
if [ ! -d "$PARAMS_DIR" ] || [ -z "$(ls -A "$PARAMS_DIR"/*.npz 2>/dev/null)" ]; then
    echo "Downloading AlphaFold2 parameters (~3.5GB)..."
    mkdir -p "$PARAMS_DIR"
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C "$PARAMS_DIR"
    echo "AlphaFold2 parameters downloaded to $PARAMS_DIR"
else
    echo "AlphaFold2 parameters already present at $PARAMS_DIR"
fi

echo "AlphaFold2 (ColabDesign) setup complete!"
