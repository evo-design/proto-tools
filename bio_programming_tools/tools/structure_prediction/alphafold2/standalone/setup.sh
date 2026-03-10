#!/bin/bash
# Setup script for AlphaFold2 (ColabDesign) standalone environment
set -euo pipefail

echo "Setting up AlphaFold2 (ColabDesign) standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Use hardware-aware JAX spec from centralized detection
# (injected by bio_programming_tools.utils.compute_deps)
# Override with ALPHAFOLD2_JAX_VARIANT/JAX_SPEC env vars if needed
JAX_VARIANT="${ALPHAFOLD2_JAX_VARIANT:-${RECOMMENDED_JAX_VARIANT:-cuda12}}"
JAX_SPEC="${ALPHAFOLD2_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"

# ============================================================================
# Install CUDA toolkit via micromamba (JAX needs CUDA libs on LD_LIBRARY_PATH)
# ============================================================================
CUDA_TOOLKIT_CONSTRAINT="${ALPHAFOLD2_CUDA_TOOLKIT_CONSTRAINT:-}"
if [ -z "$CUDA_TOOLKIT_CONSTRAINT" ]; then
    if [ -n "${DETECTED_CUDA_VERSION:-}" ]; then
        CUDA_TOOLKIT_CONSTRAINT="${DETECTED_CUDA_VERSION}.*"
    else
        CUDA_TOOLKIT_CONSTRAINT="12.*"
    fi
fi
echo "Installing CUDA toolkit ${CUDA_TOOLKIT_CONSTRAINT} locally via micromamba..."
if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_CONSTRAINT}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_CONSTRAINT}" \
    "cudnn"; then
    echo "ERROR: Failed to install CUDA toolkit via micromamba"
    exit 1
fi

echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
echo "Detected driver: ${DETECTED_DRIVER_VERSION:-unknown}, CUDA: ${DETECTED_CUDA_VERSION:-unknown}"
echo "Installing JAX: ${JAX_SPEC}"
uv pip install "${JAX_SPEC}"

echo "Installing ColabDesign and dependencies..."
uv pip install "colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@gamma"
uv pip install biopython ipython

# Download AF2 parameters (~3.5GB) into the venv so they don't pollute the source tree
PARAMS_DIR="${VENV_PATH}/data/params"
if [ ! -d "$PARAMS_DIR" ] || [ -z "$(ls -A "$PARAMS_DIR"/*.npz 2>/dev/null)" ]; then
    echo "Downloading AlphaFold2 parameters (~3.5GB)..."
    mkdir -p "$PARAMS_DIR"
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C "$PARAMS_DIR"
    echo "AlphaFold2 parameters downloaded to $PARAMS_DIR"
else
    echo "AlphaFold2 parameters already present at $PARAMS_DIR"
fi

echo "AlphaFold2 (ColabDesign) setup complete!"
