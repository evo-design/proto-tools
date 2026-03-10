#!/bin/bash
# Setup script for ProteinMPNN standalone environment
set -euo pipefail

echo "Setting up ProteinMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

# ============================================================================
# Install CUDA toolkit via micromamba (JAX needs CUDA libs on LD_LIBRARY_PATH)
# ============================================================================
DETECTED_CUDA_MAJOR="${DETECTED_CUDA_VERSION:-12}"
CUDA_TOOLKIT_VERSION="${DETECTED_CUDA_MAJOR}.*"
echo "Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} locally via micromamba..."
if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}" \
    "cudnn"; then
    echo "ERROR: Failed to install CUDA toolkit via micromamba"
    exit 1
fi

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Use hardware-aware JAX spec from centralized detection
# (injected by bio_programming_tools.utils.compute_deps)
# Override with PROTEINMPNN_JAX_VARIANT/JAX_SPEC env vars if needed
JAX_VARIANT="${PROTEINMPNN_JAX_VARIANT:-${RECOMMENDED_JAX_VARIANT:-cuda12}}"
JAX_SPEC="${PROTEINMPNN_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.4.20,<1}}"

echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
echo "Installing JAX: ${JAX_SPEC}"

uv pip install "${JAX_SPEC}"

echo "ProteinMPNN setup complete!"
