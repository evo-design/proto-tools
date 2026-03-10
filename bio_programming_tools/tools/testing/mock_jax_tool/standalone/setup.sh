#!/bin/bash
set -euo pipefail

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

# Install hardware-aware JAX version (from centralized detection)
JAX_SPEC="${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}"
echo "Installing JAX: ${JAX_SPEC} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${JAX_SPEC}" --refresh

echo "Mock JAX tool setup complete"
