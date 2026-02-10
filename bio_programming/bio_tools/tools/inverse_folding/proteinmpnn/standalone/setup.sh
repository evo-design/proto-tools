#!/bin/bash
# Setup script for ProteinMPNN standalone environment
set -euo pipefail

echo "Setting up ProteinMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Auto-detect CUDA version and install the correct JAX variant
echo "Detecting CUDA version for JAX installation..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+')
    if [ -n "$CUDA_MAJOR" ]; then
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            echo "Detected CUDA ${CUDA_MAJOR} — installing jax[cuda13]..."
            uv pip install "jax[cuda13]"
        else
            echo "Detected CUDA ${CUDA_MAJOR} — installing jax[cuda12]..."
            uv pip install -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12_local]"
        fi
    else
        echo "WARNING: Could not determine CUDA version. Installing jax[cuda12] as default..."
        uv pip install -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jax[cuda12_local]"
    fi
else
    echo "WARNING: nvidia-smi not found. Installing JAX without CUDA support..."
    uv pip install jax
fi

echo "ProteinMPNN setup complete!"
