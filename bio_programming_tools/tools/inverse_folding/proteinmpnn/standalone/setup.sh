#!/bin/bash
# Setup script for ProteinMPNN standalone environment
set -euo pipefail

echo "Setting up ProteinMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Auto-detect CUDA version and install the correct JAX variant with bundled CUDA libraries
# Modern JAX (>=0.4.20) bundles CUDA libraries by default, making venvs self-contained
echo "Detecting CUDA version for JAX installation..."
if command -v nvidia-smi &> /dev/null; then
    # Extract CUDA version from nvidia-smi output
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -n1)
    if [ -n "$CUDA_MAJOR" ]; then
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            echo "Detected CUDA ${CUDA_MAJOR} — installing jax[cuda13] (with bundled CUDA libraries)..."
            uv pip install "jax[cuda13]"
        else
            echo "Detected CUDA ${CUDA_MAJOR} — installing jax[cuda12] (with bundled CUDA libraries)..."
            uv pip install "jax[cuda12]"
        fi
    else
        echo "WARNING: Could not determine CUDA version. Installing jax[cuda12] as default..."
        uv pip install "jax[cuda12]"
    fi
else
    echo "WARNING: nvidia-smi not found. Installing JAX without CUDA support..."
    uv pip install jax
fi

echo "ProteinMPNN setup complete!"
