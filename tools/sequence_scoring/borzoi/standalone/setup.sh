#!/bin/bash
# Setup script for Borzoi standalone environment
set -euo pipefail

echo "Setting up Borzoi standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing PyTorch and borzoi-pytorch..."
uv pip install torch==2.7.1 borzoi-pytorch --torch-backend=auto

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "WARNING: Skipping flash-attn installation — pre-built wheel is x86_64 only."
    echo "Borzoi may not work correctly without flash-attn on this platform."
else
    echo "Installing flash-attn from pre-built wheel..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
fi

echo "Borzoi setup complete!"
