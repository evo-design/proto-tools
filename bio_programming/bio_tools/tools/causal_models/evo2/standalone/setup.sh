#!/bin/bash
# Setup script for Evo2 standalone environment
set -euo pipefail

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: Evo2 is not supported on aarch64."
    echo "Evo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels."
    exit 1
fi

echo "Setting up Evo2 standalone environment..."

echo "Prerequisites not installed by this script:"
echo "  - CUDA-enabled PyTorch compatible with your NVIDIA drivers"
echo "  - Python 3.12"
echo "  - CUDA 12.1+ and cuDNN 9.3+"
echo "  - GCC 9+ or Clang 10+ with C++17 support"
echo ""

echo "Installing prerequisites..."
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch==2.3.0
pip install flash-attn==2.8.0.post2

echo "Installing vortex..."
pip install vtx

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "If installation fails, follow upstream setup guides:"
echo "  - https://github.com/ArcInstitute/evo2"
echo "  - https://github.com/Zymrael/vortex"

echo "Evo2 setup complete!"
