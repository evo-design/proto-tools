#!/bin/bash
# Setup script for Boltz2 standalone environment
set -euo pipefail

echo "Setting up Boltz2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Use --torch-backend=auto to automatically detect GPU and install
# the correct PyTorch build (CUDA-enabled or CPU-only).
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "Boltz2 setup complete!"
