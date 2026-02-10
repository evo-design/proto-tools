#!/bin/bash
# Setup script for ESMFold standalone environment
set -euo pipefail

echo "Setting up ESMFold standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Use --torch-backend=auto to automatically detect GPU and install
# the correct PyTorch build (CUDA-enabled or CPU-only).
echo "Installing remaining dependencies..."
uv pip install torch transformers biopython --torch-backend=auto

echo "ESMFold setup complete!"
