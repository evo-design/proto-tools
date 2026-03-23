#!/bin/bash
# Setup script for LigandMPNN standalone environment
set -euo pipefail

echo "Setting up LigandMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Downloading LigandMPNN model weights..."
foundry install ligandmpnn

echo "LigandMPNN setup complete!"
