#!/bin/bash
# Setup script for ESM2 standalone environment
set -euo pipefail

echo "Setting up ESM2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "ESM2 setup complete!"
