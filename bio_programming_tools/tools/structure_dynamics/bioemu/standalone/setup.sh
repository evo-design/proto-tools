#!/bin/bash
# Setup script for BioEmu standalone environment
set -euo pipefail

echo "Setting up BioEmu standalone environment..."

# Clear stale colabfold venv so it rebuilds with the correct Python version.
# BioEmu creates ~/.bioemu_colabfold on first run; if it exists from a
# previous env (e.g. wrong Python version), the patches won't be reapplied.
rm -rf "${HOME}/.bioemu_colabfold"

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --torch-backend=auto

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "BioEmu setup complete!"
