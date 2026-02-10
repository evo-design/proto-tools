#!/bin/bash
# Setup script for ProGen2 standalone environment
set -euo pipefail

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: ProGen2 is not supported on aarch64."
    echo "ProGen2 pins torch==2.2.2 which has no aarch64 CUDA wheel available."
    exit 1
fi

echo "Setting up ProGen2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "ProGen2 setup complete!"
