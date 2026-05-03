#!/bin/bash
# Setup script for ProGen2 standalone environment
set -euo pipefail
source standalone_helpers.sh

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: progen2 setup: not supported on aarch64 (torch==2.2.2 pin has no aarch64 CUDA wheel)" >&2
    exit 1
fi

echo "Setting up ProGen2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "ProGen2 setup complete!"
