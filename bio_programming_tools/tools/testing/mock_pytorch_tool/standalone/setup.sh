#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Installing uv package manager..."
pip install uv

# Clear caches BEFORE installing any ABI-sensitive packages
echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true

proto_install_pytorch

echo "Mock PyTorch tool setup complete"
