#!/bin/bash
# Setup script for LigandMPNN standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up LigandMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

proto_resolve_weights_dir ligandmpnn

echo "Downloading LigandMPNN model weights..."
foundry install ligandmpnn --checkpoint-dir "$WEIGHTS_DIR"

echo "LigandMPNN setup complete!"
