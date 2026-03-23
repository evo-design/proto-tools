#!/bin/bash
# Setup script for RFdiffusion3 standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up RFdiffusion3 standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

bpt_resolve_weights_dir rfdiffusion3

echo "Downloading rfdiffusion3 model weights..."
foundry install rfd3 --checkpoint-dir "$WEIGHTS_DIR"

echo "RFdiffusion3 setup complete!"
