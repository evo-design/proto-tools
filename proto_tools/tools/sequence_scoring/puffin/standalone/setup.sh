#!/bin/bash
# Setup script for Puffin standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Puffin standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# ============================================================================
# Clone the upstream Puffin repo so the Puffin class can be imported and so
# the bundled puffin.pth weights file (in resources/) is on disk.
# ============================================================================
proto_resolve_weights_dir puffin
PUFFIN_REPO_DIR="$WEIGHTS_DIR/puffin_repo"

if [ ! -d "$PUFFIN_REPO_DIR/.git" ]; then
    echo "Cloning Puffin repository..."
    git clone --depth 1 https://github.com/jzhoulab/puffin.git "$PUFFIN_REPO_DIR"
else
    echo "Puffin repository already cloned."
fi

echo "Puffin setup complete!"
