#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up BLAST standalone environment..."

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Downloading BLAST+ binaries..."
proto_install_binary blast

echo "BLAST setup complete!"
