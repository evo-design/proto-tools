#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Segmasker standalone environment..."

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Downloading BLAST+ binaries (includes segmasker)..."
proto_install_binary segmasker

echo "Segmasker setup complete!"
