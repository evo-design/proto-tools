#!/bin/bash
# Setup script for ESMFold standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up ESMFold standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install transformers biopython

echo "ESMFold setup complete!"
