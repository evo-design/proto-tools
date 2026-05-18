#!/bin/bash
# Setup script for Malinois standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Malinois standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "Malinois setup complete!"
