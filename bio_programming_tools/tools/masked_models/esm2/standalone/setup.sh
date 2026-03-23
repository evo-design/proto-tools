#!/bin/bash
# Setup script for ESM2 standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up ESM2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "ESM2 setup complete!"
