#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Pangolin standalone environment..."
echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# Pangolin ships its model weights inside the pip package. Install with --no-deps so
# its loose setup.py cannot override the driver-aware torch installed above.
echo "Installing Pangolin (with bundled model weights)..."
uv pip install --no-deps "git+https://github.com/tkzeng/Pangolin.git"

echo "Pangolin setup complete!"
