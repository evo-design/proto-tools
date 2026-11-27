#!/bin/bash
# Setup script for PARADE standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up PARADE standalone environment..."

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "PARADE setup complete!"
