#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up FoldMason standalone environment..."

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Downloading FoldMason binary..."
proto_install_binary foldmason

echo "FoldMason setup complete!"
