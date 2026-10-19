#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Foldseek standalone environment..."

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Downloading Foldseek binary..."
proto_install_binary foldseek

echo "Foldseek setup complete!"
