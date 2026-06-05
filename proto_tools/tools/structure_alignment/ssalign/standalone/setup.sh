#!/bin/bash
set -euo pipefail
source standalone_helpers.sh
echo "Setting up SSAlign standalone environment..."
echo "Installing uv package manager..."
pip install uv
proto_install_pytorch
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt
echo "SSAlign setup complete!"
