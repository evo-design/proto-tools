#!/bin/bash
# Setup script for AbLang standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AbLang standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "AbLang setup complete!"
