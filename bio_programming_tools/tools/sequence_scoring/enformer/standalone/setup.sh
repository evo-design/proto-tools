#!/bin/bash
# Setup script for Enformer standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Enformer standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "Enformer setup complete!"
