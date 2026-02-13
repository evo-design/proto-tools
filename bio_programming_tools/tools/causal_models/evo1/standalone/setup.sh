#!/bin/bash
# Setup script for Evo1 standalone environment
set -euo pipefail

echo "Setting up Evo1 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing torch..."
uv pip install torch --torch-backend=auto

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "If installation fails, follow upstream setup guide:"
echo "  - https://github.com/evo-design/evo"

echo "Evo1 setup complete!"
