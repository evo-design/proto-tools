#!/bin/bash
# Setup script for Evo1 standalone environment
set -euo pipefail

echo "Setting up Evo1 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing torch..."
# Pin torch to 2.7.1: flash-attn pre-built wheels are ABI-sensitive and only
# work with the exact torch version they were compiled against.
# Update both this pin and flash-attn in requirements.txt together.
uv pip install torch==2.7.1 --torch-backend=auto

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto --no-build-isolation-package flash-attn

echo "If installation fails, follow upstream setup guide:"
echo "  - https://github.com/evo-design/evo"

echo "Evo1 setup complete!"
