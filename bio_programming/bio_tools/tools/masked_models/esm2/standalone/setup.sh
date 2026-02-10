#!/bin/bash
# Setup script for ESM2 standalone environment
set -euo pipefail

echo "Setting up ESM2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "ESM2 setup complete!"
