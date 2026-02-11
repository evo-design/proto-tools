#!/bin/bash
# Setup script for RFdiffusion3 standalone environment
set -euo pipefail

echo "Setting up RFdiffusion3 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "Downloading rfdiffusion3 model weights..."
foundry install rfd3

echo "RFdiffusion3 setup complete!"
