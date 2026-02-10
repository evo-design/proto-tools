#!/bin/bash
# Setup script for LigandMPNN standalone environment
set -euo pipefail

echo "Setting up LigandMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "Downloading LigandMPNN model weights..."
foundry install ligandmpnn

echo "LigandMPNN setup complete!"
