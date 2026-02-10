#!/bin/bash
# Setup script for BioEmu standalone environment
set -euo pipefail

echo "Setting up BioEmu standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "BioEmu setup complete!"
