#!/bin/bash
# Setup script for ViennaRNA standalone environment
set -euo pipefail

echo "Setting up ViennaRNA standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "ViennaRNA setup complete!"
