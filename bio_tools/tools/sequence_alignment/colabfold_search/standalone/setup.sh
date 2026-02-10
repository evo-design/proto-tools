#!/bin/bash
# Setup script for ColabFold Search standalone environment
set -euo pipefail

echo "Setting up ColabFold Search standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "ColabFold Search setup complete!"
