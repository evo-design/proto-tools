#!/bin/bash
# Setup script for Enformer standalone environment
set -euo pipefail

echo "Setting up Enformer standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto

echo "Enformer setup complete!"
