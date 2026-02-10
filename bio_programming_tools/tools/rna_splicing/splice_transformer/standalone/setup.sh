#!/bin/bash
set -euo pipefail

echo "Setting up SpliceTransformer standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt --torch-backend=auto

echo "SpliceTransformer setup complete!"
