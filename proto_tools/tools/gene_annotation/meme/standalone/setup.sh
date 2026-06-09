#!/bin/bash
set -euo pipefail

echo "Setting up MEME (FIMO) standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "MEME (FIMO) setup complete!"
