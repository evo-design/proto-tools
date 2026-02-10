#!/bin/bash
set -euo pipefail

echo "Setting up Prodigal standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Prodigal setup complete!"
