#!/bin/bash
# Setup script for Structure Metrics standalone environment
set -euo pipefail

echo "Setting up Structure Metrics standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Structure Metrics setup complete!"
