#!/bin/bash
# Setup script for MinCED standalone environment
set -euo pipefail

echo "Setting up MinCED standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Installing MinCED via conda..."
conda install -y -c bioconda minced

echo "MinCED setup complete!"
