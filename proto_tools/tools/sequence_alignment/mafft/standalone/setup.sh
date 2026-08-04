#!/bin/bash
set -euo pipefail

echo "Setting up MAFFT standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Installing MAFFT from conda-forge..."

# conda-forge ships mafft for linux-64, linux-aarch64, osx-64 and osx-arm64, so the env stays native everywhere.
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge -c bioconda mafft

echo "MAFFT setup complete!"
