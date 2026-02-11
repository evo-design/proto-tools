#!/bin/bash
# Setup script for AlphaGenome standalone environment
set -euo pipefail

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: AlphaGenome is not supported on aarch64."
    echo "AlphaGenome requires JAX with CUDA support (jax[cuda12_local]),"
    echo "which does not provide aarch64 wheels."
    exit 1
fi

echo "Setting up AlphaGenome standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

REPO_DIR="${VENV_PATH}/src/alphagenome_research"
if [ -d "$REPO_DIR" ]; then
  echo "Updating existing alphagenome_research clone..."
  git -C "$REPO_DIR" pull origin main
else
  echo "Cloning alphagenome_research..."
  git clone https://github.com/google-deepmind/alphagenome_research.git "$REPO_DIR"
fi
echo "Installing alphagenome_research from local clone..."
uv pip install "$REPO_DIR"

echo "AlphaGenome setup complete!"
