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

# TODO: Remove this workaround once the upstream bug is fixed.
# Pin to 548bbfe (Feb 3, 2026) — the last commit before the Feb 5 refactor
# that introduced a _junctions_apply_fn unpacking bug (cc16ac1).
ALPHAGENOME_RESEARCH_COMMIT="548bbfecd78874367af51f09944a0a361fd1ac9b"
REPO_DIR="${VENV_PATH}/src/alphagenome_research"
if [ -d "$REPO_DIR" ]; then
  echo "Updating existing alphagenome_research clone..."
  git -C "$REPO_DIR" fetch origin "$ALPHAGENOME_RESEARCH_COMMIT"
  git -C "$REPO_DIR" checkout "$ALPHAGENOME_RESEARCH_COMMIT"
else
  echo "Cloning alphagenome_research at pinned commit..."
  git clone https://github.com/google-deepmind/alphagenome_research.git "$REPO_DIR"
  git -C "$REPO_DIR" checkout "$ALPHAGENOME_RESEARCH_COMMIT"
fi
echo "Installing alphagenome_research from local clone..."
uv pip install "$REPO_DIR"

echo "AlphaGenome setup complete!"
