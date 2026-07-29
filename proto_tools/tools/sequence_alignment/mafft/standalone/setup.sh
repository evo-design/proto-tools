#!/bin/bash
set -euo pipefail

echo "Setting up MAFFT standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Installing MAFFT from bioconda..."

# bioconda has no osx-arm64 build, so Apple Silicon takes the osx-64 one under Rosetta 2.
MAMBA_EXTRA_ARGS=()
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "Detected macOS arm64 — using osx-64 packages via Rosetta 2..."
    MAMBA_EXTRA_ARGS=(--platform osx-64)
fi

"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge -c bioconda "${MAMBA_EXTRA_ARGS[@]}" mafft

echo "MAFFT setup complete!"
