#!/bin/bash
# Setup script for mmseqs2-homology-search standalone environment
set -euo pipefail

echo "Setting up mmseqs2-homology-search standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Installing MMseqs2 binary (GPU-capable)..."
# Walk up from this script's directory to find utils/install_binary.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [ ! -f "$SEARCH_DIR/utils/install_binary.py" ]; do
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
    if [ "$SEARCH_DIR" = "/" ]; then
        echo "ERROR: Could not find utils/install_binary.py" >&2
        exit 1
    fi
done
python "$SEARCH_DIR/utils/install_binary.py" mmseqs2_homology_search

echo "mmseqs2-homology-search setup complete!"
