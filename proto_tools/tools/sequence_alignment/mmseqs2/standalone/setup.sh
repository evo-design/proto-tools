#!/bin/bash
# Setup script for the unified mmseqs2 standalone environment.
# Hosts all four tools: mmseqs2-search-proteins, -search-genomes, -clustering,
# and -homology-search. Ships the GPU-capable MMseqs2 binary (a strict superset
# of the CPU build) plus colabfold for the homology-search MSA pipeline.
set -euo pipefail

echo "Setting up mmseqs2 toolkit standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Installing MMseqs2 binary (GPU-capable; runs CPU calls without --gpu)..."
# Walk up from this script's directory to find utils/install_binary.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [ ! -f "$SEARCH_DIR/utils/install_binary.py" ]; do
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
    if [ "$SEARCH_DIR" = "/" ]; then
        echo "Error: Could not find utils/install_binary.py"
        exit 1
    fi
done
python "$SEARCH_DIR/utils/install_binary.py" mmseqs2

echo "mmseqs2 toolkit setup complete!"
