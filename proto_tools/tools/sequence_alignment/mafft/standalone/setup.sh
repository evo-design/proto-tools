#!/bin/bash
set -euo pipefail

echo "Setting up MAFFT standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Downloading MAFFT binaries..."
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
python "$SEARCH_DIR/utils/install_binary.py" mafft

echo "MAFFT setup complete!"
