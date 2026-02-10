#!/bin/bash
set -euo pipefail

echo "Setting up BLAST standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Downloading BLAST+ binaries..."
# Walk up from this script's directory to find infra/install_binary.py at the tools/ root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [ ! -f "$SEARCH_DIR/infra/install_binary.py" ]; do
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
    if [ "$SEARCH_DIR" = "/" ]; then
        echo "ERROR: Could not find infra/install_binary.py" >&2
        exit 1
    fi
done
python "$SEARCH_DIR/infra/install_binary.py" blast

echo "BLAST setup complete!"
