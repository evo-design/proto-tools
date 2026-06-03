#!/bin/bash
set -euo pipefail
source standalone_helpers.sh
echo "Setting up SSAlign standalone environment..."
echo "Installing uv package manager..."
pip install uv
proto_install_pytorch
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt
echo "Downloading Foldseek binary..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; SEARCH_DIR="$SCRIPT_DIR"
while [ ! -f "$SEARCH_DIR/utils/install_binary.py" ]; do
  SEARCH_DIR="$(dirname "$SEARCH_DIR")"; [ "$SEARCH_DIR" = "/" ] && { echo "ERROR: install_binary.py not found" >&2; exit 1; }
done
python "$SEARCH_DIR/utils/install_binary.py" ssalign
echo "SSAlign setup complete!"
