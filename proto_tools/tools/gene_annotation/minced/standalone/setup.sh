#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up MinCED standalone environment..."

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# minced is a Java wrapper script that exec's `java -jar minced.jar`.
"$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -y openjdk

echo "Downloading MinCED..."
proto_install_binary minced

echo "MinCED setup complete!"
