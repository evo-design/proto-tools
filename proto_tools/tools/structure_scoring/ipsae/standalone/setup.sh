#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up IPSAE standalone environment..."

pip install uv
uv pip install numpy

# Download ipsae.py from DunbrackLab
IPSAE_URL="https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/ipsae.py" ]; then
    echo "Downloading ipsae.py..."
    curl -sSL "$IPSAE_URL" -o "$SCRIPT_DIR/ipsae.py"
fi

echo "IPSAE setup complete!"
