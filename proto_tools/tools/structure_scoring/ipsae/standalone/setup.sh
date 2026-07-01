#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up IPSAE standalone environment..."

pip install uv
uv pip install numpy

# Download ipsae.py from DunbrackLab, pinned to a specific commit for reproducibility.
# curl is on PATH via the host or the foundation env.
# -f makes curl exit nonzero on HTTP 4xx/5xx so a 404 doesn't silently land HTML in ipsae.py.
IPSAE_COMMIT="${IPSAE_COMMIT:-6174cf9e71cb1bd660cc805856a18c4871a6dec3}"
IPSAE_URL="https://raw.githubusercontent.com/DunbrackLab/IPSAE/${IPSAE_COMMIT}/ipsae.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/ipsae.py" ]; then
    echo "Downloading ipsae.py..."
    curl -fsSL "$IPSAE_URL" -o "$SCRIPT_DIR/ipsae.py"
fi

echo "IPSAE setup complete!"
