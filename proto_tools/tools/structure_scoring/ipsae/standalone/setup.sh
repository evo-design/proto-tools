#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up IPSAE standalone environment..."

pip install uv
uv pip install numpy

# Download ipsae.py from DunbrackLab, pinned to a specific commit for reproducibility.
# curl is on PATH via the host or the foundation env.
# -f makes curl exit nonzero on HTTP 4xx/5xx so a 404 doesn't silently land HTML in ipsae.py.
# Install into the venv so the script shares the environment's lifecycle: a rebuilt env
# re-downloads it, and a checkout that did not run setup.sh cannot appear already set up.
IPSAE_COMMIT="${IPSAE_COMMIT:-6174cf9e71cb1bd660cc805856a18c4871a6dec3}"
IPSAE_URL="https://raw.githubusercontent.com/DunbrackLab/IPSAE/${IPSAE_COMMIT}/ipsae.py"
INSTALL_DIR="${PREFIX:-$VENV_PATH}/share/ipsae"
mkdir -p "$INSTALL_DIR"
if [ ! -f "$INSTALL_DIR/ipsae.py" ]; then
    echo "Downloading ipsae.py..."
    curl -fsSL "$IPSAE_URL" -o "$INSTALL_DIR/ipsae.py"
fi

echo "IPSAE setup complete!"
