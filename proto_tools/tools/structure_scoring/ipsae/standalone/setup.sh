#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up IPSAE standalone environment..."

pip install uv
uv pip install numpy

# Download ipsae.py from DunbrackLab, pinned to a specific commit for reproducibility.
# Install into the venv so the script shares the environment's lifecycle: a rebuilt env
# re-downloads it, and a checkout that did not run setup.sh cannot appear already set up.
IPSAE_COMMIT="${IPSAE_COMMIT:-6174cf9e71cb1bd660cc805856a18c4871a6dec3}"
IPSAE_SHA256="${IPSAE_SHA256:-10cf9b08c68c91e06cb28526cf2026f47a3980c9048fd3226d13e3304eaf1c27}"
INSTALL_DIR="${PREFIX:-$VENV_PATH}/share/ipsae"
mkdir -p "$INSTALL_DIR"

proto_download_verified \
    "https://raw.githubusercontent.com/DunbrackLab/IPSAE/${IPSAE_COMMIT}/ipsae.py" \
    "$INSTALL_DIR/ipsae.py" \
    "$IPSAE_SHA256"

echo "IPSAE setup complete!"
