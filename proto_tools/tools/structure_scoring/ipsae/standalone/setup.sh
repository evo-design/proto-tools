#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up IPSAE standalone environment..."

pip install uv
uv pip install numpy

# Print the SHA-256 of a file, or nothing when it is missing (Linux sha256sum, macOS shasum).
sha256_of() {
    [ -f "$1" ] || return 0
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# Download ipsae.py from DunbrackLab, pinned to a specific commit for reproducibility.
# curl is on PATH via the host or the foundation env.
# -f makes curl exit nonzero on HTTP 4xx/5xx so a 404 doesn't silently land HTML in ipsae.py.
# Install into the venv so the script shares the environment's lifecycle: a rebuilt env
# re-downloads it, and a checkout that did not run setup.sh cannot appear already set up.
# The download is checked against a pinned digest rather than trusted for merely existing,
# so a truncated or altered file is re-fetched instead of being cached as good.
IPSAE_COMMIT="${IPSAE_COMMIT:-6174cf9e71cb1bd660cc805856a18c4871a6dec3}"
IPSAE_SHA256="${IPSAE_SHA256:-10cf9b08c68c91e06cb28526cf2026f47a3980c9048fd3226d13e3304eaf1c27}"
IPSAE_URL="https://raw.githubusercontent.com/DunbrackLab/IPSAE/${IPSAE_COMMIT}/ipsae.py"
INSTALL_DIR="${PREFIX:-$VENV_PATH}/share/ipsae"
mkdir -p "$INSTALL_DIR"

if [ "$(sha256_of "$INSTALL_DIR/ipsae.py")" != "$IPSAE_SHA256" ]; then
    echo "Downloading ipsae.py..."
    curl -fsSL "$IPSAE_URL" -o "$INSTALL_DIR/ipsae.py.part"
    DOWNLOADED_SHA256="$(sha256_of "$INSTALL_DIR/ipsae.py.part")"
    if [ "$DOWNLOADED_SHA256" != "$IPSAE_SHA256" ]; then
        rm -f "$INSTALL_DIR/ipsae.py.part"
        echo "ERROR: ipsae.py checksum mismatch (expected $IPSAE_SHA256, got $DOWNLOADED_SHA256)" >&2
        exit 1
    fi
    mv "$INSTALL_DIR/ipsae.py.part" "$INSTALL_DIR/ipsae.py"
fi

echo "IPSAE setup complete!"
