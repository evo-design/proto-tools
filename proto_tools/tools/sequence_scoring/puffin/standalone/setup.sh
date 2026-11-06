#!/bin/bash
# Setup script for Puffin standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Puffin standalone environment..."

proto_install_pytorch

# selene-sdk pulls pytabix, whose sdist-built C extension needs zlib.h.
"$MAMBA_BIN" install -p "$VENV_PATH" -c conda-forge -y zlib

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# ============================================================================
# Clone the upstream Puffin repo so the Puffin class can be imported and so
# the bundled puffin.pth weights file (in resources/) is on disk.
# ============================================================================
proto_resolve_weights_dir puffin
PUFFIN_REPO_DIR="$WEIGHTS_DIR/puffin_repo"
PUFFIN_COMMIT="9bc81eb922f44556879d1d2afd4c757a1d6c291e"

echo "Checking out pinned Puffin commit ${PUFFIN_COMMIT}..."
# Restores any repo file a purge deleted, puffin.pth included
proto_ensure_git_checkout https://github.com/jzhoulab/puffin.git "$PUFFIN_REPO_DIR" "$PUFFIN_COMMIT"

echo "Puffin setup complete!"
