#!/bin/bash
# Setup script for the NA-MPNN specificity standalone environment.
#
# NA-MPNN (https://github.com/baker-laboratory/NA-MPNN) is not on PyPI: the tool
# shells out to a local checkout of the repository, which also ships the
# specificity checkpoint (models/specificity_model/s_70114.pt) in-tree. The repo
# is public and MIT / BSD-3 licensed, so this script clones a pinned revision
# into the managed weights cache; no GitHub sign-in is required for the public
# repository. To reuse an existing checkout instead, point
# PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR (or NAMPNNSpecificityConfig) at it.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up NA-MPNN specificity standalone environment..."

# ─── Provision the NA-MPNN checkout (pinned clone into the weights cache) ────
# Pinned revision of baker-laboratory/NA-MPNN; bump deliberately when updating.
NA_MPNN_REPO_URL="https://github.com/baker-laboratory/NA-MPNN.git"
NA_MPNN_COMMIT="aafa32c87b65046a02db152f6e6235a7078e5fef"

# Resolve (and create) the managed cache dir the runtime already searches.
proto_resolve_weights_dir na_mpnn_specificity   # sets $WEIGHTS_DIR
NA_MPNN_DIR="$WEIGHTS_DIR"

if [ ! -d "${NA_MPNN_DIR}/.git" ]; then
    echo "[na-mpnn] Cloning NA-MPNN into ${NA_MPNN_DIR}"
    # Clone into the freshly created, empty cache dir. On failure (offline,
    # proxy, firewall) emit the skip sentinel so the env build / tests degrade
    # cleanly instead of hard-erroring.
    if ! git clone --quiet "$NA_MPNN_REPO_URL" "$NA_MPNN_DIR"; then
        {
            echo "[proto-tools] ASSET_NOT_AVAILABLE: na_mpnn_specificity:repo"
            echo "Could not clone ${NA_MPNN_REPO_URL}."
            echo "The repository is public, so sign-in is normally not required. If you are"
            echo "behind a proxy or firewall, authenticate git (e.g. 'gh auth setup-git') or"
            echo "clone it manually and point the tool at the checkout:"
            echo "  git clone ${NA_MPNN_REPO_URL} /path/to/NA-MPNN"
            echo "  git -C /path/to/NA-MPNN checkout ${NA_MPNN_COMMIT}"
            echo "  export PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR=/path/to/NA-MPNN"
        } >&2
        exit 64
    fi
fi

# Pin to the recorded revision (the commit is already present after a full
# clone; the fetch only matters when bumping NA_MPNN_COMMIT on a warm cache).
git -C "$NA_MPNN_DIR" fetch --quiet origin || true
git -C "$NA_MPNN_DIR" checkout --quiet "$NA_MPNN_COMMIT"
echo "[na-mpnn] NA-MPNN pinned to ${NA_MPNN_COMMIT}"
# The in-tree specificity checkpoint (models/specificity_model/s_70114.pt) is
# resolved directly by the runtime; no extra placement is needed.

echo "Installing uv package manager..."
pip install uv

echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --torch-backend=auto

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "NA-MPNN specificity setup complete."
