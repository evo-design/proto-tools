#!/bin/bash
#
# DeepPBS (https://github.com/timkartar/DeepPBS) is not on PyPI. The public
# BSD-3 repo is self-contained: it bundles the X3DNA/DSSR binaries, the
# process/predict config JSONs, and the trained inference weights. This script
# clones a pinned revision into the managed weights cache and installs it
# editable; no GitHub sign-in is required for the public repository. To reuse an
# existing checkout instead, point PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR at it.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up DeepPBS specificity standalone environment..."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv package manager..."
  pip install uv
fi

echo "Installing PyTorch stack for DeepPBS..."
uv pip install "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0" --torch-backend=auto

echo "Installing PyG dependencies..."
# Derive the PyG wheel tag from THIS env's torch CUDA build (e.g. 12.1 -> cu121).
# PYTHONNOUSERSITE=1 prevents a stray user-site torch (e.g. a different CUDA build in
# ~/.local) from shadowing the venv torch and mis-selecting a CPU/mismatched wheel,
# which yields ABI-broken torch_cluster (_version_cuda.so: undefined symbol) at runtime.
PYG_TAG="$(PYTHONNOUSERSITE=1 "$PYTHON_EXE" - <<'PY'
import torch
cuda = torch.version.cuda
print("cu" + cuda.replace(".", "") if cuda else "cpu")
PY
)"
PYG_URL="https://data.pyg.org/whl/torch-2.3.0+${PYG_TAG}.html"
PYTHONNOUSERSITE=1 uv pip install torch_scatter torch_sparse torch_cluster torch_geometric -f "${PYG_URL}"

echo "Installing DeepPBS Python dependencies from requirements.txt..."
uv pip install -r requirements.txt
uv pip install pdb2pqr

# ─── Provision the DeepPBS checkout (pinned clone into the weights cache) ────
# Pinned revision of timkartar/DeepPBS; bump deliberately when updating.
DEEPPBS_REPO_URL="https://github.com/timkartar/DeepPBS.git"
DEEPPBS_COMMIT="8bfb211dd67f02877841f6f33aa493ddf7daedf9"

# Resolve (and create) the managed cache dir the runtime already searches.
proto_resolve_weights_dir deeppbs_specificity   # sets $WEIGHTS_DIR
DEEPPBS_DIR="$WEIGHTS_DIR"

if [ ! -d "${DEEPPBS_DIR}/.git" ]; then
  echo "[deeppbs] Cloning DeepPBS into ${DEEPPBS_DIR}"
  # Clone into the freshly created, empty cache dir. On failure (offline, proxy,
  # firewall) emit the skip sentinel so the env build / tests degrade cleanly.
  if ! git clone --quiet "$DEEPPBS_REPO_URL" "$DEEPPBS_DIR"; then
    {
      echo "[proto-tools] ASSET_NOT_AVAILABLE: deeppbs_specificity:repo"
      echo "Could not clone ${DEEPPBS_REPO_URL}."
      echo "The repository is public, so sign-in is normally not required. If you are"
      echo "behind a proxy or firewall, authenticate git (e.g. 'gh auth setup-git') or"
      echo "clone it manually and point the tool at the checkout:"
      echo "  git clone ${DEEPPBS_REPO_URL} /path/to/DeepPBS"
      echo "  git -C /path/to/DeepPBS checkout ${DEEPPBS_COMMIT}"
      echo "  export PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR=/path/to/DeepPBS"
    } >&2
    exit 64
  fi
fi

# Pin to the recorded revision (already present after a full clone; the fetch
# only matters when bumping DEEPPBS_COMMIT on a warm cache).
git -C "$DEEPPBS_DIR" fetch --quiet origin || true
git -C "$DEEPPBS_DIR" checkout --quiet "$DEEPPBS_COMMIT"
echo "[deeppbs] DeepPBS pinned to ${DEEPPBS_COMMIT}"
# The bundled X3DNA/DSSR binaries must stay executable after a fresh clone.
chmod +x "${DEEPPBS_DIR}/dependencies/bin/"* 2>/dev/null || true

echo "Installing deeppbs package (editable) from the checkout..."
uv pip install -e "$DEEPPBS_DIR"

echo "DeepPBS specificity setup complete."
