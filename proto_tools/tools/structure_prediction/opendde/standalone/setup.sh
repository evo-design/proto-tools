#!/bin/bash
# Setup script for the OpenDDE standalone environment.
#
# OpenDDE is an all-atom biomolecular structure-prediction model (Apache-2.0,
# https://github.com/aurekaresearch/OpenDDE). It ships as the `opendde` pip
# package; the `opendde[gpu]` extra adds cuequivariance + triton (Linux x86_64,
# CUDA 12.6). It requires Python 3.11-3.13 (we pin 3.12) and torch 2.7.1.
#
# Weights (opendde.pt, opendde_abag.pt) plus the runtime `common/` assets are
# pulled from HuggingFace (aurekaresearch/OpenDDE — public, Apache-2.0) into the
# resolved OPENDDE_ROOT_DIR, which OpenDDE reads at runtime. Expected layout:
#   $OPENDDE_ROOT_DIR/checkpoint/opendde.pt
#   $OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt
#   $OPENDDE_ROOT_DIR/common/...
# (search_database/ is only needed for --use_template / --use_rna_msa, which are
#  off by default in the tool layer, so it is intentionally not downloaded here.)
set -euo pipefail
source standalone_helpers.sh

echo "Setting up OpenDDE standalone environment..."

echo "Installing uv package manager..."
pip install uv

# OpenDDE pins torch==2.7.1. Install it through the compute-matched proto index so
# the CUDA build matches the detected platform, pinning the exact version here so
# that `uv pip install -r requirements.txt` (which brings opendde[gpu]'s torch==2.7.1
# requirement) does not swap in a non-CUDA wheel. The [gpu] extra's cuequivariance /
# triton target CUDA 12.6; if the detected platform differs, override RECOMMENDED_*
# or adjust the pin.
proto_install_pytorch "torch==2.7.1"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# ─── Weights + runtime assets ───────────────────────────────────────────────
# OpenDDE reads checkpoints and common assets from OPENDDE_ROOT_DIR. Resolve the
# managed weights directory and treat it as OPENDDE_ROOT_DIR (inference.py resolves
# the same path at runtime via resolve_weights_dir("opendde")).
proto_resolve_weights_dir opendde
CKPT_DIR="${WEIGHTS_DIR}/checkpoint"
mkdir -p "$CKPT_DIR"

HF_REPO="aurekaresearch/OpenDDE"

# Fetch checkpoints (into CKPT_DIR) + common/ assets via huggingface_hub — idempotent, resumable, no curl needed.
echo "Downloading OpenDDE checkpoints and common assets..."
uv pip install "huggingface_hub"
python - <<PY
from huggingface_hub import hf_hub_download, snapshot_download

repo = "${HF_REPO}"
for ckpt in ("opendde.pt", "opendde_abag.pt"):
    hf_hub_download(repo, ckpt, local_dir="${CKPT_DIR}")
snapshot_download(repo, allow_patterns=["common/*", "common/**"], local_dir="${WEIGHTS_DIR}")
PY

echo "OpenDDE setup complete! (OPENDDE_ROOT_DIR=${WEIGHTS_DIR})"
