#!/bin/bash
# Setup script for ProteinMPNN standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up ProteinMPNN standalone environment..."

proto_install_cuda_toolkit

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# ColabDesign v1.1.3 uses jax.tree_map, removed in JAX 0.10.
export PROTEINMPNN_JAX_SPEC="${PROTEINMPNN_JAX_SPEC:-jax[cuda12]==0.5.3}"
proto_install_jax PROTEINMPNN

# ============================================================================
# Download AbMPNN weights (antibody-optimized ProteinMPNN) into ColabDesign's
# weights directory so mk_mpnn_model(model_name="abmpnn") finds them. The
# checkpoint is hosted outside ColabDesign, so it is pinned by digest: a
# truncated or substituted file aborts setup instead of being cached as
# present, and the environment is never marked healthy without it.
# ============================================================================
ABMPNN_URL="https://github.com/SantiagoMille/germinal/raw/1e1c1a5b79884ae45abae030c9df90d9423a990a/colabdesign/colabdesign/mpnn/weights_abmpnn/abmpnn.pkl"
ABMPNN_SHA256="751c461af7695f38c96c99ff7bc9deb58e4cf8ad8d21f38a82cfbf1e6c1fcfe2"
COLABDESIGN_WEIGHTS_DIR=$(python -c "from colabdesign.mpnn.weights import __file__ as f; import os; print(os.path.dirname(f))")
ABMPNN_PKL="${COLABDESIGN_WEIGHTS_DIR}/abmpnn.pkl"

proto_download_verified "$ABMPNN_URL" "$ABMPNN_PKL" "$ABMPNN_SHA256"

echo "ProteinMPNN setup complete!"
