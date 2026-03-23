#!/bin/bash
# Setup script for ProteinMPNN standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up ProteinMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_cuda_toolkit

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

bpt_install_jax PROTEINMPNN

# ============================================================================
# Download AbMPNN weights (antibody-optimized ProteinMPNN) into ColabDesign's
# weights directory so mk_mpnn_model(model_name="abmpnn") finds them.
# ============================================================================
COLABDESIGN_WEIGHTS_DIR=$(python -c "from colabdesign.mpnn.weights import __file__ as f; import os; print(os.path.dirname(f))")
ABMPNN_PKL="${COLABDESIGN_WEIGHTS_DIR}/abmpnn.pkl"

if [ ! -f "$ABMPNN_PKL" ]; then
    echo "Downloading AbMPNN weights..."
    curl -fsSL -o "$ABMPNN_PKL" \
        "https://github.com/SantiagoMille/germinal/raw/main/colabdesign/colabdesign/mpnn/weights_abmpnn/abmpnn.pkl"
    echo "AbMPNN weights installed to ${ABMPNN_PKL}"
else
    echo "AbMPNN weights already present at ${ABMPNN_PKL}"
fi

echo "ProteinMPNN setup complete!"
