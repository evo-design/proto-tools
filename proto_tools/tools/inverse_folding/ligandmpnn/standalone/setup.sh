#!/bin/bash
# Setup script for LigandMPNN standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up LigandMPNN standalone environment..."

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

# Download the bundled weights, the published LigandMPNN checkpoints trained with 0.10 Å and 0.20 Å
# backbone noise; either implementation loads either. Keep in sync with _BUNDLED_CHECKPOINTS in
# inference.py. implementation="original" fetches its code and side-chain packer on first use
# (_ensure_legacy_assets).
proto_resolve_weights_dir ligandmpnn
proto_download_verified "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt" \
    "${WEIGHTS_DIR}/ligandmpnn_v_32_010_25.pt" \
    "161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970"
proto_download_verified "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt" \
    "${WEIGHTS_DIR}/ligandmpnn_v_32_020_25.pt" \
    "42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981"

echo "LigandMPNN setup complete!"
