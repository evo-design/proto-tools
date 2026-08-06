#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Metal3D standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit
proto_install_pytorch ""

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

proto_resolve_weights_dir metal3d

DEVA_COMMIT="${DEVA_COMMIT:-ee771f6730d170c83d8e63074be3bdd761b21dee}"
REPO_BASE="https://raw.githubusercontent.com/gelnesr/dEVA/${DEVA_COMMIT}/models/metal3d/weights"
# Digests of the checkpoints at DEVA_COMMIT. Checked on every run so a truncated
# download is re-fetched rather than kept because the path exists: a short
# metal3d_clean.pth still carries the torch zip magic and only fails at torch.load().
declare -A WEIGHT_SHA256=(
    ["metal3d_cat.pth"]="e86bea7ebc3513d603626b5d080ed9ab5aef869be916fbaf47446b3a2004ab27"
    ["metal3d_clean.pth"]="166e5acb62b39fd722aeebc3eb5855328560f78cea855fe68285214b44760c47"
    ["metal_0.5A_v3_d0.2_16Abox.pth"]="b5a1b0c5ea6c5dcdedfae4e24b8461da107ea78c8b96c7db8f44db532c87246f"
)

for WEIGHT_FILE in "${!WEIGHT_SHA256[@]}"; do
    proto_download_verified \
        "${REPO_BASE}/${WEIGHT_FILE}" \
        "${WEIGHTS_DIR}/${WEIGHT_FILE}" \
        "${WEIGHT_SHA256[$WEIGHT_FILE]}"
done

echo "Metal3D setup complete!"
