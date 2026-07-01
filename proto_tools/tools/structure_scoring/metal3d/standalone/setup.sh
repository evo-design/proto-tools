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
declare -A WEIGHTS=(
    ["metal3d_cat.pth"]="${REPO_BASE}/metal3d_cat.pth"
    ["metal3d_clean.pth"]="${REPO_BASE}/metal3d_clean.pth"
    ["metal_0.5A_v3_d0.2_16Abox.pth"]="${REPO_BASE}/metal_0.5A_v3_d0.2_16Abox.pth"
)

for WEIGHT_FILE in "${!WEIGHTS[@]}"; do
    if [ ! -f "${WEIGHTS_DIR}/${WEIGHT_FILE}" ]; then
        echo "Downloading ${WEIGHT_FILE}..."
        curl -fsSL -o "${WEIGHTS_DIR}/${WEIGHT_FILE}" "${WEIGHTS[$WEIGHT_FILE]}"
    else
        echo "${WEIGHT_FILE} already present."
    fi
done

echo "Metal3D setup complete!"
