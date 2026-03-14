#!/bin/bash
# Setup script for ESM-IF/ProteinDPO standalone environment
set -euo pipefail

echo "Setting up ESM-IF standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --torch-backend=auto

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Download model weights
WEIGHTS_DIR="${ESMIF_WEIGHTS_DIR:-${VENV_PATH}/weights}"
mkdir -p "$WEIGHTS_DIR"

# Download ESM-IF1 vanilla weights
ESMIF_WEIGHTS_FILE="${WEIGHTS_DIR}/esm_if1_gvp4_t16_142M_UR50.pt"
if [ ! -f "$ESMIF_WEIGHTS_FILE" ]; then
    echo "Downloading ESM-IF1 vanilla weights..."
    wget -q -O "$ESMIF_WEIGHTS_FILE" \
        "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt" || {
        echo "WARNING: Failed to download ESM-IF1 weights."
    }
else
    echo "ESM-IF1 weights already present."
fi

# Download ProteinDPO weights from Zenodo
PROTEINDPO_WEIGHTS_FILE="${WEIGHTS_DIR}/paired_weights.pt"
if [ ! -f "$PROTEINDPO_WEIGHTS_FILE" ]; then
    echo "Downloading ProteinDPO weights from Zenodo..."
    wget -q -O "${WEIGHTS_DIR}/paired_weights.pt.zip" \
        "https://zenodo.org/records/11218181/files/paired_weights.pt.zip" || {
        echo "WARNING: Failed to download ProteinDPO weights."
    }
    if [ -f "${WEIGHTS_DIR}/paired_weights.pt.zip" ]; then
        cd "$WEIGHTS_DIR" && unzip -o paired_weights.pt.zip && rm -f paired_weights.pt.zip
        cd -
    fi
else
    echo "ProteinDPO weights already present."
fi

echo "ESM-IF setup complete!"
