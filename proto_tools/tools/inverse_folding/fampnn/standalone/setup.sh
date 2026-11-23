#!/bin/bash
# Setup script for FAMPNN standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up FAMPNN standalone environment..."

proto_install_cuda_toolkit
proto_install_pytorch "" torchvision

# Install torch-geometric (needed by FAMPNN)
uv pip install torch-geometric

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# ============================================================================
# Install FAMPNN from source (upstream repo lacks __init__.py files,
# so find_packages() returns nothing and normal pip install is empty)
# ============================================================================
FAMPNN_REPO_DIR="${VENV_PATH}/fampnn_repo"
FAMPNN_COMMIT="aaf788b1502ad95d5c5a84455cfc53f2544f3b45"
if [ ! -d "$FAMPNN_REPO_DIR/.git" ]; then
    echo "Cloning FAMPNN repository..."
    git clone https://github.com/richardshuai/fampnn.git "$FAMPNN_REPO_DIR"
else
    echo "FAMPNN repository already cloned."
fi

echo "Checking out pinned FAMPNN commit ${FAMPNN_COMMIT}..."
git -C "$FAMPNN_REPO_DIR" fetch origin "$FAMPNN_COMMIT"
git -C "$FAMPNN_REPO_DIR" checkout "$FAMPNN_COMMIT"

echo "Creating missing __init__.py files in FAMPNN package..."
find "$FAMPNN_REPO_DIR/fampnn" -type d -exec sh -c '
    for dir; do
        if [ ! -f "$dir/__init__.py" ]; then
            touch "$dir/__init__.py"
        fi
    done
' _ {} +

# Also handle openfold dependency (bundled in repo)
if [ -d "$FAMPNN_REPO_DIR/openfold" ]; then
    find "$FAMPNN_REPO_DIR/openfold" -type d -exec sh -c '
        for dir; do
            if [ ! -f "$dir/__init__.py" ]; then
                touch "$dir/__init__.py"
            fi
        done
    ' _ {} +
fi

echo "Installing FAMPNN in editable mode..."
pip install -e "$FAMPNN_REPO_DIR" --no-deps

# ============================================================================
# Download model weights if not already present
# ============================================================================
proto_resolve_weights_dir fampnn

REPO_BASE="https://github.com/richardshuai/fampnn/raw/aaf788b1502ad95d5c5a84455cfc53f2544f3b45/weights"

# proto_download_verified checks each checkpoint's SHA-256 before moving it into place, so neither
# an interrupted run nor a bad download leaves a file a later setup would take as present.
declare -A WEIGHT_SHA256=(
    [fampnn_0_0.pt]=afbdfda29e6f2a1bd340971bb226638afb1bf460cfbc29f115d3b5964622c006
    [fampnn_0_3.pt]=8969b3f1f3c941178076c7800952595a18b56fd3828d15bb993d3ef537938a05
    [fampnn_0_3_cath.pt]=81112a9b8d436d9baf5233a3603bac911c75b9782ced59dae5a2726316802218
)
for WEIGHT_FILE in fampnn_0_0.pt fampnn_0_3.pt fampnn_0_3_cath.pt; do
    proto_download_verified "${REPO_BASE}/${WEIGHT_FILE}" "${WEIGHTS_DIR}/${WEIGHT_FILE}" "${WEIGHT_SHA256[$WEIGHT_FILE]}"
done

echo "FAMPNN setup complete!"
