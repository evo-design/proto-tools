#!/bin/bash
# Setup script for FAMPNN standalone environment
set -euo pipefail

echo "Setting up FAMPNN standalone environment..."

echo "Installing uv package manager..."
pip install uv

# ============================================================================
# Install CUDA toolkit via micromamba (PyTorch needs CUDA libs)
# ============================================================================
DETECTED_CUDA_MAJOR="${DETECTED_CUDA_VERSION:-12}"
CUDA_TOOLKIT_VERSION="${DETECTED_CUDA_MAJOR}.*"
echo "Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} locally via micromamba..."
if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}" \
    "cudnn"; then
    echo "ERROR: Failed to install CUDA toolkit via micromamba"
    exit 1
fi

# ============================================================================
# Install PyTorch with appropriate CUDA variant
# ============================================================================
echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch}"

uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

# Install torch-geometric (needed by FAMPNN)
uv pip install torch-geometric

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# ============================================================================
# Install FAMPNN from source (upstream repo lacks __init__.py files,
# so find_packages() returns nothing and normal pip install is empty)
# ============================================================================
FAMPNN_REPO_DIR="${VENV_PATH}/fampnn_repo"
if [ ! -d "$FAMPNN_REPO_DIR/.git" ]; then
    echo "Cloning FAMPNN repository..."
    git clone --depth 1 https://github.com/richardshuai/fampnn.git "$FAMPNN_REPO_DIR"
else
    echo "FAMPNN repository already cloned."
fi

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
WEIGHTS_DIR="${FAMPNN_WEIGHTS_DIR:-${VENV_PATH}/weights}"
mkdir -p "$WEIGHTS_DIR"

REPO_BASE="https://github.com/richardshuai/fampnn/raw/main/weights"

for WEIGHT_FILE in fampnn_0_0.pt fampnn_0_3.pt fampnn_0_3_cath.pt; do
    if [ ! -f "${WEIGHTS_DIR}/${WEIGHT_FILE}" ]; then
        echo "Downloading ${WEIGHT_FILE}..."
        curl -fsSL -o "${WEIGHTS_DIR}/${WEIGHT_FILE}" "${REPO_BASE}/${WEIGHT_FILE}"
    else
        echo "${WEIGHT_FILE} already present."
    fi
done

echo "FAMPNN setup complete!"
