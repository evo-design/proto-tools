#!/bin/bash
# Setup script for Evo1 standalone environment
set -euo pipefail

echo "Setting up Evo1 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true

# ============================================================================
# Install CUDA toolkit via micromamba (needed for flash-attn source builds)
# Micromamba is provided via $MAMBA_BIN environment variable
# ============================================================================
# Determine CUDA toolkit version to install.
# DETECTED_CUDA_VERSION is injected by compute_deps.py (e.g., "12" or "13").
# Cap at 12.8 for CUDA 13+ since toolkit 13 is too new for PyTorch JIT.
DETECTED_CUDA_MAJOR="${DETECTED_CUDA_VERSION:-12}"
if [ "$DETECTED_CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
    CUDA_TOOLKIT_VERSION="12.8"
    echo "Detected CUDA ${DETECTED_CUDA_MAJOR} — using CUDA toolkit ${CUDA_TOOLKIT_VERSION} for compatibility"
else
    CUDA_TOOLKIT_VERSION="${DETECTED_CUDA_MAJOR}.1"
    echo "Detected CUDA ${DETECTED_CUDA_MAJOR} — using CUDA toolkit ${CUDA_TOOLKIT_VERSION}"
fi

echo "Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} locally via micromamba..."
if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" \
    "cuda-nvcc=${CUDA_TOOLKIT_VERSION}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}" \
    "gcc=12.*" "gxx=12.*" "sysroot_linux-64=2.17"; then
    echo "ERROR: Failed to install CUDA toolkit via micromamba"
    echo "This may indicate:"
    echo "  - Network connectivity issues"
    echo "  - Unavailable CUDA version ${CUDA_TOOLKIT_VERSION} for your platform"
    echo "  - Insufficient disk space"
    exit 1
fi

export CUDA_HOME="$VENV_PATH/cuda_env"
echo "Using local CUDA installation at: $CUDA_HOME"

# ============================================================================
# Create header symlinks and set compilation env vars
# ============================================================================
# Auto-detect CUDA target directory (e.g., x86_64-linux, aarch64-linux, sbsa-linux)
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
if [ -z "$CUDA_TARGET" ]; then
    echo "ERROR: No CUDA target directory found in $CUDA_HOME/targets/"
    exit 1
fi
echo "Detected CUDA target: $CUDA_TARGET"

# Symlink all CUDA headers from targets/ into include/
# PyTorch's cpp_extension only looks in CUDA_HOME/include for JIT compilation
CUDA_TARGETS_DIR="$CUDA_HOME/targets/${CUDA_TARGET}/include"
if [ -d "$CUDA_TARGETS_DIR" ]; then
    for item in "$CUDA_TARGETS_DIR"/*; do
        name=$(basename "$item")
        if [ ! -e "$CUDA_HOME/include/$name" ]; then
            ln -s "$item" "$CUDA_HOME/include/$name"
        fi
    done
    echo "Symlinked CUDA headers from $CUDA_TARGETS_DIR"
fi

# Fix broken libcudart.so symlink (micromamba may install different version)
if [ -L "$CUDA_HOME/lib/libcudart.so" ] && [ ! -e "$CUDA_HOME/lib/libcudart.so" ]; then
    rm -f "$CUDA_HOME/lib/libcudart.so"
    ACTUAL_CUDART=$(ls "$CUDA_HOME/lib"/libcudart.so.12* 2>/dev/null | head -1)
    if [ -n "$ACTUAL_CUDART" ]; then
        ln -s "$(basename "$ACTUAL_CUDART")" "$CUDA_HOME/lib/libcudart.so"
        echo "Fixed libcudart.so symlink -> $(basename "$ACTUAL_CUDART")"
    fi
fi

# Set compilation environment variables
export PATH="$VENV_PATH/bin:$CUDA_HOME/bin:$PATH"
CUDA_TARGETS_INCLUDE="$CUDA_HOME/targets/${CUDA_TARGET}/include"
export CPATH="${CPATH:+$CPATH:}$CUDA_HOME/include:$CUDA_TARGETS_INCLUDE"
export CXXFLAGS="${CXXFLAGS:-} -I$CUDA_HOME/include -I$CUDA_TARGETS_INCLUDE"
export LDFLAGS="${LDFLAGS:-} -L$CUDA_HOME/lib"
export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_HOME/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib"

echo "NVCC: $(which nvcc) ($(nvcc --version | tail -1))"
echo "CC: $(which gcc) ($(gcc --version | head -1))"

# ============================================================================
# Install Python packages
# ============================================================================
echo "Installing torch..."
# Pin torch to 2.7.1: flash-attn pre-built wheels are ABI-sensitive and only
# work with the exact torch version they were compiled against.
uv pip install torch==2.7.1 --torch-backend=auto --refresh

echo "Installing dependencies from requirements.txt..."
# Use --no-build-isolation-package for flash-attn to ensure it uses the installed torch
# and doesn't download a different version during build
uv pip install -r requirements.txt --torch-backend=auto --no-build-isolation-package flash-attn --refresh

# Validate flash-attn ABI compatibility — check the C++ extension actually loads
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "WARNING: flash-attn wheel has ABI mismatch (flash_attn_2_cuda import failed)"
    echo "Rebuilding from source... This can take 30+ minutes."
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==2.8.0.post2
fi

echo "If installation fails, follow upstream setup guide:"
echo "  - https://github.com/evo-design/evo"

echo "Evo1 setup complete!"
