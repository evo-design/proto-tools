#!/bin/bash
set -euo pipefail

echo "Setting up Protenix standalone environment..."

# Detect architecture for platform-specific paths
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    MAMBA_PLATFORM="linux-aarch64"
    CUDA_TARGET="aarch64-linux"
elif [ "$ARCH" = "x86_64" ]; then
    MAMBA_PLATFORM="linux-64"
    CUDA_TARGET="x86_64-linux"
else
    echo "ERROR: Unsupported architecture: $ARCH"
    exit 1
fi

echo "Installing uv package manager..."
pip install uv

# Install micromamba for managing CUDA toolkit
echo "Installing micromamba for ${ARCH}..."
MAMBA_ROOT="$VENV_PATH/micromamba"
mkdir -p "$MAMBA_ROOT"

# Download and install micromamba
if [ ! -f "$MAMBA_ROOT/bin/micromamba" ]; then
    echo "Downloading micromamba..."
    if ! curl -Ls "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" | tar -xvj -C "$MAMBA_ROOT" bin/micromamba; then
        echo "ERROR: Failed to download or extract micromamba from https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest"
        echo "Please check your internet connection and try again."
        exit 1
    fi
fi

# Verify micromamba binary exists and is executable
if [ ! -f "$MAMBA_ROOT/bin/micromamba" ]; then
    echo "ERROR: micromamba binary not found at $MAMBA_ROOT/bin/micromamba after installation attempt"
    exit 1
fi

if [ ! -x "$MAMBA_ROOT/bin/micromamba" ]; then
    echo "WARNING: micromamba binary is not executable, attempting to fix permissions..."
    chmod +x "$MAMBA_ROOT/bin/micromamba" || {
        echo "ERROR: Failed to make micromamba executable"
        exit 1
    }
fi

# Initialize micromamba for this session
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
if ! eval "$($MAMBA_ROOT/bin/micromamba shell hook -s posix)"; then
    echo "ERROR: Failed to initialize micromamba shell hook"
    exit 1
fi

# Verify micromamba command is now available
if ! command -v micromamba &> /dev/null; then
    echo "ERROR: micromamba command not found in PATH after initialization"
    echo "This may indicate an issue with the shell hook evaluation"
    exit 1
fi

echo "✓ micromamba successfully installed and initialized"

# Auto-detect CUDA version for toolkit installation
CUDA_TOOLKIT_VERSION="12.1"
if command -v nvidia-smi &> /dev/null; then
    DETECTED_CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || true)
    if [ -n "$DETECTED_CUDA_MAJOR" ] && [ "$DETECTED_CUDA_MAJOR" -ge 13 ]; then
        CUDA_TOOLKIT_VERSION="12.8"
        echo "Detected CUDA ${DETECTED_CUDA_MAJOR} — using CUDA toolkit ${CUDA_TOOLKIT_VERSION} for compatibility"
    else
        echo "Detected CUDA ${DETECTED_CUDA_MAJOR:-unknown} — using CUDA toolkit ${CUDA_TOOLKIT_VERSION}"
    fi
fi

# Install CUDA toolkit locally in the venv via micromamba
echo "Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} locally via micromamba..."
if ! micromamba create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" \
    "cuda-nvcc=${CUDA_TOOLKIT_VERSION}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}"; then
    echo "ERROR: Failed to install CUDA toolkit via micromamba"
    echo "This may indicate:"
    echo "  - Network connectivity issues"
    echo "  - Unavailable CUDA version ${CUDA_TOOLKIT_VERSION} for your platform"
    echo "  - Insufficient disk space"
    exit 1
fi

# Set CUDA_HOME to the local micromamba installation
export CUDA_HOME="$VENV_PATH/cuda_env"
echo "Using local CUDA installation at: $CUDA_HOME"

# Create symlinks for cuda/std headers (from targets/ to include/)
# This is needed because PyTorch's cpp_extension only looks in CUDA_HOME/include
CUDA_TARGETS_DIR="$CUDA_HOME/targets/${CUDA_TARGET}/include"
if [ -d "$CUDA_TARGETS_DIR" ]; then
    for header_dir in cuda thrust cub; do
        if [ -d "$CUDA_TARGETS_DIR/$header_dir" ] && [ ! -e "$CUDA_HOME/include/$header_dir" ]; then
            ln -s "$CUDA_TARGETS_DIR/$header_dir" "$CUDA_HOME/include/$header_dir"
        fi
    done
fi

# Fix broken libcudart.so symlink (micromamba may install different version than requested)
# Find the actual libcudart.so.* file and link to it
if [ -L "$CUDA_HOME/lib/libcudart.so" ] && [ ! -e "$CUDA_HOME/lib/libcudart.so" ]; then
    # Symlink is broken, recreate it
    rm -f "$CUDA_HOME/lib/libcudart.so"
    ACTUAL_CUDART=$(ls "$CUDA_HOME/lib"/libcudart.so.12* 2>/dev/null | head -1)
    if [ -n "$ACTUAL_CUDART" ]; then
        ln -s "$(basename "$ACTUAL_CUDART")" "$CUDA_HOME/lib/libcudart.so"
        echo "Fixed libcudart.so symlink to point to $(basename "$ACTUAL_CUDART")"
    fi
fi

# Set up CUDA environment variables for PyTorch JIT compilation
CUDA_INCLUDE_PATH="$CUDA_HOME/include"
CUDA_TARGETS_INCLUDE="$CUDA_HOME/targets/${CUDA_TARGET}/include"
CUDA_LIB_PATH="$CUDA_HOME/lib"

# Add venv bin directory to PATH first so ninja and other tools are available
export PATH="$VENV_PATH/bin:$CUDA_HOME/bin:$PATH"
export CPATH="${CPATH:+$CPATH:}$CUDA_INCLUDE_PATH:$CUDA_TARGETS_INCLUDE"
export CXXFLAGS="${CXXFLAGS:-} -I$CUDA_INCLUDE_PATH -I$CUDA_TARGETS_INCLUDE"
export LDFLAGS="${LDFLAGS:-} -L$CUDA_LIB_PATH"
export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_LIB_PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_LIB_PATH"

echo "Added CUDA headers from: $CUDA_INCLUDE_PATH"
echo "Added CUDA libraries from: $CUDA_LIB_PATH"
echo "NVCC location: $(which nvcc)"

echo "Installing protenix with --no-build-isolation flag..."
uv pip install --no-build-isolation protenix --torch-backend=auto

# Create a sitecustomize.py to set CUDA environment variables automatically when Python starts
# This ensures the environment is configured before any imports happen
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/sitecustomize.py" <<SITECUSTOMIZE
# Auto-generated CUDA environment setup for Protenix venv
import os

cuda_home = "$CUDA_HOME"
venv_bin = "$VENV_PATH/bin"
cuda_targets_include = cuda_home + "/targets/${CUDA_TARGET}/include"

# Set CUDA environment variables
os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = f"{venv_bin}:{cuda_home}/bin:" + os.environ.get("PATH", "")
os.environ["CPATH"] = cuda_home + "/include:" + cuda_targets_include + ":" + os.environ.get("CPATH", "")
os.environ["CXXFLAGS"] = os.environ.get("CXXFLAGS", "") + f" -I{cuda_home}/include -I{cuda_targets_include}"
os.environ["LDFLAGS"] = os.environ.get("LDFLAGS", "") + f" -L{cuda_home}/lib"
os.environ["LIBRARY_PATH"] = cuda_home + "/lib:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = cuda_home + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
SITECUSTOMIZE

echo "Testing CUDA extension compilation..."
python -c "from protenix.model.layer_norm.layer_norm import FusedLayerNorm; print('✓ Protenix CUDA extensions loaded successfully')" || {
    echo "ERROR: Failed to load Protenix CUDA extensions"
    exit 1
}

echo "Protenix setup complete!"
