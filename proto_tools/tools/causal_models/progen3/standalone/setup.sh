#!/bin/bash
# Setup script for ProGen3 standalone environment
set -euo pipefail
source standalone_helpers.sh

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: ProGen3 is not supported on aarch64."
    echo "ProGen3 requires flash-attn which has no aarch64 wheels."
    exit 1
fi

echo "Setting up ProGen3 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true
uv cache clean megablocks 2>/dev/null || true

# ============================================================================
# Install CUDA toolkit + compatible GCC via micromamba
# ============================================================================
# Pin CUDA 12.4 to match cu124 torch wheels. cuda-version pin (via conda-forge)
# forces sub-packages to align, preventing nvJitLink symbol errors from mismatched
# system CUDA on LD_LIBRARY_PATH.
proto_install_cuda_toolkit "12.4.*" cuda-nvcc cuda-nvtx "gcc=13.*" "gxx=13.*"

export CUDA_HOME="$VENV_PATH/cuda_env"
echo "Using local CUDA installation at: $CUDA_HOME"

# Auto-detect CUDA target directory (e.g., x86_64-linux, aarch64-linux)
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
if [ -z "$CUDA_TARGET" ]; then
    echo "ERROR: No CUDA target directory found in $CUDA_HOME/targets/"
    exit 1
fi
echo "Detected CUDA target: $CUDA_TARGET"

# ============================================================================
# Create header symlinks for PyTorch's cpp_extension builds
# ============================================================================
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

# nvtx3 headers may be installed under nsight-compute; symlink to standard include path
if [ ! -e "$CUDA_HOME/include/nvtx3" ]; then
    NVTX_SRC=$(find "$CUDA_HOME" -path "*/nvtx/include/nvtx3" -type d 2>/dev/null | head -1)
    if [ -n "$NVTX_SRC" ]; then
        ln -s "$NVTX_SRC" "$CUDA_HOME/include/nvtx3"
        echo "Symlinked nvtx3 headers from $NVTX_SRC"
    fi
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

# Set compilation environment variables.
# PREPEND cuda_env/lib to LD_LIBRARY_PATH so local CUDA is found before system CUDA.
export PATH="$VENV_PATH/bin:$CUDA_HOME/bin:$PATH"
CUDA_TARGETS_INCLUDE="$CUDA_HOME/targets/${CUDA_TARGET}/include"
export CPATH="${CPATH:+$CPATH:}$CUDA_HOME/include:$CUDA_TARGETS_INCLUDE"
export CXXFLAGS="${CXXFLAGS:-} -I$CUDA_HOME/include -I$CUDA_TARGETS_INCLUDE"
export LDFLAGS="${LDFLAGS:-} -L$CUDA_HOME/lib"
export LIBRARY_PATH="${LIBRARY_PATH:+$LIBRARY_PATH:}$CUDA_HOME/lib"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "NVCC: $(which nvcc) ($(nvcc --version | tail -1))"
echo "CC: $(which gcc) ($(gcc --version | head -1))"

# ============================================================================
# Install Python packages
# ============================================================================
echo "Installing torch..."
# Pin torch to upstream progen3's requirement (torch>=2.5.0,<2.5.2).
# flash-attn and megablocks wheels must match this exact torch ABI.
# Upstream uses cu124 wheels; torch 2.5.x is not on the cu126 index.
uv pip install "torch>=2.5.0,<2.5.2" --extra-index-url "https://download.pytorch.org/whl/cu124" --refresh

echo "Installing build dependencies..."
uv pip install psutil ninja packaging setuptools wheel numpy

echo "Installing flash-attn..."
# Matches upstream progen3 setup.sh: flash-attn==2.7.4.post1
MAX_JOBS=4 uv pip install --no-build-isolation flash-attn==2.7.4.post1 --refresh

# Validate flash-attn ABI compatibility
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "flash-attn wheel has ABI mismatch, rebuilding from source..."
    MAX_JOBS=4 uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==2.7.4.post1
fi

echo "Installing megablocks..."
# Matches upstream progen3 setup.sh: megablocks[gg]==0.7.0
uv pip install --no-build-isolation "megablocks[gg]==0.7.0" --refresh

echo "Installing progen3..."
uv pip install "progen3 @ git+https://github.com/Profluent-AI/progen3.git"

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Validate progen3 import
if ! python -c "from progen3.modeling import ProGen3ForCausalLM; print('ProGen3 import OK')"; then
    echo "ERROR: ProGen3 import failed after installation"
    exit 1
fi

# ============================================================================
# Generate sitecustomize.py to preload CUDA libs at Python startup
# ============================================================================
SITE_PACKAGES=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/sitecustomize.py" <<'SITECUSTOMIZE'
# Auto-generated CUDA environment setup for ProGen3 venv
import os
import glob
import site
import ctypes

sp = site.getsitepackages()[0]
venv_root = os.path.normpath(os.path.join(sp, "..", "..", ".."))
cuda_home = os.path.join(venv_root, "cuda_env")

os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = f"{venv_root}/bin:{cuda_home}/bin:" + os.environ.get("PATH", "")

# Pre-load CUDA libs from cuda_env so flash-attn/megablocks use matching versions.
_cuda_lib = os.path.join(cuda_home, "lib")
for pattern in ["libcudnn*.so.*", "libcublas.so.*", "libcublasLt.so.*",
                "libcusparse*.so.*", "libnvrtc*.so.*"]:
    for lib_path in sorted(glob.glob(os.path.join(_cuda_lib, pattern)), reverse=True):
        basename = os.path.basename(lib_path)
        so_idx = basename.find(".so.")
        if so_idx >= 0:
            ver_part = basename[so_idx + 4:]
            if ver_part.count(".") >= 1:
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
SITECUSTOMIZE

echo "ProGen3 setup complete!"
