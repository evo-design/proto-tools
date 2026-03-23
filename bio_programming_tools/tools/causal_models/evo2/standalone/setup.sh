#!/bin/bash
# Setup script for Evo2 standalone environment
set -euo pipefail

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: Evo2 is not supported on aarch64."
    echo "Evo2 requires transformer-engine and flash-attn which only provide x86_64 pre-built wheels."
    exit 1
fi

MAMBA_PLATFORM="linux-64"

echo "Setting up Evo2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true
uv cache clean flash-attn 2>/dev/null || true
uv cache clean transformer-engine 2>/dev/null || true

# ============================================================================
# Install CUDA toolkit + cuDNN via micromamba
# Micromamba is provided via $MAMBA_BIN environment variable
# ============================================================================
echo "Installing CUDA toolkit and cuDNN via micromamba..."
"$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    cuda-toolkit \
    cuda-nvcc \
    cuda-cudart-dev \
    cudnn \
    cuda-nvtx \
    "gcc=14.*" "gxx=14.*"

export CUDA_HOME="$VENV_PATH/cuda_env"
echo "Using local CUDA installation at: $CUDA_HOME"

# Auto-detect CUDA target directory (e.g., x86_64-linux, aarch64-linux, sbsa-linux)
CUDA_TARGET=$(ls "$CUDA_HOME/targets/" 2>/dev/null | head -1)
if [ -z "$CUDA_TARGET" ]; then
    echo "ERROR: No CUDA target directory found in $CUDA_HOME/targets/"
    exit 1
fi
echo "Detected CUDA target: $CUDA_TARGET"

# ============================================================================
# Create header symlinks for PyTorch's cpp_extension and transformer-engine
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
uv pip install torch==2.6.0 --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --refresh

echo "Installing build dependencies..."
uv pip install psutil ninja packaging setuptools wheel numpy

echo "Installing flash-attn..."
# flash-attn's build step imports torch, so disable build isolation.
uv pip install --no-build-isolation flash-attn==2.8.3 --refresh

# Validate flash-attn ABI compatibility — check CUDA kernels actually load.
# Test the C++ extension import that vortex uses, not just the Python wrapper.
if ! python -c "import flash_attn_2_cuda" 2>/dev/null; then
    echo "flash-attn wheel has ABI mismatch (flash_attn_2_cuda import failed), rebuilding from source..."
    echo "WARNING: Source builds can take 30+ minutes depending on hardware."
    uv pip install --no-build-isolation --no-binary flash-attn --reinstall-package flash-attn flash-attn==2.8.3
fi

echo "Installing transformer-engine..."
# transformer-engine's build step imports torch, so disable build isolation.
# TE >=2.5.0 includes pyproject.toml with __legacy__ build backend, fixing
# a build_tools import issue that broke source builds with 2.3.0.
# Clean uv's sdist cache for TE before building. TE's setup.py deletes its own
# build_tools/ directory after a successful build (cleanup step), which corrupts
# the cached sdist source. A subsequent build for a different Python version
# would reuse the dirty cache and fail with "No module named 'build_tools'".
uv cache clean transformer-engine-torch
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.5.0" --refresh

echo "Installing vortex..."
uv pip install vtx

echo "Installing evo2..."
uv pip install evo2 --constraint <(echo "torch==2.6.0")

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --constraint <(echo "torch==2.6.0")

echo "Upgrading triton..."
# torch 2.6.0 pins triton==3.2.0, which has a PY_SSIZE_T_CLEAN bug causing
# runtime failures with conda-forge Python 3.12. Upgrade AFTER all other installs
# to prevent uv from downgrading it back to 3.2.0 via torch's dependency.
uv pip install --upgrade triton

# ============================================================================
# Generate sitecustomize.py to preload CUDA libs at Python startup
# ============================================================================
# transformer-engine is compiled against cuda_env's CUDA headers, so it must
# use cuda_env's runtime libs (cublas, cudnn, etc.), not torch's bundled ones.
# EnvManager strips LD_LIBRARY_PATH, so we use ctypes.CDLL preloading instead.
SITE_PACKAGES=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/sitecustomize.py" <<'SITECUSTOMIZE'
# Auto-generated CUDA environment setup for Evo2 venv
import os
import glob
import site
import ctypes

sp = site.getsitepackages()[0]
venv_root = os.path.normpath(os.path.join(sp, "..", "..", ".."))
cuda_home = os.path.join(venv_root, "cuda_env")

os.environ["CUDA_HOME"] = cuda_home
os.environ["PATH"] = f"{venv_root}/bin:{cuda_home}/bin:" + os.environ.get("PATH", "")

# Pre-load CUDA libs from cuda_env so transformer-engine uses matching versions.
# Must happen before torch is imported, so the dynamic linker won't load torch's
# bundled (potentially version-mismatched) libs later.
# Note: micromamba stores libs as symlinks to package cache — do not filter them out.
_cuda_lib = os.path.join(cuda_home, "lib")
for pattern in ["libcudnn*.so.*", "libcublas.so.*", "libcublasLt.so.*",
                "libcusparse*.so.*", "libnvrtc*.so.*"]:
    for lib_path in sorted(glob.glob(os.path.join(_cuda_lib, pattern)), reverse=True):
        basename = os.path.basename(lib_path)
        so_idx = basename.find(".so.")
        if so_idx >= 0:
            ver_part = basename[so_idx + 4:]
            if ver_part.count(".") >= 1:  # fully versioned (e.g. .so.12.9.1.4)
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
SITECUSTOMIZE

echo ""
echo "If installation fails, follow upstream setup guides:"
echo "  - https://github.com/ArcInstitute/evo2"
echo "  - https://github.com/Zymrael/vortex"

echo "Evo2 setup complete!"
