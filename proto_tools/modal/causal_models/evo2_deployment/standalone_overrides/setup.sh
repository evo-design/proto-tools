#!/bin/bash
# Override of: proto_tools/tools/causal_models/evo2/standalone/setup.sh
# Intentional deltas: force cu124 torch index, install flash-attn 2.7.4.post1 from Dao-AILab's pre-built wheel, pin CUDA/cuDNN to CUDA 12, and preload the upgraded cuBLAS wheel path on Modal.
# Last reviewed: 2026-06-17.
set -euo pipefail

# Modal defaults to cu128, which doesn't publish torch==2.6.0; force cu124.
export RECOMMENDED_TORCH_INDEX="https://download.pytorch.org/whl/cu124"

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
    "cuda-toolkit=12.*" \
    "cuda-nvcc=12.*" \
    "cuda-cudart-dev=12.*" \
    "cuda-nvtx=12.*" \
    "cudnn=9.*=cuda12*" \
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
# torch 2.6.0+cu124 pins nvidia-cudnn-cu12==9.1.0.70, which the cu124 index no longer carries;
# without this uv's first-index default refuses to source it from PyPI, where it remains.
uv pip install torch==2.6.0 --extra-index-url "${RECOMMENDED_TORCH_INDEX}" --index-strategy unsafe-best-match --refresh

echo "Installing build dependencies..."
uv pip install psutil ninja packaging setuptools wheel numpy

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
uv pip install vtx==1.1.0

echo "Installing evo2..."
uv pip install evo2==0.5.5 --constraint <(echo "torch==2.6.0")

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --constraint <(echo "torch==2.6.0")

echo "Upgrading triton..."
# torch 2.6.0 pins triton==3.2.0, which has a PY_SSIZE_T_CLEAN bug causing
# runtime failures with conda-forge Python 3.12. Upgrade AFTER all other installs
# to prevent uv from downgrading it back to 3.2.0 via torch's dependency.
uv pip install --upgrade triton

echo "Installing flash-attn 2.7.4.post1 from Dao-AILab pre-built wheel (cxx11abiFALSE + torch2.6)..."
# Both cxx11abiFALSE and cxx11abiTRUE variants of flash-attn 2.8.3 fail at
# runtime against torch 2.6.0+cu124 with
# "undefined symbol c10::Error::Error(SourceLocation, std::__cxx11::basic_string)".
# The 2.7.4.post1 wheel is the exact version the pre-refactor service used
# successfully against this same torch. Pinning until Dao-AILab resolves
# whatever ABI regression the 2.8.x wheels introduced. Install LAST (and
# with --force-reinstall --no-deps) so nothing earlier in the install
# chain can override us.
uv pip install --no-build-isolation --force-reinstall --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# ============================================================================
# cuBLAS 12.8+ upgrade for transformer-engine compatibility
# ============================================================================
# TE 2.5.0 calls into cuBLAS APIs that require cuBLAS 12.8+, but torch 2.6.0
# bundles cuBLAS 12.4 (via its nvidia-cublas-cu12 dep). Without this upgrade,
# TE's cublaslt_gemm raises "cuBLAS Error: an unsupported value or parameter".
# pip install --force-reinstall corrupts the nvidia namespace package, so we
# extract the upgraded wheel's .so files directly into the installed
# nvidia.cublas package dir, preserving the namespace structure. Ported from
# the pre-refactor service's working recipe.
echo "Upgrading cuBLAS to 12.8+ for transformer-engine compatibility..."
"$PIP_EXE" download --no-deps 'nvidia-cublas-cu12>=12.8' -d /tmp/cublas_wheel
"$PYTHON_EXE" <<'PYEOF'
import glob
import os
import shutil
import zipfile

import nvidia.cublas

dst = os.path.join(nvidia.cublas.__path__[0], "lib")
whl = glob.glob("/tmp/cublas_wheel/*.whl")[0]
with zipfile.ZipFile(whl) as z:
    members = [m for m in z.namelist() if m.startswith("nvidia/cublas/lib/") and ".so" in m]
    for m in members:
        z.extract(m, "/tmp/cublas_extract")
src = "/tmp/cublas_extract/nvidia/cublas/lib"
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print(f"Copied {len(os.listdir(src))} cuBLAS 12.8+ files to {dst}")
PYEOF
rm -rf /tmp/cublas_wheel /tmp/cublas_extract
# Remove cuda_env's libcublas so sitecustomize.py's RTLD_GLOBAL preload
# doesn't shadow the upgraded version at runtime.
rm -f "$CUDA_HOME"/lib/libcublas*.so* "$CUDA_HOME"/lib/libcublasLt*.so* 2>/dev/null || true

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

# Pre-load CUDA libs (RTLD_GLOBAL) before torch imports. Includes
# site-packages/nvidia/cublas/lib so the cuBLAS 12.8 upgrade wheel below is
# reachable when LD_LIBRARY_PATH is stripped at runtime.
_lib_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(sp, "nvidia", "cublas", "lib"),
]
_loaded = set()
for pattern in ["libcudnn*.so.*", "libcublas.so.*", "libcublasLt.so.*",
                "libcusparse*.so.*", "libnvrtc*.so.*"]:
    matches = []
    for d in _lib_dirs:
        matches.extend(glob.glob(os.path.join(d, pattern)))
    matches.sort(key=lambda p: (-len(os.path.basename(p)), p))
    for lib_path in matches:
        real = os.path.realpath(lib_path)
        if real in _loaded:
            continue
        _loaded.add(real)
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
