#!/bin/bash
set -euo pipefail

echo "Setting up Protenix standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Determine CUDA toolkit version to install for JIT compilation.
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

# Install CUDA toolkit locally in the venv via micromamba
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

# Set CUDA_HOME to the local micromamba installation
export CUDA_HOME="$VENV_PATH/cuda_env"
echo "Using local CUDA installation at: $CUDA_HOME"

# Auto-detect CUDA target directory (e.g., x86_64-linux, aarch64-linux, sbsa-linux)
# The conda CUDA toolkit uses different target names depending on version and arch.
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
echo "CC: $(which gcc) ($(gcc --version | head -1))"

# Install protenix with CUDA-aware PyTorch wheels. unsafe-best-match is required
# because uv's default index strategy treats the PyTorch index as authoritative for
# common packages like tqdm that exist on both indices, causing resolution failures.
# The version floor (>=1.0.0) prevents silent fallback to the empty 0.0.1 placeholder.
echo "Installing protenix (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})..."
uv pip install -r requirements.txt \
    --extra-index-url "${RECOMMENDED_TORCH_INDEX}" \
    --index-strategy unsafe-best-match

echo "Upgrading triton..."
# conda-forge Python ships triton 3.2.0 which causes "libcuda.so not found" runtime
# failures. Upgrade AFTER all other installs to prevent uv from downgrading it back.
uv pip install --upgrade triton

# Locate site-packages for patching and sitecustomize.py
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Patch protenix's torch_ext_compile.py to use the current GPU's architecture
# instead of hardcoded sm_70/sm_80. Protenix upstream overrides TORCH_CUDA_ARCH_LIST
# and hardcodes -gencode flags, so our env var alone is insufficient.
# TORCH_CUDA_ARCH_LIST is injected by compute_deps.py (e.g., "12.0").
COMPILE_PY="$SITE_PACKAGES/protenix/model/layer_norm/torch_ext_compile.py"
GPU_ARCH="${TORCH_CUDA_ARCH_LIST:-}"
if [ -f "$COMPILE_PY" ] && [ -n "$GPU_ARCH" ]; then
    GPU_SM="sm_${GPU_ARCH//./}"  # e.g., "12.0" -> "sm_120"
    GPU_COMPUTE="compute_${GPU_ARCH//./}"
    echo "Patching $COMPILE_PY for $GPU_SM (arch $GPU_ARCH)"
    # Replace the hardcoded TORCH_CUDA_ARCH_LIST
    sed -i "s|os.environ\[\"TORCH_CUDA_ARCH_LIST\"\] = .*|os.environ[\"TORCH_CUDA_ARCH_LIST\"] = \"$GPU_ARCH\"|" "$COMPILE_PY"
    # Replace hardcoded -gencode flags with the detected arch
    python -c "
import re

with open('$COMPILE_PY', 'r') as f:
    content = f.read()

# Replace all -gencode flag pairs with a single one for the detected arch
content = re.sub(
    r'(\"-gencode\",\s*\n\s*\"arch=compute_\d+,code=sm_\d+\",\s*\n\s*)+',
    '\"-gencode\",\n            \"arch=$GPU_COMPUTE,code=$GPU_SM\",\n            ',
    content,
)

with open('$COMPILE_PY', 'w') as f:
    f.write(content)
"
    echo "Patched torch_ext_compile.py for $GPU_SM"
fi

# Create a sitecustomize.py to set CUDA environment variables automatically when Python starts
# This ensures the environment is configured before any imports happen
cat > "$SITE_PACKAGES/sitecustomize.py" <<SITECUSTOMIZE
# Auto-generated CUDA environment setup for Protenix venv
import os

cuda_home = "$CUDA_HOME"
venv_bin = "$VENV_PATH/bin"
cuda_targets_include = cuda_home + "/targets/${CUDA_TARGET}/include"

# Set CUDA environment variables
os.environ["CUDA_HOME"] = cuda_home
# Use conda-forge GCC 12 from cuda_env for runtime JIT compilation (nvcc compat)
os.environ["CC"] = cuda_home + "/bin/gcc"
os.environ["CXX"] = cuda_home + "/bin/g++"
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
