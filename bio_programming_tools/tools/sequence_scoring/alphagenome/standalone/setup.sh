#!/bin/bash
# Setup script for AlphaGenome standalone environment
set -euo pipefail

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: AlphaGenome is not supported on aarch64."
    echo "AlphaGenome requires JAX with CUDA support,"
    echo "which does not provide aarch64 wheels."
    exit 1
fi

echo "Setting up AlphaGenome standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install CUDA toolkit and cuDNN locally via micromamba (required for JAX GPU support)
MAMBA_ROOT="$VENV_PATH/micromamba"
mkdir -p "$MAMBA_ROOT"
if [ ! -f "$MAMBA_ROOT/bin/micromamba" ]; then
    echo "Installing micromamba..."
    curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
eval "$($MAMBA_ROOT/bin/micromamba shell hook -s posix)"

CUDA_TOOLKIT_VERSION="12.1"
if command -v nvidia-smi &> /dev/null; then
    DETECTED_CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || true)
    if [ -n "$DETECTED_CUDA_MAJOR" ] && [ "$DETECTED_CUDA_MAJOR" -ge 13 ]; then
        CUDA_TOOLKIT_VERSION="12.8"
    fi
    echo "Detected CUDA ${DETECTED_CUDA_MAJOR:-unknown} — using toolkit ${CUDA_TOOLKIT_VERSION}"
fi

echo "Installing CUDA toolkit ${CUDA_TOOLKIT_VERSION} and cuDNN via micromamba..."
micromamba create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_VERSION}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_VERSION}" \
    "cudnn"

export CUDA_HOME="$VENV_PATH/cuda_env"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Auto-detect CUDA version and install the correct JAX variant with bundled CUDA libraries
# Modern JAX (>=0.4.20) bundles CUDA libraries by default, making venvs self-contained
echo "Detecting CUDA version for JAX installation..."
if command -v nvidia-smi &> /dev/null; then
    # Extract CUDA version from nvidia-smi output
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -n1)
    if [ -n "$CUDA_MAJOR" ]; then
        if [ "$CUDA_MAJOR" -ge 13 ]; then
            echo "Detected CUDA ${CUDA_MAJOR} — installing jax[cuda13] (with bundled CUDA libraries)..."
            uv pip install "jax[cuda13]"
        else
            echo "Detected CUDA ${CUDA_MAJOR} — installing jax[cuda12] (with bundled CUDA libraries)..."
            uv pip install "jax[cuda12]"
        fi
    else
        echo "WARNING: Could not determine CUDA version. Installing jax[cuda12] as default..."
        uv pip install "jax[cuda12]"
    fi
else
    echo "WARNING: nvidia-smi not found. Installing JAX without CUDA support..."
    uv pip install jax
fi

REPO_DIR="${VENV_PATH}/src/alphagenome_research"
if [ -d "$REPO_DIR" ]; then
  echo "Updating existing alphagenome_research clone..."
  git -C "$REPO_DIR" pull origin main
else
  echo "Cloning alphagenome_research..."
  git clone https://github.com/google-deepmind/alphagenome_research.git "$REPO_DIR"
fi
echo "Installing alphagenome_research from local clone..."
uv pip install "$REPO_DIR"

# Preload CUDA/cuDNN at Python startup so JAX can find them.
# os.environ alone doesn't affect the current process's dlopen on Linux,
# so we must load the shared libraries explicitly via ctypes.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PACKAGES/sitecustomize.py" <<SITECUSTOMIZE
import ctypes, os
cuda_lib = "$CUDA_HOME/lib"
os.environ["LD_LIBRARY_PATH"] = cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
for lib in ["libcudart.so.12", "libcudnn.so.9"]:
    try:
        ctypes.CDLL(os.path.join(cuda_lib, lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
SITECUSTOMIZE

echo "AlphaGenome setup complete!"
