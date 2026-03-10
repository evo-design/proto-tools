#!/bin/bash
# Setup script for AlphaGenome standalone environment
set -euo pipefail

echo "Setting up AlphaGenome standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Use hardware-aware JAX spec from centralized detection
# (injected by bio_programming_tools.utils.compute_deps)
# Override with ALPHAGENOME_JAX_VARIANT/JAX_SPEC env vars if needed
JAX_VARIANT="${ALPHAGENOME_JAX_VARIANT:-${RECOMMENDED_JAX_VARIANT:-cuda12}}"
JAX_SPEC="${ALPHAGENOME_JAX_SPEC:-${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}}"

# Validate JAX variant if user override is provided
if [ "${ALPHAGENOME_JAX_VARIANT:-}" != "" ] && [ "$JAX_VARIANT" != "cuda12" ] && [ "$JAX_VARIANT" != "cuda13" ]; then
    echo "ERROR: ALPHAGENOME_JAX_VARIANT must be one of: cuda12, cuda13"
    exit 1
fi

# ============================================================================
# Install CUDA toolkit via micromamba (JAX needs CUDA libs on LD_LIBRARY_PATH)
# ============================================================================
CUDA_TOOLKIT_CONSTRAINT="${ALPHAGENOME_CUDA_TOOLKIT_CONSTRAINT:-}"
if [ -z "$CUDA_TOOLKIT_CONSTRAINT" ] && [ -n "${ALPHAGENOME_CUDA_TOOLKIT_VERSION:-}" ]; then
    # Backward compatibility with prior exact-version override.
    CUDA_TOOLKIT_CONSTRAINT="${ALPHAGENOME_CUDA_TOOLKIT_VERSION}"
fi
if [ -z "$CUDA_TOOLKIT_CONSTRAINT" ]; then
    if [ -n "${DETECTED_CUDA_VERSION:-}" ]; then
        CUDA_TOOLKIT_CONSTRAINT="${DETECTED_CUDA_VERSION}.*"
    else
        CUDA_TOOLKIT_CONSTRAINT="12.*"
    fi
fi
echo "Installing CUDA toolkit ${CUDA_TOOLKIT_CONSTRAINT} locally via micromamba..."
if ! "$MAMBA_BIN" create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
    "cuda-toolkit=${CUDA_TOOLKIT_CONSTRAINT}" \
    "cuda-cudart-dev=${CUDA_TOOLKIT_CONSTRAINT}" \
    "cudnn"; then
    echo "ERROR: Failed to install CUDA toolkit via micromamba"
    exit 1
fi

echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
echo "Detected driver: ${DETECTED_DRIVER_VERSION:-unknown}, CUDA: ${DETECTED_CUDA_VERSION:-unknown}"
echo "Installing JAX: ${JAX_SPEC}"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

uv pip install "${JAX_SPEC}"

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

# alphagenome_research leaves JAX unconstrained and may upgrade jax/jaxlib to
# versions incompatible with the selected CUDA plugin. Re-apply the selected
# JAX range to keep GPU wheels aligned with host driver/CUDA.
echo "Re-applying JAX compatibility spec after alphagenome_research install..."
uv pip install --upgrade "${JAX_SPEC}"

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
rm -f "$SITE_PACKAGES/sitecustomize.py"

echo "AlphaGenome setup complete!"
