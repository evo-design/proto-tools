#!/bin/bash
# Setup script for AlphaGenome standalone environment
set -euo pipefail

echo "Setting up AlphaGenome standalone environment..."

echo "Installing uv package manager..."
pip install uv

DRIVER_MAJOR=""
CUDA_MAJOR=""
if command -v nvidia-smi &> /dev/null; then
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1 || true)
    CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p' | head -n1 || true)
fi

JAX_VARIANT="${ALPHAGENOME_JAX_VARIANT:-}"
if [ -z "$JAX_VARIANT" ] && [ -n "$CUDA_MAJOR" ]; then
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        JAX_VARIANT="cuda13"
    else
        JAX_VARIANT="cuda12"
    fi
fi
if [ -n "$JAX_VARIANT" ] && [ "$JAX_VARIANT" != "cuda12" ] && [ "$JAX_VARIANT" != "cuda13" ]; then
    echo "ERROR: ALPHAGENOME_JAX_VARIANT must be one of: cuda12, cuda13"
    exit 1
fi

JAX_SPEC="${ALPHAGENOME_JAX_SPEC:-}"
if [ -z "$JAX_SPEC" ]; then
    if [ -n "$JAX_VARIANT" ]; then
        if [ -n "$DRIVER_MAJOR" ]; then
            if [ "$DRIVER_MAJOR" -ge 570 ]; then
                JAX_SPEC="jax[${JAX_VARIANT}]>=0.9,<1"
            elif [ "$DRIVER_MAJOR" -ge 550 ]; then
                JAX_SPEC="jax[${JAX_VARIANT}]>=0.6,<0.9"
            elif [ "$DRIVER_MAJOR" -ge 535 ]; then
                JAX_SPEC="jax[${JAX_VARIANT}]>=0.5,<0.6"
            else
                JAX_SPEC="jax[${JAX_VARIANT}]>=0.4.25,<0.5"
            fi
        else
            JAX_SPEC="jax[${JAX_VARIANT}]>=0.5,<1"
        fi
    else
        JAX_SPEC="jax>=0.5,<1"
    fi
fi

USE_LOCAL_CUDA_ENV=$(echo "${ALPHAGENOME_USE_LOCAL_CUDA_ENV:-false}" | tr '[:upper:]' '[:lower:]')
if [ "$USE_LOCAL_CUDA_ENV" = "true" ]; then
    if [ -z "${VENV_PATH:-}" ]; then
        echo "ERROR: VENV_PATH is required when ALPHAGENOME_USE_LOCAL_CUDA_ENV=true"
        exit 1
    fi
    MAMBA_ROOT="$VENV_PATH/micromamba"
    mkdir -p "$MAMBA_ROOT"
    if [ ! -f "$MAMBA_ROOT/bin/micromamba" ]; then
        echo "Installing micromamba..."
        curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xvj -C "$MAMBA_ROOT" bin/micromamba
    fi
    export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
    eval "$($MAMBA_ROOT/bin/micromamba shell hook -s posix)"

    CUDA_TOOLKIT_CONSTRAINT="${ALPHAGENOME_CUDA_TOOLKIT_CONSTRAINT:-}"
    if [ -z "$CUDA_TOOLKIT_CONSTRAINT" ] && [ -n "${ALPHAGENOME_CUDA_TOOLKIT_VERSION:-}" ]; then
        # Backward compatibility with prior exact-version override.
        CUDA_TOOLKIT_CONSTRAINT="${ALPHAGENOME_CUDA_TOOLKIT_VERSION}"
    fi
    if [ -z "$CUDA_TOOLKIT_CONSTRAINT" ]; then
        if [ -n "$CUDA_MAJOR" ]; then
            CUDA_TOOLKIT_CONSTRAINT="${CUDA_MAJOR}.*"
        else
            CUDA_TOOLKIT_CONSTRAINT="12.*"
        fi
    fi
    echo "Installing local CUDA env (toolkit=${CUDA_TOOLKIT_CONSTRAINT}) via micromamba..."
    micromamba create -y -p "$VENV_PATH/cuda_env" -c nvidia -c conda-forge \
        "cuda-toolkit=${CUDA_TOOLKIT_CONSTRAINT}" \
        "cuda-cudart-dev=${CUDA_TOOLKIT_CONSTRAINT}" \
        "cudnn"

    export CUDA_HOME="$VENV_PATH/cuda_env"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "Detected driver=${DRIVER_MAJOR:-unknown} cuda=${CUDA_MAJOR:-unknown}"
echo "Installing JAX using spec: ${JAX_SPEC}"

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
