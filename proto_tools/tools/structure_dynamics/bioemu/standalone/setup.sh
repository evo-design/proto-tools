#!/bin/bash
# Setup script for BioEmu standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up BioEmu standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit "${BIOEMU_CUDA_TOOLKIT_CONSTRAINT:-}"
proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Upgrade JAX to GPU-enabled build with driver-compatible CUDA libs.
# bioemu pins jax==0.4.35 (CPU-only); we need CUDA support for Evoformer.
# Hardcoded instead of proto_install_jax because bioemu requires exactly
# jax==0.4.35, and cuda12 is the only GPU extra at that version.
JAX_SPEC="jax[cuda12]==0.4.35"
echo "Installing JAX with CUDA support: ${JAX_SPEC}"
uv pip install --force-reinstall "${JAX_SPEC}" "numpy==1.26.4"

echo "BioEmu setup complete!"
