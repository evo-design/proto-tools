#!/bin/bash
# Setup script for BioEmu standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up BioEmu standalone environment..."

proto_install_cuda_toolkit "${BIOEMU_CUDA_TOOLKIT_CONSTRAINT:-}"
proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# Upgrade JAX to GPU-enabled build with driver-compatible CUDA libs.
# bioemu pins the CPU-only jax==0.4.35 package. Evoformer needs CUDA support.
# Hardcoded instead of proto_install_jax because bioemu requires exactly
# jax==0.4.35, and cuda12 is the only GPU extra at that version.
JAX_SPEC="jax[cuda12]==0.4.35"
echo "Installing JAX with CUDA support: ${JAX_SPEC}"
uv pip install --force-reinstall "${JAX_SPEC}" "numpy==1.26.4"

# AF2 params for the Evoformer embeddings, shared with the alphafold2 toolkit (~5.3 GB, downloaded
# once). bioemu's own downloader skips a folder holding download_finished.txt, so mark it as done.
proto_ensure_af2_params
touch "${AF2_PARAMS_DIR}/download_finished.txt"

echo "BioEmu setup complete!"
