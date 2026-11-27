#!/bin/bash
# Setup script for AlphaFold2 (ColabDesign) standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AlphaFold2 (ColabDesign) standalone environment..."

proto_install_cuda_toolkit "${ALPHAFOLD2_CUDA_TOOLKIT_CONSTRAINT:-}"

echo "Installing JAX, ColabDesign, and deps..."
uv pip install -r requirements.txt

# AF2 parameters (~5.3 GB), shared with bindcraft, freebindcraft, and germinal
proto_ensure_af2_params

echo "AlphaFold2 (ColabDesign) setup complete!"
