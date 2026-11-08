#!/bin/bash
# Setup script for Boltz2 standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Boltz2 standalone environment..."

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# torch 2.6 pins triton 3.2.0, which breaks on conda-forge Python 3.12 (PY_SSIZE_T_CLEAN, and
# "libcuda.so not found"), so that triton alone is replaced, after every other install so uv cannot
# pull it back. Any other torch keeps the triton it pins: newer tritons build sm_90 kernels with a
# CUDA 13 ptxas that CUDA 12 drivers refuse ("device kernel image is invalid").
if python -c "import sys, triton; sys.exit(not triton.__version__.startswith('3.2.'))"; then
    echo "Replacing triton 3.2..."
    uv pip install "triton==3.3.1"
fi

echo "Boltz2 setup complete!"
