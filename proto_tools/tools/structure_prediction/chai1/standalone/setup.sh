#!/bin/bash
# Setup script for Chai1 standalone environment
set -euo pipefail
source standalone_helpers.sh

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: chai1 setup: not supported on aarch64 (chai_lab==0.6.1 pins torch<2.7 lacks sm_121 + ships x86_64-only TorchScript ESM2)" >&2
    exit 1
fi

echo "Setting up Chai1 standalone environment..."

proto_install_pytorch

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

# torch 2.6 pins triton 3.2.0, which breaks on conda-forge Python 3.12 (PY_SSIZE_T_CLEAN, and
# "libcuda.so not found"), so that triton alone is replaced, after every other install so uv cannot
# pull it back. Any other torch keeps the triton it pins: newer tritons build sm_90 kernels with a
# CUDA 13 ptxas that CUDA 12 drivers refuse ("device kernel image is invalid").
if python -c "import sys, triton; sys.exit(not triton.__version__.startswith('3.2.'))"; then
    echo "Replacing triton 3.2..."
    uv pip install "triton==3.3.1"
fi

# Model weights, staged in the persistent weights dir (also used by germinal)
proto_ensure_chai1_weights

# Warn if GPU compute capability may be incompatible with chai_lab's pinned torch version.
if command -v nvidia-smi &> /dev/null; then
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]') || true
    if [ -n "$compute_cap" ]; then
        major=$(echo "$compute_cap" | cut -d. -f1)
        minor=$(echo "$compute_cap" | cut -d. -f2)
        # chai_lab==0.6.1 pins torch<2.7 which only supports compute capability up to 12.0
        if [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -gt 0 ]; }; then
            echo ""
            echo "WARNING: Your GPU has CUDA compute capability ${compute_cap}, but chai_lab==0.6.1"
            echo "pins torch<2.7 which only supports up to compute capability 12.0."
            echo "You may see 'no kernel image is available for execution on the device' errors."
            echo "This is a chai_lab version constraint, not a setup issue."
            echo ""
        fi
    fi
fi

echo "Chai1 setup complete!"
