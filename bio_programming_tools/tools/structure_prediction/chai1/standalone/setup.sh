#!/bin/bash
# Setup script for Chai1 standalone environment
set -euo pipefail

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "ERROR: Chai is not supported on aarch64."
    echo "chai_lab==0.6.1 pins torch<2.7 which lacks sm_121 support, and its"
    echo "pre-compiled TorchScript ESM2 model is incompatible with newer GPU architectures."
    exit 1
fi

echo "Setting up Chai1 standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Installing remaining dependencies..."
uv pip install -r requirements.txt

echo "Upgrading triton..."
# torch 2.6.0 pins triton==3.2.0, which has a PY_SSIZE_T_CLEAN bug causing
# runtime failures with conda-forge Python 3.12. Upgrade AFTER all other installs
# to prevent uv from downgrading it back to 3.2.0 via torch's dependency.
uv pip install --upgrade triton

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
