#!/bin/bash
set -euo pipefail

echo "Installing uv package manager..."
pip install uv

# Clear caches BEFORE installing any ABI-sensitive packages
echo "Clearing package caches for ABI-sensitive dependencies..."
uv cache clean torch 2>/dev/null || true

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --refresh

echo "Mock PyTorch tool setup complete"
