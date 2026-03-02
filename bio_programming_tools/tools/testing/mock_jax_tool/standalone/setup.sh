#!/bin/bash
set -euo pipefail

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware JAX version (from centralized detection)
JAX_SPEC="${RECOMMENDED_JAX_SPEC:-jax[cuda12]>=0.5,<1}"
echo "Installing JAX: ${JAX_SPEC} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${JAX_SPEC}" --refresh

echo "Mock JAX tool setup complete"
