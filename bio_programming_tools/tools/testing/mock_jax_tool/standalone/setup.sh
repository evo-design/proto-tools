#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Installing uv package manager..."
pip install uv

bpt_install_cuda_toolkit
bpt_install_jax

echo "Mock JAX tool setup complete"
