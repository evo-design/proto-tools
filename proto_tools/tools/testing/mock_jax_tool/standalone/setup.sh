#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit
proto_install_jax

echo "Mock JAX tool setup complete"
