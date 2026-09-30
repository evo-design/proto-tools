#!/bin/bash
# Shared env: Biohub ESM family (ESM3, ESM C).
# Both model families ship in the same `esm` package, so they share one env on disk.
set -euo pipefail
source standalone_helpers.sh

# All ESM3 and ESM C weights are MIT-licensed and ungated, so no HF token is needed.

echo "Setting up Biohub ESM env (covers ESM3 and ESM C)..."

# esm pins torch>=2.11,<2.12. Install it from the driver-matched index here; otherwise
# the requirements step swaps in PyPI's torch 2.11, a CUDA 13 build that needs driver 580+.
proto_install_pytorch "torch>=2.11,<2.12"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Biohub ESM env setup complete!"
