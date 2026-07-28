#!/bin/bash
# Shared env: Biohub ESM family (ESM3, ESM C).
# Both model families ship in the same `esm` package, so they share one env on disk.
set -euo pipefail
source standalone_helpers.sh

# All ESM3 and ESM C weights are MIT-licensed and ungated, so no HF token is needed.

echo "Setting up Biohub ESM env (covers ESM3 and ESM C)..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Biohub ESM env setup complete!"
