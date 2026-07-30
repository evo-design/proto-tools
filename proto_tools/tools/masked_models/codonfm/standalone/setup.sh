#!/bin/bash
# Setup script for the CodonFM (Encodon) standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up CodonFM standalone environment..."

echo "Installing uv package manager..."
pip install uv

# CodonFM was published against torch 2.5.1; pin it so the vendored model matches upstream.
# torch 2.5.1 has no cu126 wheel, so when the auto-detected torch index is cu126 (or newer) uv's
# default first-index strategy can't find it. unsafe-best-match lets uv source torch 2.5.1 from
# PyPI (a CUDA build that runs on 12.6+ drivers) instead of failing.
proto_install_pytorch "torch==2.5.1" --index-strategy unsafe-best-match

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Validating the complete vendored CodonFM import closure..."
python -c "from src.inference.encodon import EncodonInference"

echo "CodonFM setup complete!"
