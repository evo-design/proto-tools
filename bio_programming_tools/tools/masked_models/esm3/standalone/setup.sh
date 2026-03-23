#!/bin/bash
# Setup script for ESM3 standalone environment
set -euo pipefail
source standalone_helpers.sh

bpt_check_gated_hf_repo "EvolutionaryScale/esm3-sm-open-v1" "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"

echo "Setting up ESM3 standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "ESM3 setup complete!"
