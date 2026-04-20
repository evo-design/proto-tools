#!/bin/bash
# Shared env: EvolutionaryScale ESM family (ESM3, ESM C).
# Both model families ship in the same `esm` package, so they share one env on disk.
set -euo pipefail
source standalone_helpers.sh

# ESM3 is gated on HuggingFace; the check ensures the user has accepted the license.
# ESM C 300M is open; ESM C 600M is non-commercial-only and not gated on HF.
proto_check_gated_hf_repo "EvolutionaryScale/esm3-sm-open-v1" "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"

echo "Setting up EvolutionaryScale ESM env (covers ESM3 and ESM C)..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "EvolutionaryScale ESM env setup complete!"
