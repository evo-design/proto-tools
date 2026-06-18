#!/bin/bash
# Shared env: Biohub ESM family (ESM3, ESM C).
# Both model families ship in the same `esm` package, so they share one env on disk.
set -euo pipefail
source standalone_helpers.sh

# Verify the public biohub fork repo is reachable (it is not gated on HF, so this is a
# reachability check only; ESM3 license/token enforcement happens at runtime via require_hf_token()).
# ESM C 300M is open; ESM C 600M is non-commercial-only and not gated on HF.
proto_check_gated_hf_repo "biohub/esm3-sm-open-v1" "https://huggingface.co/biohub/esm3-sm-open-v1"

echo "Setting up Biohub ESM env (covers ESM3 and ESM C)..."

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Biohub ESM env setup complete!"
