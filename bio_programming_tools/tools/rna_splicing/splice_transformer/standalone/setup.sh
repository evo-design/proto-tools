#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up SpliceTransformer standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Override with SPLICE_TRANSFORMER_TORCH_SPEC or TORCH_SPEC env vars if needed
TORCH_SPEC="${SPLICE_TRANSFORMER_TORCH_SPEC:-${TORCH_SPEC:-${RECOMMENDED_TORCH_SPEC:-torch}}}"
proto_install_pytorch "$TORCH_SPEC"

echo "Installing remaining Python dependencies..."
uv pip install -r requirements.txt

echo "SpliceTransformer setup complete!"
