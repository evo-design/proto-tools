#!/bin/bash
set -euo pipefail

echo "Setting up SpliceTransformer standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Use hardware-aware PyTorch spec from centralized detection
# (injected by bio_programming_tools.utils.compute_deps)
# Override with SPLICE_TRANSFORMER_TORCH_SPEC or TORCH_SPEC env vars if needed
TORCH_SPEC="${SPLICE_TRANSFORMER_TORCH_SPEC:-${TORCH_SPEC:-${RECOMMENDED_TORCH_SPEC:-torch}}}"

echo "Detected platform: ${DETECTED_COMPUTE_PLATFORM:-unknown}"
echo "Installing PyTorch: ${TORCH_SPEC}"
uv pip install --extra-index-url "${RECOMMENDED_TORCH_INDEX}" "${TORCH_SPEC}"

echo "Installing remaining Python dependencies..."
uv pip install -r requirements.txt

echo "SpliceTransformer setup complete!"
