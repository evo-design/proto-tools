#!/bin/bash
# Setup script for Cherimoya standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Cherimoya standalone environment..."

# cherimoya registers Triton autotuned kernels when imported. Older Triton releases (those paired
# with torch < 2.9) raise "0 active drivers" at that point on a host without a visible GPU.
proto_install_pytorch "torch>=2.9,<3"

# --no-deps skips cherimoya's training and analysis dependencies (macs3, modisco, lightning, ...);
# inference needs only torch, numpy and tangermeme.
uv pip install --no-deps "cherimoya==0.3.0"
uv pip install -r requirements.txt

echo "Cherimoya setup complete!"
