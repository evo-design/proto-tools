#!/bin/bash
# Setup script for RoseTTAFold3 (RF3) standalone environment.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up RoseTTAFold3 standalone environment..."

# Platform gate: rc-foundry[rf3] pulls cuequivariance_ops_cu12, whose wheels are
# Linux x86_64 + CUDA 12 only. Refuse to set up on unsupported platforms with a
# clear message rather than a downstream wheel-install failure.
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "RF3 requires Linux x86_64 (cuequivariance_ops_cu12 wheel constraint)." >&2
    echo "Detected: $(uname -s) $(uname -m)" >&2
    exit 1
fi

echo "Installing uv package manager..."
pip install uv

proto_install_pytorch

# RF3 runs triton-compiled kernels that JIT a small CUDA launcher stub at first use
# for each new (sequence-length, batch-size) it sees. The JIT shells out to `cc` /
# `nvcc`, so we install a tool-local gcc + CUDA toolkit (the same pattern AF3 /
# Protenix / Germinal use) rather than depending on the host having them.
proto_install_cuda_toolkit "${RF3_CUDA_TOOLKIT_CONSTRAINT:-}" cuda-nvcc "gcc=12.*" "gxx=12.*"

echo "Installing rc-foundry[rf3] and dependencies..."
uv pip install -r requirements.txt

# Download the RF3 model checkpoint to a proto-tools managed cache (default:
# $PROTO_HOME/proto_model_cache/rf3). The runtime sets FOUNDRY_CHECKPOINT_DIRS
# so rf3 fold finds it without polluting $HOME/.foundry/checkpoints/.
proto_resolve_weights_dir rf3
if [ ! -f "$WEIGHTS_DIR/rf3_foundry_01_24_latest_remapped.ckpt" ]; then
    echo "Downloading RF3 checkpoint to $WEIGHTS_DIR ..."
    foundry install rf3 --checkpoint-dir "$WEIGHTS_DIR"
fi

echo "RoseTTAFold3 setup complete!"
