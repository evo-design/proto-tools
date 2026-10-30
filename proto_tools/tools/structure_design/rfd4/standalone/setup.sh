#!/bin/bash
# Setup script for the RFD4-Proteina standalone environment.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up RFD4-Proteina standalone environment..."

# rfd4-proteina and the CUDA wheels its all-cuda12 extra pulls are Linux x86_64 only.
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "RFD4-Proteina requires Linux x86_64 (tmol / cuequivariance CUDA wheel constraint)." >&2
    echo "Detected: $(uname -s) $(uname -m)" >&2
    exit 1
fi

# No CPU path upstream; a CPU env would build fine, be marked SUCCESS, and fail only at inference.
if [ "${DETECTED_COMPUTE_PLATFORM:-cpu}" != "cuda" ]; then
    echo "RFD4-Proteina requires a CUDA GPU; upstream builds its Lightning trainer with" >&2
    echo "accelerator=\"gpu\" unconditionally, so there is no CPU path." >&2
    echo "Detected compute platform: ${DETECTED_COMPUTE_PLATFORM:-cpu}" >&2
    exit 1
fi
# torch>=2.11 wheels exist only on the cu126/cu128 indices, which need driver >= 525.
if [ "${DETECTED_DRIVER_VERSION:-0}" -lt 525 ]; then
    echo "RFD4-Proteina needs torch>=2.11, which requires NVIDIA driver >= 525 (CUDA 12+)." >&2
    echo "Detected driver major version: ${DETECTED_DRIVER_VERSION:-unknown}" >&2
    exit 1
fi

# torch>=2.11 is above the centralized floor, so pin it here on the hardware-matched index.
proto_install_pytorch "torch>=2.11,<3"

# Toolkit for the inductor/triton JIT, plus the xorg libs Open Babel's plugins need at import.
proto_install_cuda_toolkit "${RFD4_CUDA_TOOLKIT_CONSTRAINT:-}" cuda-nvcc "gcc=12.*" "gxx=12.*" \
    xorg-libxrender xorg-libxext xorg-libx11

# all-cuda12 stays in requirements.txt so uv keeps the torch wheel installed above.
echo "Installing rfd4-proteina[all-cuda12] and dependencies..."
uv pip install -r requirements.txt

# DATA_PATH backs config nodes hf_ckpt_path overrides but that must still resolve.
mkdir -p "$VENV_PATH/rfd4_data"

# Weights are not fetched here: the repo is ungated and downloads on first run into HF_HOME.

if [ ! -x "$VENV_PATH/bin/rfd4-generate" ]; then
    echo "ERROR: rfd4-generate console script missing from $VENV_PATH/bin after install." >&2
    exit 1
fi
# Assert rather than print: a wrong torch here surfaces only at inference, against a SUCCESS env.
python - <<'PYCHECK'
import torch

print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
major, minor = (int(part) for part in torch.__version__.split(".")[:2])
assert (major, minor) >= (2, 11), f"rfd4-proteina requires torch>=2.11, got {torch.__version__}"
assert torch.version.cuda is not None, f"rfd4-proteina requires a CUDA torch build, got {torch.__version__}"
PYCHECK

echo "RFD4-Proteina setup complete!"
