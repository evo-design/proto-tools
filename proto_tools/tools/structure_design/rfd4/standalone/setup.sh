#!/bin/bash
# Setup script for the RFD4-Proteina standalone environment.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up RFD4-Proteina standalone environment..."

# Platform gate: rfd4-proteina is published Linux-only, and its mandatory all-cuda12
# extra pulls tmol-ops-cu12 / cuequivariance-ops-cu12, whose wheels are Linux x86_64
# only. Refuse here rather than failing mid-resolution.
if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "RFD4-Proteina requires Linux x86_64 (tmol / cuequivariance CUDA wheel constraint)." >&2
    echo "Detected: $(uname -s) $(uname -m)" >&2
    exit 1
fi

# RFD4 has no CPU path: upstream builds its Lightning trainer with accelerator="gpu"
# unconditionally. Refuse a CPU-detected host explicitly, because the CPU torch index does
# carry torch>=2.11, so the build would otherwise succeed, mark the env SUCCESS, and fail only
# at inference -- and the setup hash ignores the detected platform, so that CPU env would be
# reused even after a GPU appears on the same PROTO_HOME.
if [ "${DETECTED_COMPUTE_PLATFORM:-cpu}" != "cuda" ]; then
    echo "RFD4-Proteina requires a CUDA GPU; upstream builds its Lightning trainer with" >&2
    echo "accelerator=\"gpu\" unconditionally, so there is no CPU path." >&2
    echo "Detected compute platform: ${DETECTED_COMPUTE_PLATFORM:-cpu}" >&2
    exit 1
fi
# torch>=2.11 wheels exist only on the cu126 and cu128 indices, which centralized detection
# selects for driver >= 525; a cu118 host cannot resolve them at all.
if [ "${DETECTED_DRIVER_VERSION:-0}" -lt 525 ]; then
    echo "RFD4-Proteina needs torch>=2.11, which requires NVIDIA driver >= 525 (CUDA 12+)." >&2
    echo "Detected driver major version: ${DETECTED_DRIVER_VERSION:-unknown}" >&2
    exit 1
fi

# rfd4-proteina requires torch>=2.11, above the centralized floor (2.8 on driver 570+),
# so pin the floor here while keeping the hardware-matched wheel index.
proto_install_pytorch "torch>=2.11,<3"

# cuEquivariance and TMol ship prebuilt kernels (and RFD4 defaults CUEQ_DISABLE_AOT_TUNING=1),
# but torch.compile — reachable through use_torch_compile=True — JITs via inductor/triton,
# which shells out to cc/nvcc. The runtime PATH carries no guaranteed host compiler, so
# install a tool-local CUDA toolkit + gcc, as rf3 does.
#
# The three xorg libs are not optional. rfd4-proteina depends on atomworks[ml,openbabel],
# and `rfproteina.generate` imports atomworks' openbabel_utils at module scope, which
# imports `openbabel.pybel`. Every one of Open Babel's 122 format plugins links against
# libXrender/libXext/libX11, so on a headless host without them each plugin fails to load
# and pybel's own broken-plugin bookkeeping then dies with
# "ValueError: not enough values to unpack (expected 2, got 1)" at import time -- before
# any design runs. env_vars.txt puts cuda_env/lib on LD_LIBRARY_PATH, so installing them
# here is what makes them resolvable at runtime.
proto_install_cuda_toolkit "${RFD4_CUDA_TOOLKIT_CONSTRAINT:-}" cuda-nvcc "gcc=12.*" "gxx=12.*" \
    xorg-libxrender xorg-libxext xorg-libx11

# The all-cuda12 extra lives in requirements.txt rather than on a `uv pip install
# --torch-backend` line: torch is already installed from the detected CUDA index above, and
# every torch floor in the extra's closure (tmol, tmol-ops-cu12, lightning, peft, atomworks)
# is <= 2.11, so uv keeps the installed wheel instead of pulling PyPI's default CUDA build.
echo "Installing rfd4-proteina[all-cuda12] and dependencies..."
uv pip install -r requirements.txt

# DATA_PATH backs the native-checkpoint config nodes (ckpt_path, autoencoder_ckpt_path) that
# our default hf_ckpt_path overrides. env_vars.txt points DATA_PATH here so an OmegaConf
# resolve of those nodes can never die on an unset interpolation.
mkdir -p "$VENV_PATH/rfd4_data"

# Model weights are deliberately NOT fetched here: the nvidia/RFD4-Proteina repo is ungated
# and huggingface-hub downloads it on first run into the HF_HOME the framework points at the
# shared model cache, so warming it at build time would only duplicate the ~2.8 GB bundle and
# make env setup depend on the Hub.

if [ ! -x "$VENV_PATH/bin/rfd4-generate" ]; then
    echo "ERROR: rfd4-generate console script missing from $VENV_PATH/bin after install." >&2
    exit 1
fi
# Assert rather than print: the "uv keeps the installed wheel" argument above is the only thing
# standing between us and a silently CPU-only or too-old torch, and a wrong answer there surfaces
# as an inference-time failure against an env marked SUCCESS.
python - <<'PYCHECK'
import torch

print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
major, minor = (int(part) for part in torch.__version__.split(".")[:2])
assert (major, minor) >= (2, 11), f"rfd4-proteina requires torch>=2.11, got {torch.__version__}"
assert torch.version.cuda is not None, f"rfd4-proteina requires a CUDA torch build, got {torch.__version__}"
PYCHECK

echo "RFD4-Proteina setup complete!"
