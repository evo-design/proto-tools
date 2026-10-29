#!/bin/bash
# Evo 2 Flash uses vLLM's CUDA 12.9 wheels, including on CUDA 13 drivers.
set -euo pipefail

if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "ERROR: Evo 2 Flash requires Linux x86_64 and an NVIDIA GPU" >&2
    exit 1
fi

VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv pip install --python "$VENV_PATH/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    "torch==2.11.0+cu129" "$VLLM_WHEEL" "torchcodec==0.16.0" \
    "evo2-flash @ git+https://github.com/evo-design/evo2_flash.git@c71eb6f4bee7d45683c3a4b76a4eacf0dda0e189"
