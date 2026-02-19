#!/bin/bash
set -euo pipefail

echo "Setting up SpliceTransformer standalone environment..."

echo "Installing uv package manager..."
pip install uv

DRIVER_MAJOR=""
CUDA_MAJOR=""
if command -v nvidia-smi &> /dev/null; then
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1 || true)
    CUDA_MAJOR=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\..*/\1/p' | head -n1 || true)
fi

TORCH_SPEC="${SPLICE_TRANSFORMER_TORCH_SPEC:-}"
if [ -z "$TORCH_SPEC" ]; then
    if [ -n "$DRIVER_MAJOR" ]; then
        if [ "$DRIVER_MAJOR" -ge 570 ]; then
            TORCH_SPEC="torch>=2.10,<3"
        elif [ "$DRIVER_MAJOR" -ge 550 ]; then
            TORCH_SPEC="torch>=2.7,<2.10"
        elif [ "$DRIVER_MAJOR" -ge 535 ]; then
            TORCH_SPEC="torch>=2.4,<2.8"
        else
            TORCH_SPEC="torch>=2.1,<2.5"
        fi
    else
        TORCH_SPEC="torch>=2.1,<3"
    fi
fi

echo "Detected driver=${DRIVER_MAJOR:-unknown} cuda=${CUDA_MAJOR:-unknown}"
echo "Installing Torch using spec: ${TORCH_SPEC}"
uv pip install --torch-backend=auto "${TORCH_SPEC}"

echo "Installing remaining Python dependencies..."
uv pip install -r requirements.txt

echo "SpliceTransformer setup complete!"
