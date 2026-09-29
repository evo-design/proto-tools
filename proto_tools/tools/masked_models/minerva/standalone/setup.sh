#!/bin/bash
# Managed mixed-model inference environment.
set -euo pipefail
source standalone_helpers.sh

pip install uv
proto_install_pytorch
uv pip install -r requirements.txt
uv pip install --no-deps minerva-dna==0.1.0
python -c "from minerva import MinervaForMaskedLM; from transformers import AutoTokenizer"
echo "Minerva setup complete!"
