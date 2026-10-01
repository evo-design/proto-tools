#!/bin/bash
# Managed mixed-model inference environment.
set -euo pipefail
source standalone_helpers.sh

proto_install_pytorch
uv pip install -r requirements.txt
python -c "from transformers import AutoModelForMaskedLM, AutoTokenizer; import einops"
echo "GLM2 setup complete!"
