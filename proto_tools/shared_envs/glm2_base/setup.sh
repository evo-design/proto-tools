#!/bin/bash
# Shared env: gLM2-style mixed protein/DNA masked language models (gLM2, Minerva).
set -euo pipefail
source standalone_helpers.sh

proto_install_pytorch
uv pip install -r requirements.txt
# minerva-dna without deps skips its analysis extras (ViennaRNA, plotly, datasets, peft) and torch reinstall.
uv pip install --no-deps minerva-dna==0.1.0
python -c "from minerva import MinervaForMaskedLM; from transformers import AutoModelForMaskedLM, AutoTokenizer; import einops"
echo "gLM2 base env setup complete!"
