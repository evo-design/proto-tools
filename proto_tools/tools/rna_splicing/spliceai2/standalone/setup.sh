#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up SpliceAI2 standalone environment..."

# The checkpoints are gated on HuggingFace; fail fast before installing anything.
proto_check_gated_hf_repo "illumina-ai/SpliceAI2" "https://huggingface.co/illumina-ai/SpliceAI2" "README.md"

proto_install_pytorch

echo "Installing SpliceAI2 and dependencies..."
uv pip install -r requirements.txt

# Fetch both ensemble checkpoints into the managed weights directory (inference.py
# resolves the same path at runtime via resolve_weights_dir("spliceai2")).
proto_resolve_weights_dir spliceai2
echo "Downloading SpliceAI2 checkpoints..."
python - <<PY
from huggingface_hub import hf_hub_download

for name in ("spliceai2_multispecies_13M_01", "spliceai2_multispecies_13M_02"):
    hf_hub_download("illumina-ai/SpliceAI2", f"{name}/model.ckpt", local_dir="${WEIGHTS_DIR}")
PY

# Fail fast if the package import is broken.
python -c "from spliceai2.model import SpliceAI2, VariantAnnotator; from spliceai2.utils import decode_topk_tx; print('SpliceAI2 OK')"

echo "SpliceAI2 setup complete!"
