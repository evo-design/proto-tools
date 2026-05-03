#!/bin/bash
# Setup script for AlphaFold2 (ColabDesign) standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AlphaFold2 (ColabDesign) standalone environment..."

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit "${ALPHAFOLD2_CUDA_TOOLKIT_CONSTRAINT:-}"

echo "Installing JAX, ColabDesign, and deps..."
uv pip install -r requirements.txt

echo "Installing Germinal ColabDesign fork (gradient backend with alpha/bias and framework contacts)..."
GERMINAL_DIR="${TOOL_VENV_PATH:-$VIRTUAL_ENV}/data/colabdesign_germinal"
mkdir -p "$GERMINAL_DIR"
uv pip install --no-deps --target "$GERMINAL_DIR" "colabdesign @ git+https://github.com/SantiagoMille/germinal.git@1e1c1a5#subdirectory=colabdesign"
uv pip install cvxopt

# JAX 0.5+ compat patches for the Germinal fork:
# 1. grad_merge_method: string default breaks JAX JIT — convert to bool dict
sed -i "s/\"grad_merge_method\": 'pcgrad'/\"grad_merge_method\": {\"scale\": False, \"mgda\": False, \"pcgrad\": True}/" "$GERMINAL_DIR/colabdesign/af/model.py"
grep -q '"grad_merge_method": {"scale"' "$GERMINAL_DIR/colabdesign/af/model.py" \
  || { echo "ERROR: alphafold2 setup: grad_merge_method sed patch did not apply (Germinal ColabDesign fork)" >&2; exit 1; }
# 2. iglm/ablang: imported at module level but require torch — stub them out
#    (our pipeline scores AbLang externally via proto-tools' ablang-gradient tool)
mkdir -p "$GERMINAL_DIR/colabdesign/iglm" "$GERMINAL_DIR/colabdesign/ablang"
cat > "$GERMINAL_DIR/colabdesign/iglm/model.py" << 'IGLM_STUB'
class CustomIgLM:
    def __init__(self, **kw):
        self.is_scfv = kw.get("is_scfv", False)
    def get_ablm_grad(self, seq, method=None):
        import numpy as np
        return np.zeros_like(seq), 0.0
IGLM_STUB
touch "$GERMINAL_DIR/colabdesign/iglm/__init__.py"
cat > "$GERMINAL_DIR/colabdesign/ablang/model.py" << 'ABLANG_STUB'
import numpy as np

class CustomAbLang:
    def __init__(self, **kwargs):
        self.is_scfv = kwargs.get("is_scfv", False)
        self.vh_len = kwargs.get("vh_len")
        self.vl_len = kwargs.get("vl_len")
        self.vh_first = kwargs.get("vh_first", True)
    def get_ablm_grad(self, seq, method=None, pll_chunk_size=None):
        return np.zeros_like(seq), 0.0
ABLANG_STUB

# Download AF2 parameters (~3.5GB)
proto_resolve_weights_dir alphafold2
WEIGHTS_DIR="${WEIGHTS_DIR}/params"
mkdir -p "$WEIGHTS_DIR"
PARAMS_DIR="$WEIGHTS_DIR"
if [ ! -d "$PARAMS_DIR" ] || [ -z "$(ls -A "$PARAMS_DIR"/*.npz 2>/dev/null)" ]; then
    echo "Downloading AlphaFold2 parameters (~3.5GB)..."
    mkdir -p "$PARAMS_DIR"
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C "$PARAMS_DIR"
    echo "AlphaFold2 parameters downloaded to $PARAMS_DIR"
else
    echo "AlphaFold2 parameters already present at $PARAMS_DIR"
fi

echo "AlphaFold2 (ColabDesign) setup complete!"
