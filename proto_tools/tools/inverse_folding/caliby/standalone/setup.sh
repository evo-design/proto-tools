#!/bin/bash
# Setup script for the Caliby standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Caliby standalone environment..."

# Caliby requires torch>=2.6, which is above the floor of some centralized recommendations.
proto_install_pytorch "torch>=2.6,<3"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Caliby declares an SPDX license string, which setuptools>=77 canonicalizes through
# packaging>=24.2. Its build-system requires only "setuptools>=61" and "wheel", so an isolated
# build environment installs a setuptools that needs packaging without installing packaging.
# Building against this environment instead supplies both.
uv pip install "setuptools>=77" "packaging>=24.2" wheel

# Pinned install; uv resolves caliby's nested git deps (atomworks-caliby, protpardelle) at upstream's pins.
CALIBY_COMMIT="41d31560c3c73d7980d94f40f3c852b90bfab5c0"
echo "Installing Caliby at ${CALIBY_COMMIT}..."
uv pip install "caliby @ git+https://github.com/ProteinDesignLab/caliby@${CALIBY_COMMIT}" \
    --no-build-isolation-package caliby \
    --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

# Fail fast if the package import is broken.
if ! python -c "from caliby import load_model, make_constraints, make_ensemble_constraints; print('Caliby import OK')"; then
    echo "ERROR: caliby setup: 'from caliby import load_model' failed after install" >&2
    exit 1
fi

# The caliby install resolves torch itself, so confirm it did not land below the required floor.
if ! python -c "
from importlib.metadata import version
installed = version('torch')
assert tuple(int(part) for part in installed.split('.')[:2]) >= (2, 6), installed
"; then
    echo "ERROR: caliby setup: installing caliby left torch below the required 2.6" >&2
    exit 1
fi

# ============================================================================
# Download model weights into the managed weights directory
# ============================================================================
# inference.py resolves the same path at runtime and exports it as MODEL_PARAMS_DIR.
proto_resolve_weights_dir caliby
echo "Downloading Caliby checkpoints..."
python - <<PY
from huggingface_hub import hf_hub_download

# Mirrors DESIGN_CHECKPOINTS + PACKER_CHECKPOINTS in caliby_sample.py.
for name in (
    "caliby",
    "soluble_caliby",
    "soluble_caliby_v1",
    "caliby_distill",
    "soluble_caliby_distill",
    "caliby_packer_000",
    "caliby_packer_010",
    "caliby_packer_030",
):
    hf_hub_download("ProteinDesignLab/caliby-weights", f"caliby/{name}.ckpt", local_dir="${WEIGHTS_DIR}")
PY

echo "Caliby setup complete!"
