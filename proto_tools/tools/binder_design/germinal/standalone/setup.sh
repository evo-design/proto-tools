#!/bin/bash
# Provisioning script for the Germinal antibody-design standalone env.
set -euo pipefail
source standalone_helpers.sh

echo "Germinal license notices:"
echo "  PyRosetta — non-commercial / academic only (https://www.rosettacommons.org/software/license-and-download)"
echo "  IgLM — non-commercial academic only (https://github.com/Graylab/IgLM)"

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit "${GERMINAL_CUDA_TOOLKIT_CONSTRAINT:-12.4.*}"

export GERMINAL_TORCH_SPEC="${GERMINAL_TORCH_SPEC:-torch==2.6.*}"
proto_install_pytorch "$GERMINAL_TORCH_SPEC" torchvision torchaudio
export GERMINAL_JAX_SPEC="${GERMINAL_JAX_SPEC:-jax[cuda12]==0.5.3}"
proto_install_jax GERMINAL

GERMINAL_DIR="${VENV_PATH}/data/germinal"
GERMINAL_COMMIT="${GERMINAL_COMMIT:-1e1c1a5b79884ae45abae030c9df90d9423a990a}"
mkdir -p "${VENV_PATH}/data"
if [ ! -d "$GERMINAL_DIR/.git" ]; then
    echo "Cloning SantiagoMille/germinal..."
    git clone https://github.com/SantiagoMille/germinal.git "$GERMINAL_DIR"
fi
git -C "$GERMINAL_DIR" fetch --depth=200 origin
git -C "$GERMINAL_DIR" checkout "$GERMINAL_COMMIT"
echo "Germinal pinned to ${GERMINAL_COMMIT}"

# JAX 0.5+ compat: convert grad_merge_method's string default to a bool dict.
sed -i \
    "s/\"grad_merge_method\": 'pcgrad'/\"grad_merge_method\": {\"scale\": False, \"mgda\": False, \"pcgrad\": True}/" \
    "$GERMINAL_DIR/colabdesign/colabdesign/af/model.py" || true
grep -q '"grad_merge_method": {"scale"' "$GERMINAL_DIR/colabdesign/colabdesign/af/model.py" \
    || { echo "ERROR: grad_merge_method sed patch did not apply to ${GERMINAL_DIR}"; exit 1; }

echo "Installing ColabDesign fork + Germinal (editable)..."
uv pip install -e "$GERMINAL_DIR/colabdesign"
uv pip install -e "$GERMINAL_DIR"

# colabfold (--no-deps): germinal/filters/af3.py imports run_mmseqs2 at module
# scope, and filter_utils.py imports af3 unconditionally for every backend.
echo "Installing colabfold (--no-deps)..."
uv pip install --no-deps colabfold==1.6.1

echo "Installing PyRosetta via conda channel..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" \
    -c https://conda.rosettacommons.org \
    -c conda-forge \
    pyrosetta

# Chai-1 (default validation backend; aarch64 lacks sm_121 support).
if [ "$(uname -m)" = "aarch64" ]; then
    echo "WARNING: skipping chai-lab on aarch64; other structure prediction models are available."
else
    echo "Installing Chai-1..."
    uv pip install "chai-lab==0.6.1"
fi

echo "Installing antibody language models (--no-deps; would otherwise downgrade torch/jax)..."
uv pip install --no-deps ablang2
uv pip install --no-deps iglm
uv pip install rotary_embedding_torch

# colabdesign pulls iglm, whose unpinned transformers resolves to 5.x — broken under
# our torch 2.6 pin (refs torch.float8_e8m0fnu, torch>=2.7). Pin to 4.x.
uv pip install "transformers==4.46.*"

echo "Installing pinned Python dependencies..."
uv pip install -r requirements.txt

proto_resolve_weights_dir germinal
mkdir -p "$WEIGHTS_DIR/params" "$WEIGHTS_DIR/binaries"

# AF-Multimer params (~7 GB). Reuse the AF2 standalone's copy if available
# (env-var redirect; safer than symlinks across filesystems).
AF2_PARAMS=""
if [ -n "${PROTO_ALPHAFOLD2_WEIGHTS_DIR:-}" ] && [ -d "${PROTO_ALPHAFOLD2_WEIGHTS_DIR}/params" ]; then
    AF2_PARAMS="${PROTO_ALPHAFOLD2_WEIGHTS_DIR}/params"
elif [ -n "${PROTO_MODEL_CACHE:-}" ] && [ -d "${PROTO_MODEL_CACHE}/alphafold2/params" ]; then
    AF2_PARAMS="${PROTO_MODEL_CACHE}/alphafold2/params"
elif [ -d "${PROTO_HOME:-$HOME/.proto}/proto_model_cache/alphafold2/params" ]; then
    AF2_PARAMS="${PROTO_HOME:-$HOME/.proto}/proto_model_cache/alphafold2/params"
fi
if [ -n "$AF2_PARAMS" ] && ls "$AF2_PARAMS"/*.npz >/dev/null 2>&1; then
    echo "Reusing AlphaFold-Multimer params from ${AF2_PARAMS}"
    echo "PROTO_GERMINAL_AF_PARAMS_DIR=${AF2_PARAMS}" > "$WEIGHTS_DIR/.params_redirect"
elif ! ls "$WEIGHTS_DIR/params"/*.npz >/dev/null 2>&1; then
    echo "Downloading AlphaFold-Multimer parameters (~7 GB)..."
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar \
        | tar x -C "$WEIGHTS_DIR/params"
fi

# DSSP from conda-forge (salilab is dead + linked an unavailable boost 1.73). Symlink in place —
# copying to $WEIGHTS_DIR breaks RPATH — and always reinstall so its libs stay in the env.
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge dssp 2>/dev/null || true
mkdir -p "$GERMINAL_DIR/params"
if [ -f "$VENV_PATH/bin/mkdssp" ]; then
    ln -sf "$VENV_PATH/bin/mkdssp" "$GERMINAL_DIR/params/dssp"
else
    echo "WARNING: DSSP install failed; secondary-structure-dependent filters may be skipped."
fi

# DAlphaBall (buried-unsat-hbond computation) — compiled from source, non-fatal.
if [ ! -f "$WEIGHTS_DIR/binaries/DAlphaBall.gcc" ]; then
    (
        set +e
        "$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge gfortran gmp 2>/dev/null
        DALPHABALL_DIR=$(mktemp -d)
        git clone --depth 1 https://github.com/outpace-bio/DAlphaBall.git "$DALPHABALL_DIR" 2>&1
        cd "$DALPHABALL_DIR/src" && make 2>&1
        if [ -f DAlphaBall.gcc ]; then
            cp DAlphaBall.gcc "$WEIGHTS_DIR/binaries/DAlphaBall.gcc"
            chmod +x "$WEIGHTS_DIR/binaries/DAlphaBall.gcc"
        else
            echo "WARNING: DAlphaBall compilation failed; delta_unsat_hbonds will be unavailable."
        fi
        rm -rf "$DALPHABALL_DIR"
    ) || echo "WARNING: DAlphaBall build pipeline failed; continuing without it."
fi

# Symlink DAlphaBall so Germinal's relative `params/DAlphaBall.gcc` resolves at runtime
# (inference.py spawns run_germinal.py with cwd=$GERMINAL_DIR; dssp is symlinked above).
mkdir -p "$GERMINAL_DIR/params"
[ -f "$WEIGHTS_DIR/binaries/DAlphaBall.gcc" ] && \
    ln -sf "$WEIGHTS_DIR/binaries/DAlphaBall.gcc" "$GERMINAL_DIR/params/DAlphaBall.gcc"

# Smoke-test the Germinal package imports cleanly
python -c "
import sys
sys.path.insert(0, '$GERMINAL_DIR')
from germinal.utils import config as _gconfig
from germinal.utils import io as _gio
from germinal.design import design as _gdesign
print('Germinal imports OK')
"
echo "Germinal setup complete!"
