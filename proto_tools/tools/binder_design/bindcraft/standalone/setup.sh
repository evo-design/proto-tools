#!/bin/bash
# Provisioning script for the BindCraft standalone env.
set -euo pipefail
source standalone_helpers.sh

echo "BindCraft license notices:"
echo "  BindCraft — MIT (https://github.com/martinpacesa/BindCraft)"
echo "  PyRosetta — non-commercial / academic only (https://www.rosettacommons.org/software/license-and-download)"
echo "  AlphaFold2 weights — CC BY 4.0 (https://github.com/google-deepmind/alphafold)"

echo "Installing uv package manager..."
pip install uv

proto_install_cuda_toolkit "${BINDCRAFT_CUDA_TOOLKIT_CONSTRAINT:-}"

export BINDCRAFT_JAX_SPEC="${BINDCRAFT_JAX_SPEC:-jax[cuda12]==0.5.3}"
proto_install_jax BINDCRAFT

echo "Installing pinned Python dependencies..."
uv pip install -r requirements.txt

echo "Installing ColabDesign (pinned, --no-deps)..."
uv pip install --no-deps "colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@e31a56fe1d9b4de25c8697f3a28b75892941cc72"

echo "Installing PyRosetta via conda channel..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" \
    -c https://conda.rosettacommons.org \
    -c conda-forge \
    pyrosetta

# AF2 weights are shared with the alphafold2 toolkit (~5.5 GB, avoid re-download).
proto_resolve_weights_dir alphafold2
PARAMS_DIR="${WEIGHTS_DIR}/params"
mkdir -p "$PARAMS_DIR"
# Gate on a completion sentinel, not any .npz, so a partial extraction is re-downloaded.
PARAMS_SENTINEL="${PARAMS_DIR}/.params_complete"
if [ ! -f "$PARAMS_SENTINEL" ]; then
    echo "Downloading AlphaFold2 parameters (~5.5GB)..."
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C "$PARAMS_DIR"
    touch "$PARAMS_SENTINEL"
else
    echo "AlphaFold2 parameters already present at $PARAMS_DIR"
fi

# BindCraft is not a Python package — clone the repo so bindcraft.py + functions/ are on disk.
BINDCRAFT_COMMIT="7cd4ace1b7407adf66a50dfefa47de2270f5e4a9"
BINDCRAFT_DIR="${TOOL_VENV_PATH:-$VIRTUAL_ENV}/data/BindCraft"
if [ ! -d "$BINDCRAFT_DIR/.git" ]; then
    echo "Cloning BindCraft repository..."
    mkdir -p "$(dirname "$BINDCRAFT_DIR")"
    git clone https://github.com/martinpacesa/BindCraft.git "$BINDCRAFT_DIR"
fi
git -C "$BINDCRAFT_DIR" fetch origin
git -C "$BINDCRAFT_DIR" checkout "$BINDCRAFT_COMMIT"
echo "BindCraft pinned to ${BINDCRAFT_COMMIT}"

chmod +x "$BINDCRAFT_DIR/functions/dssp" "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"

# DAlphaBall bundled binary is x86_64-only; rebuild on aarch64.
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "Rebuilding DAlphaBall for aarch64..."
    (
        set +e
        "$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge gfortran gmp 2>/dev/null
        DALPHABALL_DIR=$(mktemp -d)
        git clone --depth 1 https://github.com/outpace-bio/DAlphaBall.git "$DALPHABALL_DIR" 2>&1
        cd "$DALPHABALL_DIR/src" && make 2>&1
        if [ -f DAlphaBall.gcc ]; then
            cp DAlphaBall.gcc "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"
            chmod +x "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"
            echo "DAlphaBall rebuilt for aarch64 successfully."
        else
            echo "WARNING: DAlphaBall aarch64 rebuild failed."
            echo "Buried unsatisfied H-bond metrics may not be computed correctly."
        fi
        rm -rf "$DALPHABALL_DIR"
    ) || {
        echo "WARNING: DAlphaBall aarch64 rebuild failed."
    }
fi

# Wrap DAlphaBall.gcc (run by Rosetta for the BUNS surf_vol calc) so it finds libgfortran.so.5 in
# the conda env lib/ — scoped to this helper, since LD_LIBRARY_PATH is stripped for JAX/CUDA tools.
# Idempotent: only wraps a real ELF.
DAB="$BINDCRAFT_DIR/functions/DAlphaBall.gcc"
if [ -f "$DAB" ] && head -c4 "$DAB" | grep -q "ELF"; then
    mv -f "$DAB" "$BINDCRAFT_DIR/functions/DAlphaBall.real"
    cat > "$DAB" <<'SHIM'
#!/bin/sh
# proto-tools: resolve libgfortran from the conda env lib/ for this helper only.
here="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export LD_LIBRARY_PATH="$here/../../../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/DAlphaBall.real" "$@"
SHIM
    chmod +x "$DAB"
    echo "Wrapped DAlphaBall.gcc to resolve libgfortran from the conda env lib/."
fi

echo "BindCraft setup complete!"
