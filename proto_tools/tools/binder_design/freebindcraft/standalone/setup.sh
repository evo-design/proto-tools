#!/bin/bash
# Provisioning script for the FreeBindCraft standalone env (PyRosetta-free).
#
# FreeBindCraft (cytokineking/FreeBindCraft) is a drop-in fork of BindCraft that
# replaces every PyRosetta-dependent step with open-source equivalents: OpenMM +
# PDBFixer for relaxation, FreeSASA/Biopython for surface analysis, and the sc-rs
# binary for shape complementarity. It auto-detects the missing pyrosetta import
# and falls back to those routines; we additionally pass --no-pyrosetta at runtime.
# JAX is pinned to 0.6.0 and the image base to cuda 12.1.1 (matches FreeBindCraft's
# official Dockerfile env, where the AF2 forward pass compiles and runs without a
# SIGSEGV).
set -euo pipefail
source standalone_helpers.sh

echo "FreeBindCraft license notices:"
echo "  BindCraft — MIT (https://github.com/martinpacesa/BindCraft)"
echo "  FreeBindCraft — MIT (https://github.com/cytokineking/FreeBindCraft)"
echo "  AlphaFold2 weights — CC BY 4.0 (https://github.com/google-deepmind/alphafold)"

echo "Installing uv package manager..."
pip install uv

# Skip proto_install_cuda_toolkit: jax[cuda12] brings its own nvidia-cu12 wheel
# libs, and the GPU image base already provides system CUDA + cuDNN. A third
# (micromamba) toolkit only adds version conflicts — FreeBindCraft's own install
# script likewise installs JAX without a separate toolkit step.
export BINDCRAFT_JAX_SPEC="${BINDCRAFT_JAX_SPEC:-jax[cuda12]==0.6.0}"
proto_install_jax BINDCRAFT

echo "Installing pinned Python dependencies..."
uv pip install -r requirements.txt

echo "Installing ColabDesign (pinned, --no-deps)..."
uv pip install --no-deps "colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@e31a56fe1d9b4de25c8697f3a28b75892941cc72"

# FreeBindCraft no-pyrosetta deps: OpenMM (relax), PDBFixer (cleanup), FreeSASA.
echo "Installing OpenMM + PDBFixer via conda channel..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" \
    -c conda-forge \
    openmm pdbfixer

echo "Installing FreeSASA Python module (--no-deps; brings only its C wheel)..."
uv pip install --no-deps freesasa

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

# Clone FreeBindCraft (ships sc/FASPR/dssp binaries + pr_alternative_utils.py).
BINDCRAFT_COMMIT="${BINDCRAFT_COMMIT:-d12747dbc907435622559b81891ad73e0a45c2e4}"
BINDCRAFT_DIR="${TOOL_VENV_PATH:-$VIRTUAL_ENV}/data/BindCraft"
if [ ! -d "$BINDCRAFT_DIR/.git" ]; then
    echo "Cloning FreeBindCraft repository..."
    mkdir -p "$(dirname "$BINDCRAFT_DIR")"
    git clone https://github.com/cytokineking/FreeBindCraft.git "$BINDCRAFT_DIR"
fi
git -C "$BINDCRAFT_DIR" fetch origin
git -C "$BINDCRAFT_DIR" checkout "$BINDCRAFT_COMMIT"
echo "FreeBindCraft pinned to ${BINDCRAFT_COMMIT}"

# DAlphaBall is PyRosetta-only and omitted in --no-pyrosetta mode.
for bin_name in dssp sc FASPR; do
    bin_path="$BINDCRAFT_DIR/functions/$bin_name"
    if [ -f "$bin_path" ]; then
        chmod +x "$bin_path"
    else
        echo "WARNING: $bin_name binary missing at $bin_path; some filters may be unavailable."
    fi
done

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "WARNING: FreeBindCraft binaries are x86_64; aarch64 hosts must rebuild sc/FASPR or expect placeholder values."
fi

echo "FreeBindCraft setup complete!"
