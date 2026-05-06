#!/bin/bash
# Setup script for BindCraft standalone environment.
#
# BindCraft is an AlphaFold2-hallucination + ProteinMPNN + PyRosetta binder-design
# pipeline. This script provisions:
#   1. CUDA toolkit + JAX (matching the alphafold2 toolkit's pin)
#   2. BindCraft's pinned Python deps (numpy/pandas/flax/etc., all pinned)
#   3. ColabDesign at BindCraft's official pinned commit
#   4. PyRosetta from the Rosetta Commons conda channel
#   5. AlphaFold2 weights — SHARED with the alphafold2 toolkit so users with
#      AF2 already installed don't re-download (~5.5 GB)
#   6. The pinned BindCraft repo clone (for bindcraft.py + functions/)
#   7. DAlphaBall fallback rebuild on aarch64 (the bundled binary is x86_64)

set -euo pipefail
source standalone_helpers.sh

echo "=============================================="
echo "BindCraft + PyRosetta License Notice"
echo "=============================================="
echo "BindCraft is distributed under the MIT License."
echo "  https://github.com/martinpacesa/BindCraft"
echo ""
echo "PyRosetta is distributed under the Rosetta Software"
echo "License. Free for academic and non-commercial use."
echo "Commercial users must obtain a license from UW CoMotion."
echo "  https://www.rosettacommons.org/software/license-and-download"
echo ""
echo "AlphaFold2 parameters are released under CC BY 4.0."
echo "  https://github.com/google-deepmind/alphafold"
echo ""
echo "By proceeding, you accept these terms."
echo "=============================================="

echo "Setting up BindCraft standalone environment..."

echo "Installing uv package manager..."
pip install uv

# CUDA toolkit (cuda-toolkit + cudnn) for JAX
proto_install_cuda_toolkit "${BINDCRAFT_CUDA_TOOLKIT_CONSTRAINT:-}"

# JAX with CUDA12 plugin — matches alphafold2 standalone (jax[cuda12]==0.5.3),
# satisfies BindCraft's `jax>=0.4,<=0.6.0` constraint.
export BINDCRAFT_JAX_SPEC="${BINDCRAFT_JAX_SPEC:-jax[cuda12]==0.5.3}"
proto_install_jax BINDCRAFT

echo "Installing BindCraft Python dependencies..."
uv pip install -r requirements.txt

# ColabDesign — pinned to the same commit BindCraft's official install_bindcraft.sh
# resolves (latest sokrypton/ColabDesign main as of 2025-10-23). --no-deps mirrors
# upstream (BindCraft's deps are already pinned in requirements.txt).
echo "Installing ColabDesign (pinned)..."
uv pip install --no-deps "colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@e31a56fe1d9b4de25c8697f3a28b75892941cc72"

# PyRosetta — same install path used by tools/structure_scoring/pyrosetta.
# The conda channel ships latest stable per platform; no version pin.
echo "Installing PyRosetta from Rosetta Commons conda channel..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" \
    -c https://conda.rosettacommons.org \
    -c conda-forge \
    pyrosetta

# AlphaFold2 weights — shared with the alphafold2 toolkit so users who already
# have AF2 installed don't re-download (~5.5 GB).
echo "Resolving AlphaFold2 weights (shared with alphafold2 toolkit)..."
proto_resolve_weights_dir alphafold2
PARAMS_DIR="${WEIGHTS_DIR}/params"
mkdir -p "$PARAMS_DIR"
if [ -z "$(ls -A "$PARAMS_DIR"/*.npz 2>/dev/null)" ]; then
    echo "Downloading AlphaFold2 parameters (~5.5GB)..."
    curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C "$PARAMS_DIR"
    echo "AlphaFold2 parameters downloaded to $PARAMS_DIR"
else
    echo "AlphaFold2 parameters already present at $PARAMS_DIR (shared with alphafold2 toolkit)"
fi

# Clone the pinned BindCraft repo. We need bindcraft.py + functions/ on disk
# (BindCraft is not a Python package — it's a script invoked from its repo root).
BINDCRAFT_COMMIT="7cd4ace1b7407adf66a50dfefa47de2270f5e4a9"
BINDCRAFT_DIR="${TOOL_VENV_PATH:-$VIRTUAL_ENV}/data/BindCraft"
if [ ! -d "$BINDCRAFT_DIR/.git" ]; then
    echo "Cloning BindCraft repository..."
    mkdir -p "$(dirname "$BINDCRAFT_DIR")"
    git clone https://github.com/martinpacesa/BindCraft.git "$BINDCRAFT_DIR"
fi
echo "Pinning BindCraft to commit $BINDCRAFT_COMMIT..."
git -C "$BINDCRAFT_DIR" fetch origin
git -C "$BINDCRAFT_DIR" checkout "$BINDCRAFT_COMMIT"

# DSSP + DAlphaBall ship pre-built in functions/ — make them executable.
chmod +x "$BINDCRAFT_DIR/functions/dssp" "$BINDCRAFT_DIR/functions/DAlphaBall.gcc"

# DAlphaBall fallback rebuild for aarch64 (bundled binary is x86_64-only).
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo "Rebuilding DAlphaBall for aarch64 (the bundled binary is x86_64-only)..."
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

echo "BindCraft setup complete!"
