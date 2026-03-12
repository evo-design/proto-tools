#!/bin/bash
# Setup script for CRISPRtracrRNA standalone environment
set -euo pipefail

echo "Setting up CRISPRtracrRNA standalone environment..."

echo "Cloning CRISPRtracrRNA repository..."
INSTALL_DIR="${PREFIX:-$VENV_PATH}/CRISPRtracrRNA"
if [ ! -d "$INSTALL_DIR" ]; then
    git clone https://github.com/BackofenLab/CRISPRtracrRNA.git "$INSTALL_DIR"
fi

echo "Cloning CRISPRidentify into CRISPRtracrRNA tools directory..."
IDENTIFY_DIR="$INSTALL_DIR/tools/CRISPRidentify/CRISPRidentify"
if [ ! -f "$IDENTIFY_DIR/CRISPRidentify.py" ]; then
    git clone https://github.com/BackofenLab/CRISPRidentify.git "$IDENTIFY_DIR"
fi

echo "Cloning CRISPRcasIdentifier into CRISPRtracrRNA tools directory..."
CAS_ID_DIR="$INSTALL_DIR/tools/CRISPRcasIdentifier/CRISPRcasIdentifier"
if [ ! -f "$CAS_ID_DIR/CRISPRcasIdentifier.py" ]; then
    git clone https://github.com/BackofenLab/CRISPRcasIdentifier.git "$CAS_ID_DIR"
fi

echo "Creating isolated conda environment (Python 3.8 + scikit-learn 0.22)..."
echo "CRISPRidentify's pickled models require sklearn 0.22 (incompatible with 3.12)."
echo "Using $VENV_PATH/conda_deps to avoid polluting base env..."

# Detect platform for package installation.
# scikit-learn 0.22, vmatch, and several bioconda tools only have x86_64
# builds.  On macOS arm64 we force osx-64 packages and run via Rosetta 2.
# Linux aarch64 is unsupported because there is no transparent x86_64
# emulation layer equivalent to Rosetta.
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" = "Linux" ] && [ "$ARCH" != "x86_64" ]; then
    echo "ERROR: CRISPRtracrRNA requires x86_64 bioconda packages (vmatch, etc.)" >&2
    echo "       that are not available on Linux $ARCH." >&2
    exit 1
fi

# Platform for package installation (force x86_64 on macOS arm64)
MAMBA_EXTRA_ARGS=()
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "Detected macOS arm64 — using osx-64 packages via Rosetta 2..."
    MAMBA_EXTRA_ARGS=(--platform osx-64)
fi

# Use the project-level micromamba provided by tool_instance.py
export MAMBA_ROOT_PREFIX="$VENV_PATH/micromamba"
eval "$("$MAMBA_BIN" shell hook -s posix)"

echo "Creating conda environment with micromamba..."
"$MAMBA_BIN" create -p "$VENV_PATH/conda_deps" -y -c conda-forge -c bioconda \
    "${MAMBA_EXTRA_ARGS[@]}" \
    python=3.8 \
    scikit-learn=0.22.1 \
    "numpy<1.24" h5py dill networkx pyyaml regex requests biopython pandas scipy joblib python-levenshtein \
    intarna infernal prodigal hmmer viennarna \
    vmatch clustalo blast fasta3

echo "Applying upstream patches..."
python3 -c "
from pathlib import Path; import sys; d = Path(sys.argv[1])
patches = [
    ('modules/run_identify_and_identifyer.py',
     '--fast_run True\"', '--fast_run True --strand False\"', '--strand False'),
    ('modules/consistency_score_maker.py',
     'header, info_lines = lines[0]',
     'if not lines:\n            return\n        header, info_lines = lines[0]', 'if not lines'),
    ('modules/candidate_ranking.py',
     'header, info_lines = lines[0]',
     'if not lines:\n            return\n        header, info_lines = lines[0]', 'if not lines'),
    # Fix fasta36 arg order: -m8 before positional args (macOS Rosetta compat, #93)
    ('modules/anti_repeat_search.py', ' -m 8', '', '-m8'),
    ('modules/anti_repeat_search.py', 'fasta36 ', 'fasta36 -m8 ', 'fasta36 -m8'),
]
for rel, find, repl, guard in patches:
    p = d / rel
    if not p.exists(): continue
    t = p.read_text()
    if guard in t or find not in t: print(f'  OK: {rel}'); continue
    p.write_text(t.replace(find, repl)); print(f'  PATCHED: {rel}')
" "$INSTALL_DIR"

echo "Setting CRISPR_TRACR_PATH..."
export CRISPR_TRACR_PATH="$INSTALL_DIR"

echo "CRISPRtracrRNA setup complete!"
echo "Set CRISPR_TRACR_PATH=$INSTALL_DIR to use this installation."
