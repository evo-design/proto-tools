#!/bin/bash
# Setup script for CRISPRtracrRNA standalone environment
set -euo pipefail

echo "Setting up CRISPRtracrRNA standalone environment..."

echo "Cloning CRISPRtracrRNA repository..."
INSTALL_DIR="${PREFIX:-$VIRTUAL_ENV}/CRISPRtracrRNA"
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
echo "Using $VIRTUAL_ENV/conda_deps to avoid polluting base env..."

# Detect platform for micromamba download and package installation.
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

# Platform for downloading micromamba binary
if [ "$OS" = "Linux" ]; then
    MAMBA_PLATFORM="linux-64"
elif [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        MAMBA_PLATFORM="osx-arm64"
    else
        MAMBA_PLATFORM="osx-64"
    fi
else
    echo "Unsupported platform: $OS $ARCH"
    exit 1
fi

# Platform for package installation (force x86_64 on macOS arm64)
MAMBA_EXTRA_ARGS=()
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "Detected macOS arm64 — using osx-64 packages via Rosetta 2..."
    MAMBA_EXTRA_ARGS=(--platform osx-64)
fi

# Install micromamba if not already available
MAMBA_ROOT="$VIRTUAL_ENV/micromamba"
MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
if [ ! -x "$MAMBA_BIN" ]; then
    echo "Installing micromamba ($MAMBA_PLATFORM)..."
    mkdir -p "$MAMBA_ROOT/bin"
    curl -fsSL "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" | tar -xvj -C "$MAMBA_ROOT/bin" --strip-components=1 bin/micromamba
fi

# Initialize micromamba for this shell
export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
eval "$("$MAMBA_BIN" shell hook -s posix)"

echo "Creating conda environment with micromamba..."
"$MAMBA_BIN" create -p "$VIRTUAL_ENV/conda_deps" -y -c conda-forge -c bioconda \
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
