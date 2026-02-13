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
conda create -p "$VIRTUAL_ENV/conda_deps" -y -c conda-forge -c bioconda \
    python=3.8 \
    scikit-learn=0.22.1 \
    "numpy<1.24" h5py dill networkx pyyaml regex requests biopython pandas scipy joblib python-levenshtein \
    intarna infernal prodigal hmmer viennarna \
    vmatch clustalo blast fasta3

echo "Patching CRISPRidentify call to disable strand prediction (requires TensorFlow)..."
# CRISPRstrand requires TensorFlow which is too heavy for this env.
# Strand prediction is not needed for tracrRNA detection.
IDENTIFY_RUNNER="$INSTALL_DIR/modules/run_identify_and_identifyer.py"
if grep -q 'fast_run True"' "$IDENTIFY_RUNNER" && ! grep -q -- '--strand False' "$IDENTIFY_RUNNER"; then
    sed -i 's/--fast_run True"/--fast_run True --strand False"/' "$IDENTIFY_RUNNER"
fi

echo "Patching empty-CSV crashes in consistency_score_maker.py and candidate_ranking.py..."
# Upstream bug: both files crash with IndexError when the output CSV is empty
# (i.e., no tracrRNA candidates found for the input sequences).
for PYFILE in "$INSTALL_DIR/modules/consistency_score_maker.py" "$INSTALL_DIR/modules/candidate_ranking.py"; do
    if [ -f "$PYFILE" ] && grep -q 'header, info_lines = lines\[0\]' "$PYFILE" && ! grep -q 'if not lines' "$PYFILE"; then
        sed -i '/lines = .*readlines()/a\        if not lines:\n            return' "$PYFILE"
    fi
done

echo "Setting CRISPR_TRACR_PATH..."
export CRISPR_TRACR_PATH="$INSTALL_DIR"

echo "CRISPRtracrRNA setup complete!"
echo "Set CRISPR_TRACR_PATH=$INSTALL_DIR to use this installation."
