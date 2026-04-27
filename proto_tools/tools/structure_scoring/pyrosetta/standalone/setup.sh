#!/bin/bash
# Setup script for PyRosetta standalone environment
set -euo pipefail

echo "=============================================="
echo "PyRosetta License Notice"
echo "PyRosetta is distributed under the Rosetta"
echo "Software License. Free for academic and"
echo "non-commercial use. Commercial users must"
echo "obtain a license from UW CoMotion."
echo "https://www.rosettacommons.org/software/license-and-download"
echo "By proceeding, you accept these terms."
echo "=============================================="

ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo ""
    echo "##############################################################################"
    echo "# WARNING: PyRosetta on Linux aarch64 uses a stale 2023.11 build."
    echo "#"
    echo "# The conda.rosettacommons.org linux-aarch64 channel has not been updated"
    echo "# since March 2023. The newest available pyrosetta is 2023.11 (py39/py310"
    echo "# only), while linux-64 and osx-arm64 are at 2026.06."
    echo "#"
    echo "# Scoring functions, FastRelax, and SAP have evolved over the past ~3 years."
    echo "# Results computed on linux-aarch64 may differ from results on x86_64 or"
    echo "# macOS. Use this build for prototyping; reproduce final numbers on x86_64."
    echo "#"
    echo "# Compare channels for yourself:"
    echo "#   linux-aarch64: https://conda.rosettacommons.org/linux-aarch64/repodata.json"
    echo "#   linux-64:      https://conda.rosettacommons.org/linux-64/repodata.json"
    echo "#   osx-arm64:     https://conda.rosettacommons.org/osx-arm64/repodata.json"
    echo "##############################################################################"
    echo ""
fi

echo "Installing uv package manager..."
pip install uv

echo "Installing PyRosetta via conda channel..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" \
    -c https://conda.rosettacommons.org \
    -c conda-forge \
    pyrosetta

echo "Installing additional Python dependencies..."
uv pip install -r requirements.txt

echo "Building DAlphaBall for buried unsatisfied H-bond computation..."
(
    set +e
    "$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge gfortran gmp 2>/dev/null
    DALPHABALL_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/outpace-bio/DAlphaBall.git "$DALPHABALL_DIR" 2>&1
    cd "$DALPHABALL_DIR/src" && make 2>&1
    if [ -f DAlphaBall.gcc ]; then
        cp DAlphaBall.gcc "$VENV_PATH/bin/DAlphaBall"
        chmod +x "$VENV_PATH/bin/DAlphaBall"
        echo "DAlphaBall installed successfully."
    else
        echo "WARNING: DAlphaBall compilation failed."
        echo "Buried unsatisfied H-bonds (delta_unsat_hbonds) will not be computed."
    fi
    rm -rf "$DALPHABALL_DIR"
) || {
    echo "WARNING: DAlphaBall build failed. delta_unsat_hbonds will not be computed."
}

echo "Verifying PyRosetta installation..."
python -c "import pyrosetta; pyrosetta.init('-mute all'); print('PyRosetta OK')"

echo "PyRosetta setup complete!"
