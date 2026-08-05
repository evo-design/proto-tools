#!/bin/bash
set -euo pipefail
source standalone_helpers.sh

echo "Setting up AutoDock Vina standalone environment..."

echo "Installing AutoDock Vina from conda-forge..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge "vina=1.2.7"

echo "Installing uv package manager..."
pip install uv

echo "Installing Meeko and chemistry dependencies..."
uv pip install -r requirements.txt

echo "Verifying AutoDock Vina environment..."
python - <<'PY'
from meeko import MoleculePreparation, PDBQTMolecule, Polymer, RDKitMolCreate
from rdkit import Chem
from vina import Vina

assert Chem.MolFromSmiles("CCO") is not None
assert MoleculePreparation is not None
assert PDBQTMolecule is not None
assert Polymer is not None
assert RDKitMolCreate is not None
assert Vina is not None
print("AutoDock Vina environment OK")
PY

echo "AutoDock Vina setup complete!"
