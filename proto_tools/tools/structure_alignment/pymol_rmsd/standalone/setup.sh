#!/bin/bash
set -euo pipefail

echo "Setting up PyMOL RMSD standalone environment..."

echo "Installing Open-Source PyMOL from conda-forge..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge pymol-open-source

echo "Verifying PyMOL installation..."
python - <<'PY'
import pymol  # noqa: F401
from pymol import cmd  # noqa: F401

print("PyMOL OK")
PY

echo "PyMOL RMSD setup complete!"
