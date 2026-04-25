#!/bin/bash
set -euo pipefail

echo "Setting up DSSP standalone environment..."

echo "Installing DSSP binary from conda-forge..."
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge dssp

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Verifying DSSP installation..."
python - <<'PY'
import shutil
from Bio.PDB.DSSP import DSSP  # noqa: F401

if not (shutil.which("mkdssp") or shutil.which("dssp")):
    raise SystemExit("DSSP binary not found on PATH")
print("DSSP OK")
PY

echo "DSSP setup complete!"
