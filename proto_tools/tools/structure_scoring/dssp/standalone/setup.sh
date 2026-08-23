#!/bin/bash
set -euo pipefail

echo "Setting up DSSP standalone environment..."

echo "Installing DSSP binary from conda-forge..."
# DSSP 4.6.1 is built against the libmcfp 1.4.2 ABI. libmcfp 2.x is not
# binary-compatible and makes mkdssp fail at startup with an undefined symbol.
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge "dssp=4.6.1" "libmcfp=1.4.2"

echo "Installing uv package manager..."
pip install uv

echo "Installing Python dependencies..."
uv pip install -r requirements.txt

echo "Verifying DSSP installation..."
python - <<'PY'
import shutil
import subprocess

from Bio.PDB.DSSP import DSSP  # noqa: F401

executable = shutil.which("mkdssp") or shutil.which("dssp")
if not executable:
    raise SystemExit("DSSP binary not found on PATH")
subprocess.run([executable, "--version"], check=True)
print("DSSP OK")
PY

echo "DSSP setup complete!"
