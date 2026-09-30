#!/bin/bash
set -euo pipefail

echo "Setting up DSSP standalone environment..."

echo "Installing DSSP binary from conda-forge..."
# libmcfp=1.4.2 excludes the dssp 4.6.1 _0/_1 builds, whose mkdssp needs a libmcfp symbol no release has.
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge "dssp=4.6.1" "libmcfp=1.4.2"

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
