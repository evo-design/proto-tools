#!/bin/bash
set -euo pipefail

echo "Setting up DSSP standalone environment..."

echo "Installing DSSP binary from conda-forge..."
# NOTE: conda-forge's dssp=4.6.1 (and the dssp=4.6.0 "_1" rebuild) declare a
# runtime dependency on libmcfp>=2.0.1,<3.0a0, but the mkdssp binary in those
# builds actually calls mcfp::config::get_last_option_ref(), a symbol that
# does not exist in ANY published libmcfp release (1.4.2 through at least
# 2.1.1 only ever expose get_last_option(), no "_ref" variant -- see
# https://mhekkel.github.io/libmcfp/api/classmcfp_1_1config.html). This makes
# dssp=4.6.1 permanently broken on conda-forge: pinning libmcfp=1.4.2
# alongside it (the old fix) no longer even solves, since dssp=4.6.1 declares
# libmcfp>=2.0.1,<3.0a0 -- and pinning any *available* libmcfp version still
# fails at runtime with "mkdssp: undefined symbol
# _ZN4mcfp6config19get_last_option_refB5cxx11Ev", since the symbol was never
# actually published.
#
# dssp=4.6.0 build "_0" (np2py312h4f94bcb_0) predates the libmcfp dependency
# entirely -- it was only added in the "_1" rebuild of the same version -- so
# pin the exact build string to sidestep the broken symbol altogether.
"$MAMBA_BIN" install -y -p "$VENV_PATH" -c conda-forge --force-reinstall "dssp=4.6.0=np2py312h4f94bcb_0"

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
