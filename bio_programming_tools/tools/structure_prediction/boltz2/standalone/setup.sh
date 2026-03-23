#!/bin/bash
# Setup script for Boltz2 standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up Boltz2 standalone environment..."

echo "Installing uv package manager..."
pip install uv

bpt_install_pytorch

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Upgrading triton..."
# torch 2.6.0 pins triton==3.2.0, which has a PY_SSIZE_T_CLEAN bug causing
# runtime failures with conda-forge Python 3.12. Upgrade AFTER all other installs
# to prevent uv from downgrading it back to 3.2.0 via torch's dependency.
uv pip install --upgrade triton

echo "Boltz2 setup complete!"
