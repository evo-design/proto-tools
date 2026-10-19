#!/bin/bash
# Setup script for the unified mmseqs2 standalone environment.
# Hosts all four tools: mmseqs2-search-proteins, -search-genomes, -clustering,
# and -homology-search. Ships the GPU-capable MMseqs2 binary (a strict superset
# of the CPU build) plus colabfold for the homology-search MSA pipeline.
set -euo pipefail
source standalone_helpers.sh

echo "Setting up mmseqs2 toolkit standalone environment..."

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "Installing MMseqs2 binary (GPU-capable; runs CPU calls without --gpu)..."
proto_install_binary mmseqs2

echo "mmseqs2 toolkit setup complete!"
