#!/bin/bash
set -euo pipefail

echo "Installing uv package manager..."
pip install uv

echo "Installing AlphaFold3 worker dependencies..."
# Only numpy and biopython needed for structure extraction
# MSA generation (ColabFold) runs in main process, not subprocess
uv pip install -r requirements.txt
