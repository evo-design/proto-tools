#!/bin/bash
# Install miRanda 3.3a from bioconda.
set -euo pipefail
source standalone_helpers.sh

echo "Installing miRanda 3.3a from bioconda..."
"$MAMBA_BIN" install -p "$VENV_PATH" -c bioconda -c conda-forge -y "miranda=3.3a"

echo "miRanda setup complete!"
