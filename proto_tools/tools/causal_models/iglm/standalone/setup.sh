#!/bin/bash
# Setup script for IgLM standalone environment
set -euo pipefail
source standalone_helpers.sh

echo "Setting up IgLM standalone environment..."

proto_install_pytorch

# IgLM's weights ship inside the wheel (iglm/trained_models/), so there is nothing to download.
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "IgLM setup complete!"
