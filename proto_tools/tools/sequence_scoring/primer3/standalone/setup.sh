#!/bin/bash
set -euo pipefail

echo "Setting up Primer3 standalone environment..."
pip install uv
uv pip install -r requirements.txt
echo "Primer3 setup complete!"
