#!/bin/bash
set -euo pipefail

echo "Setting up Promoter Calculator standalone environment..."
pip install uv
uv pip install -r requirements.txt
echo "Promoter Calculator setup complete!"
