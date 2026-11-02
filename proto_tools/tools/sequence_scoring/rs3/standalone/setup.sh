#!/bin/bash
set -euo pipefail

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "ERROR: rs3 setup: not supported on macOS arm64 (lightgbm 3.3.5 has no macOS arm64 wheel)" >&2
    exit 1
fi

echo "Setting up Rule Set 3 standalone environment..."
uv pip install -r requirements.txt
echo "Rule Set 3 setup complete!"
