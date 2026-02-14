#!/bin/bash
# Setup script for Evo1 standalone environment
set -euo pipefail

echo "Setting up Evo1 standalone environment..."

echo "Installing uv package manager..."
pip install uv

echo "Installing torch..."
# Pin torch to 2.7.1: flash-attn pre-built wheels are ABI-sensitive and only
# work with the exact torch version they were compiled against.
# Update both this pin and flash-attn in requirements.txt together.
uv pip install torch==2.7.1 --torch-backend=auto

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt --torch-backend=auto --no-build-isolation-package flash-attn

echo "Patching stripedhyena tokenizer for numpy >=2.0 compatibility..."
# stripedhyena 0.2.2 uses np.fromstring (removed in numpy 2.0).
# Patch to np.frombuffer until upstream fixes it.
TOKENIZER_PY=$(python -c "import stripedhyena.tokenizer; print(stripedhyena.tokenizer.__file__)")
if grep -q 'np.fromstring' "$TOKENIZER_PY" 2>/dev/null; then
    sed -i 's/np\.fromstring(text, dtype=np\.uint8)/np.frombuffer(text.encode(), dtype=np.uint8)/g' "$TOKENIZER_PY"
    # Clear stale .pyc so Python picks up the patched source
    TOKENIZER_DIR=$(dirname "$TOKENIZER_PY")
    rm -f "$TOKENIZER_DIR/__pycache__/tokenizer"*.pyc 2>/dev/null || true
    echo "  Patched: $TOKENIZER_PY"
else
    echo "  Already patched or not needed"
fi

echo "If installation fails, follow upstream setup guide:"
echo "  - https://github.com/evo-design/evo"

echo "Evo1 setup complete!"
