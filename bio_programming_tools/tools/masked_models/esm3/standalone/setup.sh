#!/bin/bash
# Setup script for ESM3 standalone environment
set -euo pipefail

# ── Gated repo access check ──────────────────────────────────────────────────
# ESM3 weights are hosted in a gated HuggingFace repo that requires
# authentication.  Catch this early so the user gets a clear message
# instead of a cryptic 401 buried in a Python traceback.
REPO_ID="EvolutionaryScale/esm3-sm-open-v1"
REPO_URL="https://huggingface.co/${REPO_ID}"

HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.git-credentials" ]; then
    HF_TOKEN="$(grep -oP 'https?://[^:]+:\Khf_[^@]+(?=@huggingface\.co)' "$HOME/.git-credentials" | head -1)"
fi

if [ -n "$HF_TOKEN" ]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        "https://huggingface.co/${REPO_ID}/resolve/main/config.json")
else
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://huggingface.co/${REPO_ID}/resolve/main/config.json")
fi

if [ "$HTTP_CODE" != "200" ]; then
    echo ""
    echo "============================================================"
    echo "ERROR: Cannot access HuggingFace repo '${REPO_ID}'"
    echo "============================================================"
    echo ""
    if [ -z "$HF_TOKEN" ]; then
        echo "No HuggingFace token found. This is a gated model that"
        echo "requires authentication."
        echo ""
        echo "To fix this:"
        echo "  1. Create a HuggingFace account at https://huggingface.co"
        echo "  2. Accept the model license at:"
        echo "     ${REPO_URL}"
        echo "  3. Create an access token at:"
        echo "     https://huggingface.co/settings/tokens"
        echo "  4. Set the token in your environment:"
        echo "     export HF_TOKEN=hf_..."
        echo "     Or log in with: huggingface-cli login"
    else
        echo "A HuggingFace token was found but access was denied (HTTP ${HTTP_CODE})."
        echo ""
        echo "To fix this:"
        echo "  1. Visit: ${REPO_URL}"
        echo "  2. Accept the license/terms for this model"
        echo "  3. Re-run the setup"
    fi
    echo ""
    echo "============================================================"
    exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

echo "Setting up ESM3 standalone environment..."

echo "Installing uv package manager..."
pip install uv

# Install hardware-aware PyTorch version (from centralized detection)
echo "Installing PyTorch: ${RECOMMENDED_TORCH_SPEC:-torch} (platform: ${DETECTED_COMPUTE_PLATFORM:-unknown})"
uv pip install "${RECOMMENDED_TORCH_SPEC:-torch}" --extra-index-url "${RECOMMENDED_TORCH_INDEX}"

echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

echo "ESM3 setup complete!"
