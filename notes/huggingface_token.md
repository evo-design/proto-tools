# HuggingFace Token Setup

Several tools (ESM3, AlphaGenome, etc.) require a HuggingFace token to download gated model weights. The tool infrastructure passes `HF_TOKEN` from the host environment into isolated tool subprocesses via the `[passthrough]` section in each tool's `env_vars.txt`.

## Checking if a token is set

```bash
# Check environment variable (used by tool passthrough at runtime)
echo $HF_TOKEN

# Check cached token file (used by setup.sh as fallback, written by huggingface-cli login)
cat ~/.cache/huggingface/token
```

Both should contain the same `hf_...` value. `HUGGING_FACE_HUB_TOKEN` is the legacy name — only `HF_TOKEN` is needed.

## Setting the token on a new device

If the user provides a token value, set it permanently:

```bash
echo 'export HF_TOKEN=hf_...' >> ~/.bashrc   # or ~/.zshrc
source ~/.bashrc
```

If no token is available, the user needs to:
1. Create one at https://huggingface.co/settings/tokens
2. Accept the license for any gated models (e.g., ESM3 at https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)
