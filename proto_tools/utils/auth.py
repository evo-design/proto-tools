"""proto_tools/utils/auth.py

Authentication helpers for gated model providers."""
from __future__ import annotations

import os
import re


def resolve_hf_token() -> str | None:
    """Resolve a HuggingFace token from all known sources.

    Checks:
      1. ``HF_TOKEN`` environment variable
      2. ``HUGGING_FACE_HUB_TOKEN`` environment variable
      3. ``~/.cache/huggingface/token`` file (written by ``hf auth login``)
      4. ``~/.git-credentials`` file (written by ``hf auth login --add-to-git-credential``)

    Returns the token string, or ``None`` if no token is found.
    """
    token = os.environ.get("HF_TOKEN", "") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    )
    if token:
        return token

    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_file):
        try:
            with open(token_file) as f:
                token = f.read().strip()
        except Exception:
            pass
    if token:
        return token

    git_creds = os.path.expanduser("~/.git-credentials")
    if os.path.isfile(git_creds):
        try:
            with open(git_creds) as f:
                for line in f:
                    m = re.search(
                        r"https?://[^:]+:(hf_[^@]+)@huggingface\.co", line
                    )
                    if m:
                        return m.group(1)
        except Exception:
            pass

    return None


def require_hf_token(tool_name: str, repo_url: str = "") -> None:
    """Check that a HuggingFace token is available, raising a clear error if not.

    Call this at the top of tool functions that use gated HuggingFace models.
    Fails fast with an actionable error message before dispatching to the
    standalone subprocess.

    Args:
        tool_name (str): Human-readable tool name for the error message (e.g., "ESM3").
        repo_url (str): HuggingFace repo URL where the user can accept the license.

    Raises:
        EnvironmentError: If no HuggingFace token is found.
    """
    if resolve_hf_token():
        return

    msg = (
        f"{tool_name} requires a HuggingFace token to download gated model weights.\n"
        "\n"
        "To fix this:\n"
        "  1. Create a HuggingFace account at https://huggingface.co\n"
    )
    if repo_url:
        msg += f"  2. Accept the model license at: {repo_url}\n"
    else:
        msg += "  2. Accept the model license on its HuggingFace page\n"
    msg += (
        "  3. Create an access token at: https://huggingface.co/settings/tokens\n"
        "  4. Set it in your environment: export HF_TOKEN=hf_...\n"
    )
    raise EnvironmentError(msg)
