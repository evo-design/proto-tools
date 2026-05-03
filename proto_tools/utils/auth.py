"""proto_tools/utils/auth.py.

Authentication helpers for gated model providers.
"""

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
    token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    if token:
        return token

    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.isfile(token_file):
        try:
            with open(token_file) as f:
                token = f.read().strip()
        except Exception:  # noqa: S110 -- best-effort token file read
            pass
    if token:
        return token

    git_creds = os.path.expanduser("~/.git-credentials")
    if os.path.isfile(git_creds):
        try:
            with open(git_creds) as f:
                for line in f:
                    m = re.search(r"https?://[^:]+:(hf_[^@]+)@huggingface\.co", line)
                    if m:
                        return m.group(1)
        except Exception:  # noqa: S110 -- best-effort credential file read
            pass

    return None


def require_hf_token(tool_display_name: str, repo_url: str = "") -> None:
    """Check that a HuggingFace token is available, raising a clear error if not.

    Call this at the top of tool functions that use gated HuggingFace models.
    Fails fast with an actionable error message before dispatching to the
    standalone subprocess.

    Args:
        tool_display_name (str): Human-readable tool name for the error message (e.g., "ESM3").
        repo_url (str): HuggingFace repo URL where the user can accept the license.

    Raises:
        OSError: If no HuggingFace token is found.
    """
    if resolve_hf_token():
        return

    repo = repo_url or "the model's HuggingFace page"
    raise OSError(
        f"{tool_display_name}: HF_TOKEN unset and no cached token; accept license at {repo} "
        f"and run 'export HF_TOKEN=hf_...' (or 'hf auth login')"
    )
