"""proto_tools/utils/proto_home.py.

Unified PROTO_HOME resolution for all persistent data.

Provides a single function to determine where model weights, tool
environments, and micromamba live.  Lightweight: only ``os``, ``sys``,
and ``pathlib`` are imported so it can be used anywhere without side effects.

``PROTO_HOME`` is the single source of truth for all install modes.

Layout under PROTO_HOME::

    PROTO_HOME/                   (default: ~/.proto/)
    ├── proto_model_cache/        model weights (HF_HOME, TORCH_HOME, resolve_weights_dir)
    ├── proto_tool_envs/          micromamba-managed tool venvs
    ├── uv_cache/                 uv package download cache (UV_CACHE_DIR)
    ├── pip_cache/                pip HTTP cache (PIP_CACHE_DIR)
    └── .micromamba/              micromamba binary + package cache
"""

import os
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_proto_home() -> Path:
    """Return the PROTO_HOME root directory.

    Resolution:

    1. ``PROTO_HOME`` environment variable (if set).
    2. ``~/.proto/`` (default for all install modes).

    The default is the same for editable and non-editable installs.
    This keeps data separate from source code and gives every user
    a consistent, predictable location.  To use a custom location
    (e.g. on HPC), set ``PROTO_HOME`` in your shell profile.

    The result is cached for the lifetime of the process.  Call
    ``get_proto_home.cache_clear()`` if you need to re-resolve
    (e.g. in tests after monkeypatching ``PROTO_HOME``).
    """
    env_val = os.environ.get("PROTO_HOME")
    if env_val:
        return Path(env_val).expanduser().resolve()

    return Path.home() / ".proto"


_DOCS_URL = "https://github.com/evo-design/proto-tools/blob/main/notes/storage.md"


def show_first_run_notice() -> None:
    """Print a one-time notice when using default storage locations.

    Shows a notice to stderr if either ``PROTO_HOME`` or
    ``PROTO_MODEL_CACHE`` is not explicitly set, so the user knows
    where data is being stored and how to customize it.

    The notice is shown once per ``PROTO_HOME`` location (a
    ``.initialized`` sentinel file is created after the first notice).
    """
    proto_home = get_proto_home()
    sentinel = proto_home / ".initialized"

    if sentinel.exists():
        return

    proto_home_set = bool(os.environ.get("PROTO_HOME"))
    model_cache_set = bool(os.environ.get("PROTO_MODEL_CACHE"))

    if proto_home_set and model_cache_set:
        # User configured everything; create sentinel silently
        proto_home.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        return

    # At least one default location in use, show notice
    proto_home.mkdir(parents=True, exist_ok=True)

    tool_envs_path = str(proto_home / "proto_tool_envs")
    weights_path = os.environ["PROTO_MODEL_CACHE"] if model_cache_set else str(proto_home / "proto_model_cache")

    # Collect content lines first, then size the box to fit
    content = []
    content.append("  Tool environments: " + tool_envs_path)
    content.append("  Model weights:     " + weights_path)
    content.append("")
    content.append("  To customize, add to your shell profile (~/.bashrc):")
    if not proto_home_set:
        content.append("    export PROTO_HOME=/your/path")
    if not model_cache_set:
        content.append("    export PROTO_MODEL_CACHE=/your/weights/path")
    content.append("")
    content.append("  Docs: " + _DOCS_URL)

    title = "proto-tools: first-run setup"
    w = max(len(title) + 4, max(len(line) for line in content) + 2)

    border = "═" * w
    lines = [
        "",
        f"  ╔{border}╗",
        f"  ║{title:^{w}}║",
        f"  ║{'':{w}}║",
        *[f"  ║{line:<{w}}║" for line in content],
        f"  ╚{border}╝",
        "",
    ]

    print("\n".join(lines), file=sys.stderr)

    sentinel.touch()
