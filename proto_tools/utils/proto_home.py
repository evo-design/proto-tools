"""Find the storage directory for model weights, tool environments, and micromamba.

All install modes use ``PROTO_HOME`` to locate persistent data. This module
imports only ``os``, ``sys``, and ``pathlib``, so importing it does not load
tool dependencies or set up storage.

Layout under PROTO_HOME::

    PROTO_HOME/                   (default: ~/.proto/)
    ├── proto_model_cache/        model weights, one folder per toolkit
    │   ├── {toolkit}/            huggingface/ (HF_HOME), torch/ (TORCH_HOME),
    │   │                         jax_cache/ (JAX), weights/ (resolve_weights_dir)
    │   └── databases/            provisioned MMseqs2 databases
    ├── proto_tool_envs/          micromamba-managed tool venvs
    ├── uv_cache/                 uv package download cache (UV_CACHE_DIR)
    ├── uv_cache.prune/           uv cache prune stamp, lock, and log
    └── .micromamba/              micromamba binary + package cache
"""

import os
import sys
from pathlib import Path


def get_proto_home() -> Path:
    """Return the directory where proto-tools stores persistent data.

    Use the first available location below.

    1. ``PROTO_HOME`` environment variable (if set).
    2. ``~/.proto/`` (default for all install modes).

    The default is the same for editable and non-editable installs.
    Data stays separate from the source checkout. To choose another location,
    such as storage on an HPC cluster, set ``PROTO_HOME`` in your shell profile.
    """
    # Expand and resolve a custom path before returning it.
    env_val = os.environ.get("PROTO_HOME")
    if env_val:
        return Path(env_val).expanduser().resolve()

    # Use the same default for every install mode.
    return Path.home() / ".proto"


_DOCS_URL = "https://github.com/evo-design/proto-tools/blob/main/notes/storage.md"


def show_first_run_notice() -> None:
    """Show the storage locations once when either location uses its default.

    Write a notice to stderr if either ``PROTO_HOME`` or ``PROTO_MODEL_CACHE``
    is unset. The notice shows where data is stored and how to change it.

    The notice is shown once per ``PROTO_HOME`` location (a
    ``.initialized`` sentinel file is created after the first notice).
    """
    proto_home = get_proto_home()
    sentinel = proto_home / ".initialized"

    # Skip the notice if this storage directory has already been initialized.
    if sentinel.exists():
        return

    proto_home_set = bool(os.environ.get("PROTO_HOME"))
    model_cache_set = bool(os.environ.get("PROTO_MODEL_CACHE"))

    if proto_home_set and model_cache_set:
        # Both locations are configured, so mark initialization without showing a notice.
        proto_home.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
        return

    # Create the storage directory before showing the default locations.
    proto_home.mkdir(parents=True, exist_ok=True)

    tool_envs_path = str(proto_home / "proto_tool_envs")
    weights_path = os.environ["PROTO_MODEL_CACHE"] if model_cache_set else str(proto_home / "proto_model_cache")

    # Build the notice and include instructions for each location that uses its default.
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

    # Size the box to fit the title and the longest line.
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

    # Record that the notice was shown for this storage directory.
    sentinel.touch()
