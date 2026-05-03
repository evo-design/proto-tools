"""Weights directory resolution for standalone scripts.

Resolves the canonical location where a tool should store its model weights,
following the ``PROTO_HOME`` / ``PROTO_MODEL_CACHE`` / ``PROTO_{TOOL}_WEIGHTS_DIR``
precedence documented in ``notes/storage.md``.
"""

import logging
import os

logger = logging.getLogger(__name__)


def resolve_weights_dir(toolkit: str) -> str | None:
    """Resolve the weights directory for a tool based on PROTO_MODEL_CACHE.

    Precedence:
        1. PROTO_{TOOL}_WEIGHTS_DIR (per-tool override, always wins)
        2. PROTO_MODEL_CACHE:
           - (default): {PROTO_HOME}/proto_model_cache/{toolkit}/ (survives env rebuilds)
           - "/absolute/path": /absolute/path/{toolkit}/  (shared directory)
           - "IN_ENV": {TOOL_VENV_PATH}/model_weight_cache/ (legacy, per-venv)
           - "NONE": {VENV_PATH}/weights/ (pass-through, matches shell helper)

    Args:
        toolkit (str): The tool's directory name (e.g., "fampnn", "protenix").

    Returns:
        str | None: Absolute path string to the weights directory. Returns ``None``
            only in ``NONE`` mode when no venv path is available (caller falls back
            to its own default). Creates the directory if it doesn't exist.

    Raises:
        RuntimeError: If ``PROTO_MODEL_CACHE=IN_ENV`` but neither
            ``TOOL_VENV_PATH`` nor ``VENV_PATH`` is set (cannot resolve a venv-local
            weights directory without a venv).
    """
    # 1. Per-tool override always wins
    override_var = f"PROTO_{toolkit.upper()}_WEIGHTS_DIR"
    override = os.environ.get(override_var)
    if override:
        os.makedirs(override, exist_ok=True)
        return override

    # 2. PROTO_MODEL_CACHE
    mode = os.environ.get("PROTO_MODEL_CACHE", "")

    if mode == "NONE":
        # Pass-through: no managed cache, but match the shell helper's fallback
        # (setup.sh downloads to ${VENV_PATH}/weights in NONE mode)
        venv_path = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv_path:
            path = os.path.join(venv_path, "weights")
            os.makedirs(path, exist_ok=True)
            return path
        return None

    if mode == "IN_ENV":
        # Legacy: weights inside the tool's venv
        venv_path = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv_path:
            path = os.path.join(venv_path, "model_weight_cache")
            os.makedirs(path, exist_ok=True)
            return path
        raise RuntimeError(
            f"resolve_weights_dir({toolkit!r}): PROTO_MODEL_CACHE=IN_ENV but TOOL_VENV_PATH/VENV_PATH unset"
        )

    if mode:
        # Explicit path (absolute or relative)
        cache_dir = mode
    else:
        # Default: PROTO_HOME/proto_model_cache/ directory  # noqa: ERA001 -- path description, not code
        proto_home = os.environ.get("PROTO_HOME", "")
        if not proto_home:
            # Same default as get_proto_home(): ~/.proto/
            proto_home = os.path.join(os.path.expanduser("~"), ".proto")
            logger.warning(
                "PROTO_HOME not set in subprocess environment. Falling back to %s. Set PROTO_HOME to customize.",
                proto_home,
            )
        cache_dir = os.path.join(proto_home, "proto_model_cache")

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, toolkit)
    os.makedirs(path, exist_ok=True)
    return path
