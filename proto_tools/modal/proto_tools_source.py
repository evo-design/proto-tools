"""Resolution of the proto-tools source tree that images are built from."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set by images to point at the tree they already carry, and usable directly to
# build from a checkout other than the installed one.
PATH_ENV = "PROTO_MODAL_PROTO_TOOLS"

# proto_tools/modal/proto_tools_source.py -> the repository root.
_TREE = Path(__file__).resolve().parents[2]


class ProtoToolsUnavailableError(RuntimeError):
    """Raised when no proto-tools source tree can be resolved."""


def _holds_package(path: Path) -> bool:
    """Report whether ``path`` contains an importable ``proto_tools`` package."""
    return (path / "proto_tools").is_dir()


def _is_checkout(path: Path) -> bool:
    """Report whether ``path`` is a source checkout, which alone can supply dependencies."""
    return _holds_package(path) and (path / "pyproject.toml").is_file()


def checkout_or_none() -> Path | None:
    """Return the source checkout to build from, or ``None`` for an installed package.

    This is what picks the image recipe: a checkout supplies both the source and
    ``pyproject.toml``; an installed package supplies source plus dependency metadata.

    Returns:
        Path | None: The checkout root, or ``None`` when running from an install.
    """
    override = os.environ.get(PATH_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        return path if _is_checkout(path) else None
    return _TREE if _is_checkout(_TREE) else None


def resolve_proto_tools() -> Path:
    """Return the proto-tools source tree to build images from.

    Resolution order: an explicit path override, then the tree this module lives in.

    Returns:
        Path: A directory containing ``pyproject.toml`` and ``proto_tools/``.

    Raises:
        ProtoToolsUnavailableError: If no tree can be resolved.
    """
    override = os.environ.get(PATH_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not _holds_package(path):
            raise ProtoToolsUnavailableError(f"{PATH_ENV}={path} holds no proto_tools package")
        _announce(f"proto-tools: {path} (from {PATH_ENV})")
        return path

    if _is_checkout(_TREE):
        _announce(f"proto-tools: {_TREE} (checkout — local edits are included in builds)")
        return _TREE

    if _holds_package(_TREE):
        _announce(f"proto-tools: {_TREE} (installed package — shipped catalogue only)")
        return _TREE

    raise ProtoToolsUnavailableError(
        f"No proto_tools package found at {_TREE}. Install proto-tools, or set {PATH_ENV}=/path/to/proto-tools."
    )


def _announce(message: str) -> None:
    """Report the resolved source once, so a build is never ambiguous about it.

    Written to stderr: a developer must be able to see which tree a build used,
    but stdout carries the MCP server's protocol.
    """
    if getattr(_announce, "_said", None) == message:
        return
    _announce._said = message  # type: ignore[attr-defined]
    print(message, file=sys.stderr)  # noqa: T201
