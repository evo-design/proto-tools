"""Modal credential discovery, and which apps are live in the active workspace.

Lives outside ``proto_tools.modal`` because that package builds Modal objects at import
time, which is exactly what fails when Modal is unconfigured. Nothing here imports the
SDK at module scope, so the credential checks still answer in that state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Both must be set for the SDK to authenticate from the environment.
TOKEN_VARS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")


def config_path() -> Path:
    """Return the config file Modal reads, resolved the way the SDK resolves it.

    ``os.path.expanduser`` rather than ``Path.home()``: the latter raises when ``HOME`` is
    unset and the uid has no passwd entry, an ordinary state inside a container.
    """
    return Path(os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml"))


def config_state() -> str:
    """Report the config file as ``readable``, ``absent``, or ``unreadable``."""
    try:
        with config_path().open("rb"):
            return "readable"
    except FileNotFoundError:
        return "absent"
    except OSError:
        # Present but closed to this process, or inside a directory it cannot traverse.
        # This is the container case, and it is invisible to a plain existence check.
        return "unreadable"


def _variable_state(name: str) -> str:
    """Report an environment variable as ``set``, ``empty``, or ``unset``.

    Modal tests membership rather than truthiness, so a variable set to the empty string
    still takes precedence over the config file — and then fails to authenticate.
    """
    if name not in os.environ:
        return "unset"
    return "set" if os.environ[name] else "empty"


def credentials_checked() -> dict[str, Any]:
    """Report which credential sources are present, by presence only and never by value."""
    return {
        "MODAL_TOKEN_ID": _variable_state("MODAL_TOKEN_ID"),
        "MODAL_TOKEN_SECRET": _variable_state("MODAL_TOKEN_SECRET"),
        "MODAL_PROFILE": _variable_state("MODAL_PROFILE"),
        "config_file": str(config_path()),
        "config_file_state": config_state(),
    }


def auth_mechanism() -> str | None:
    """Name the source the SDK will authenticate from, or ``None`` when there is none.

    Environment variables take precedence over the config file, so they are reported first
    when both are available. An empty variable still counts: Modal reads it and fails,
    rather than falling back to the file.
    """
    if all(var in os.environ for var in TOKEN_VARS):
        return "/".join(TOKEN_VARS)
    if config_state() == "readable":
        return str(config_path())
    return None


def deployed_apps() -> set[str]:
    """Apps that currently resolve in the Modal environment this session dispatches into.

    One hydrate per app, no containers started. Failures read as "not deployed"
    rather than propagating.

    The environment is named rather than inherited, and must be: a dispatch resolves
    ``proto-env`` while an unconfigured Modal profile resolves the workspace default, so asking
    ambiently reports on a different environment than the one a call would actually reach.
    """
    import modal

    from proto_tools.modal.app import resolve_environment
    from proto_tools.modal.manifest import APP_BUCKETS

    environment = resolve_environment()
    live = set()
    for app_name, services in APP_BUCKETS.items():
        try:
            modal.Cls.from_name(app_name, services[0], environment_name=environment).hydrate()
            live.add(app_name)
        except Exception:  # noqa: S112 — an unreachable app is "not deployed", not an error
            continue
    return live
