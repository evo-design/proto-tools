"""What state this machine's Modal setup is in: credentials, and which apps are live.

Imports no ``modal`` at module scope, so the credential checks below still answer in an
environment where importing the SDK's configured app would itself fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Both must be set for the SDK to authenticate from the environment.
TOKEN_VARS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET")


def config_path() -> Path:
    """Return the config file Modal reads, honoring ``MODAL_CONFIG_PATH``."""
    override = os.environ.get("MODAL_CONFIG_PATH")
    return Path(override) if override else Path.home() / ".modal.toml"


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


def credentials_checked() -> dict[str, Any]:
    """Report which credential sources are present, by presence only and never by value."""
    return {
        "MODAL_TOKEN_ID": bool(os.environ.get("MODAL_TOKEN_ID")),
        "MODAL_TOKEN_SECRET": bool(os.environ.get("MODAL_TOKEN_SECRET")),
        "MODAL_PROFILE": bool(os.environ.get("MODAL_PROFILE")),
        "config_file": str(config_path()),
        "config_file_state": config_state(),
    }


def auth_mechanism() -> str | None:
    """Name the source the SDK will authenticate from, or ``None`` when there is none.

    Environment variables take precedence over the config file, so they are reported
    first when both are available.
    """
    if all(os.environ.get(var) for var in TOKEN_VARS):
        return "/".join(TOKEN_VARS)
    if config_state() == "readable":
        return str(config_path())
    return None


def deployed_apps() -> set[str]:
    """Apps that currently resolve in the active Modal workspace.

    One hydrate per app, no containers started. Failures read as "not deployed"
    rather than propagating.
    """
    import modal

    from proto_tools.modal.manifest import APP_BUCKETS

    live = set()
    for app_name, services in APP_BUCKETS.items():
        try:
            modal.Cls.from_name(app_name, services[0]).hydrate()
            live.add(app_name)
        except Exception:  # noqa: S112 — an unreachable app is "not deployed", not an error
            continue
    return live
