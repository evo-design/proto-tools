"""Modal credential discovery, and which apps are live in the active workspace.

Lives outside ``proto_tools.modal`` because that package builds Modal objects at import
time, which is exactly what fails when Modal is unconfigured. Nothing here imports the
SDK at module scope, so the credential checks still answer in that state.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def _listed_apps(environment: str, client: Any | None) -> set[str]:
    """Names of the deployed apps in ``environment``, from one call.

    This is the request ``modal app list`` makes, and the SDK's own blocking bridge is what runs
    it, so a caller holding an ordinary ``modal.Client`` needs no event loop of its own.
    """
    import modal
    from modal._utils.async_utils import synchronizer
    from modal_proto import api_pb2

    @synchronizer.create_blocking
    async def _list(asking_as: Any) -> Any:
        return await asking_as.stub.AppList(api_pb2.AppListRequest(environment_name=environment))

    response = _list(client if client is not None else modal.Client.from_env())
    return {app.description for app in response.apps if app.state == api_pb2.APP_STATE_DEPLOYED}


def _hydrated_apps(environment: str, client: Any | None) -> set[str]:
    """The same answer, asked one app at a time.

    Kept as the fallback for :func:`deployed_apps` because the fast path reaches into SDK
    internals: if a Modal upgrade moves them, this still answers, only slowly.
    """
    import modal

    from proto_tools.modal.manifest import APP_BUCKETS

    live = set()
    for app_name, services in APP_BUCKETS.items():
        try:
            modal.Cls.from_name(app_name, services[0], environment_name=environment, client=client).hydrate()
            live.add(app_name)
        except Exception:  # noqa: S112 — an unreachable app is "not deployed", not an error
            continue
    return live


def deployed_apps(environment: str | None = None, client: Any | None = None) -> set[str]:
    """Apps that currently resolve in the Modal environment a dispatch would reach.

    One listing call, no containers started. Failures read as "not deployed" rather than
    propagating.

    Asking per app instead costs a round trip each, which is seconds across the catalogue and
    grows with every tool added. Callers use this to describe a whole catalogue at once, so the
    question is "what is deployed here" rather than "is this one app deployed", and Modal answers
    that in a single request.

    The environment is named rather than inherited, and must be: a dispatch resolves
    ``proto-env`` while an unconfigured Modal profile resolves the workspace default, so asking
    ambiently reports on a different environment than the one a call would actually reach.

    Args:
        environment (str | None): Modal environment to look in. ``None`` resolves the default.
        client (Any | None): Modal client to ask as. ``None`` uses the process's own credentials,
            which is what a local caller wants; a server answering for someone else passes theirs,
            or it would report its own deployments as though they were the caller's.

    Returns:
        set[str]: App names that resolve.
    """
    from proto_tools.modal.app import resolve_environment

    environment = resolve_environment(environment)
    try:
        return _listed_apps(environment, client)
    except Exception:
        logger.warning("could not list Modal apps; falling back to per-app lookup", exc_info=True)
        return _hydrated_apps(environment, client)
