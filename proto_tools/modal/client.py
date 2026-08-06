"""Dispatch tool calls to proto-tools apps deployed in a Modal workspace."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
import uuid
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from proto_tools.modal.app import resolve_environment
from proto_tools.modal.progress import open_progress_queue, stream_modal_progress
from proto_tools.utils.logging_config import verbose_level_from_env
from proto_tools.utils.progress import (
    has_active_progress_bar,
    remote_connecting_status,
    remote_progress,
    set_substatus,
)

if TYPE_CHECKING:
    from proto_tools.utils.base_config import BaseConfig
    from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput

_client_logger = logging.getLogger(__name__)

# Device strings that mean "run this somewhere else". Translated to the
# physical device of whichever container the service actually runs on.


# Where a user goes to install, deploy, or set up Modal. Every dispatch error
# ends with this so the fix is always one link away.
PROTO_TOOLS_REPO = "https://github.com/proto-bio/proto-tools"


class ModalDispatchError(RuntimeError):
    """Base for ``device='modal'`` failures a user must act on before a call can run."""


class ToolNotShippedError(ModalDispatchError, KeyError):
    """Raised when a tool has no deployment at all.

    Subclasses ``KeyError`` for back-compatibility with callers that caught the
    plain lookup failure ``resolve_tool`` used to raise.
    """

    def __init__(self, tool_key: str, shipped_count: int) -> None:
        """Build the error, distinguishing "not shipped" from "not deployed"."""
        super().__init__(
            f"Tool {tool_key!r} is not available on device='modal': this repository ships no "
            f"deployment for it (it ships {shipped_count} tools). See the catalog with "
            f"`proto-tools deploy --list`.\n"
            f"More: {PROTO_TOOLS_REPO}"
        )
        self.tool_key = tool_key


class ModalCredentialsError(ModalDispatchError):
    """Raised when Modal credentials are missing or rejected."""

    def __init__(self, detail: str) -> None:
        """Build the error, naming the command that configures credentials."""
        super().__init__(
            f"device='modal' needs Modal credentials, but {detail}.\n"
            f"Authenticate with:  modal token new\n"
            f"Then confirm with:  modal profile current\n"
            f"Setup guide: {PROTO_TOOLS_REPO}"
        )


class ToolNotDeployedError(ModalDispatchError):
    """Raised when a tool is shipped but its app is not deployed in the active workspace."""

    def __init__(self, tool_key: str, app_name: str) -> None:
        """Build the error, naming the exact deploy command that fixes it."""
        slug = app_name.removeprefix("proto-tools-")
        super().__init__(
            f"Tool {tool_key!r} needs Modal app {app_name!r}, which is not deployed "
            f"in the active workspace. Deploy it with:\n\n"
            f"    proto-tools deploy --apps {slug} --env <name>\n\n"
            f"More: {PROTO_TOOLS_REPO}"
        )
        self.tool_key = tool_key
        self.app_name = app_name


class ModalEnvironmentNotFoundError(ModalDispatchError):
    """Raised when the Modal environment a call names has not been created.

    Modal reports a missing environment and a missing app identically, so without this a
    workspace that never ran the one-time setup is told its tools are undeployed, and sent to
    deploy things that cannot land anywhere.
    """

    def __init__(self, environment: str, available: list[str] | None = None) -> None:
        """Build the error, naming the command that creates the environment."""
        known = f" This workspace has: {', '.join(sorted(available))}." if available else ""
        super().__init__(
            f"Modal environment {environment!r} has not been created in this workspace.{known}\n"
            f"Nothing can be deployed to it or run in it until it exists. Create it with:\n\n"
            f"    proto-tools deploy --create-env --env {environment}\n\n"
            f"More: {PROTO_TOOLS_REPO}"
        )
        self.environment = environment
        self.available = available


def available_tools() -> dict[str, tuple[str, str]]:
    """Return ``{tool_key: (service_class_name, method_name)}`` for every shipped tool.

    Reads the generated :data:`proto_tools.modal.tool_map.TOOL_MAP` rather than
    importing the service modules — those construct Modal images and need the
    Modal image definitions, which a caller should not require.
    Regenerate with ``python scripts/generate_tool_map.py``.
    """
    from proto_tools.modal.tool_map import TOOL_MAP

    return {key: (entry.service, entry.method) for key, entry in TOOL_MAP.items()}


def resolve_tool(tool_key: str) -> tuple[str, str, str]:
    """Return ``(app_name, service_class_name, method_name)`` for ``tool_key``.

    Raises:
        ToolNotShippedError: If the tool has no deployment.
    """
    from proto_tools.modal.tool_map import TOOL_MAP

    entry = TOOL_MAP.get(tool_key)
    if entry is None:
        raise ToolNotShippedError(tool_key, len(TOOL_MAP))
    return entry.app, entry.service, entry.method


def _require_modal_credentials(client: Any | None = None) -> None:
    """Fail with an actionable error when no Modal credentials are configured.

    A presence check only — token *validity* needs a network round-trip and is
    surfaced by translating Modal's ``AuthError`` at call time. This catches the
    common "never ran ``modal token new``" case up front, before any work.

    Args:
        client (Any | None): A caller-supplied Modal client. One carries its own
            credentials, so the process is not consulted at all.

    Raises:
        ModalCredentialsError: If neither token env vars nor ``~/.modal.toml`` exist.
    """
    from pathlib import Path

    if client is not None:
        return
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return
    if (Path.home() / ".modal.toml").is_file():
        return
    raise ModalCredentialsError("none are configured (no MODAL_TOKEN_ID/SECRET, no ~/.modal.toml)")


def _resolve_device(config: BaseConfig | None, service_class: str) -> BaseConfig | None:
    """Replace a logical device with the physical device of the target container.

    proto-tools' ``BaseConfig`` defaults to ``device="cpu"``, and ``"proto"``
    is not a device any container can bind. Either would run a GPU model on
    the CPU of a GPU container, so the caller's intent is translated here
    rather than silently honoured.
    """
    from proto_tools.modal.tool_map import TOOL_MAP
    from proto_tools.utils.device import is_remote_device

    if config is None:
        return None
    on_gpu = any(e.service == service_class and e.gpu for e in TOOL_MAP.values())
    physical = "cuda" if on_gpu else "cpu"
    if is_remote_device(config.device) or config.device == "cpu":
        config = config.model_copy(update={"device": physical})
    return config


def _bound_method(
    app_name: str,
    service_class: str,
    method_name: str,
    tool_key: str,
    scaledown_window: int | None = None,
    *,
    environment: str | None = None,
    client: Any | None = None,
) -> Any:
    """Resolve a deployed service method, translating a missing app into a clear error.

    ``scaledown_window`` overrides the deployed value for this call via
    ``Cls.with_options``. Modal autoscales each option set independently, so
    an override gets its own container pool — pass the same value on every
    call in a session, or none at all. Alternating fragments the pool and
    produces cold starts on both sides.

    ``environment`` and ``client`` are handed to Modal rather than left to the process, so a
    call names its own target instead of inheriting whatever the process last set. ``None``
    for either returns that half to Modal's own resolution, which is what a local caller wants.
    """
    import modal

    try:
        service_cls = modal.Cls.from_name(app_name, service_class, environment_name=environment, client=client)
        if scaledown_window is not None:
            service_cls = service_cls.with_options(scaledown_window=scaledown_window)
        service_cls.hydrate()
    except modal.exception.NotFoundError as exc:
        raise _missing_lookup_error(tool_key, app_name, environment, client) from exc
    except modal.exception.AuthError as exc:
        # Credentials were present (checked up front) but Modal rejected them.
        raise ModalCredentialsError("Modal rejected them (expired or invalid)") from exc
    return getattr(service_cls(), method_name)


def _missing_lookup_error(
    tool_key: str, app_name: str, environment: str | None, client: Any | None
) -> ModalDispatchError:
    """Decide whether a Modal lookup missed because of the environment or the app.

    Modal raises the same error for both, and the two need opposite actions: create an
    environment, or deploy into one that already exists. Only asked once a lookup has already
    failed, and a workspace that cannot be listed leaves the app answer, which is the safe
    assumption when the environment is almost always the one this machine already uses.
    """
    from proto_tools.modal.app import environment_names

    if environment is None:
        return ToolNotDeployedError(tool_key, app_name)
    names = environment_names(client)
    if names is not None and environment not in names:
        return ModalEnvironmentNotFoundError(environment, names)
    return ToolNotDeployedError(tool_key, app_name)


class DeploymentDriftWarning(UserWarning):
    """Local proto-tools disagrees with what is deployed."""


_DRIFT_WARNED: set[tuple[str, str, str]] = set()


def _warn_once_on_drift(
    tool_key: str, service_class: str, *, environment: str | None = None, client: Any | None = None
) -> None:
    """Emit drift warnings for a tool the first time it is dispatched.

    Said once per tool per environment, since the same tool can be aligned in one deployment and
    stale in another. Tracked in a set rather than cached on the arguments, because ``client``
    differs per caller and a cache key holding one would keep it alive for the process's lifetime.

    Reading the deployed manifest is a client-side volume read — no container starts — and any
    failure is swallowed rather than blocking work.
    """
    from proto_tools.modal.fingerprint import drift_warnings

    seen = (tool_key, service_class, environment or "")
    if seen in _DRIFT_WARNED:
        return
    _DRIFT_WARNED.add(seen)
    for message in drift_warnings(tool_key, service_class, environment=environment, client=client):
        warnings.warn(message, DeploymentDriftWarning, stacklevel=2)


class LongRunningToolWarning(UserWarning):
    """A dispatched tool can occupy a container for hours."""


@functools.cache
def _warn_once_if_long_running(tool_key: str, service_class: str) -> None:
    """Warn before the first dispatch of a batch-tier tool, which can run for most of a day.

    A design pipeline gives no output until it finishes, so without this the caller cannot tell a
    run that will take twelve hours from one that has hung — and either way they are paying for GPU
    the whole time. Said once per tool per session, like the drift warning, since a caller running a
    sweep already knows after the first.
    """
    from proto_tools.modal.manifest import SERVICE_MODAL_TIMEOUTS, runs_for_hours

    if not runs_for_hours(service_class):
        return
    hours = SERVICE_MODAL_TIMEOUTS[service_class] / 3600
    warnings.warn(
        f"{tool_key} is a long-running pipeline: one call occupies a container until it finishes, "
        f"for up to {hours:.0f} hours of billed compute, and returns nothing before then. It is "
        f"deployed without retries, so a failure costs one run rather than several — but the run it "
        f"does cost is charged to your own Modal account whether or not the result is usable. "
        f"Consider running it locally first with device='cpu'/'cuda' on a small input.",
        LongRunningToolWarning,
        stacklevel=2,
    )


def _raise_if_asset_missing(tool_key: str, result: dict[str, Any]) -> None:
    """Rebuild :class:`MissingAssetError` from a container that reported an unprovisioned asset.

    The container cannot send an exception, so it flags the condition instead. Rebuilding it here
    is what lets a caller tell "this machine was never given the weights" from a genuine failure —
    the distinction the type exists for, and the one the test layer turns into a skip.

    Left to validation, the signal is destroyed: the payload has no item field, so the caller is
    told its result schema is malformed and the real cause survives only inside a discarded string.

    Args:
        tool_key (str): Registry key, used to name the toolkit when the container did not.
        result (dict[str, Any]): Raw payload returned by the container.

    Raises:
        MissingAssetError: If the container reported a missing asset.
    """
    if not result.get("missing_asset"):
        return
    from proto_tools.utils.tool_io import MissingAssetError

    # A container built before these fields existed sends only the flag and a formatted string, so
    # fall back rather than lose the signal against a deployment that has not been rebuilt.
    toolkit = result.get("missing_asset_toolkit") or tool_key.split("-")[0]
    kind = result.get("missing_asset_kind") or "asset"
    details = "; ".join(result.get("errors") or [])
    raise MissingAssetError(toolkit, kind, details)


def _validated_output(tool_key: str, result: dict[str, Any]) -> BaseToolOutput:
    """Validate a raw result dict against the tool's declared output model."""
    from pydantic import ValidationError

    from proto_tools.tools import ToolRegistry

    _raise_if_asset_missing(tool_key, result)
    output_class = ToolRegistry.get(tool_key).output_model
    try:
        return output_class.model_validate(result)
    except ValidationError as exc:
        raise TypeError(f"Tool {tool_key!r} result does not conform to {output_class.__name__}: {exc}") from exc


@contextlib.contextmanager
def _live_progress(
    configs: list[BaseConfig | None],
    expected_ends: int,
    *,
    environment: str | None = None,
    client: Any | None = None,
) -> Iterator[None]:
    """Stream the workers' log output to this process for the duration of the block.

    Stamps every config with a fresh queue partition, starts a daemon that tails it, and replays
    each record through the local logger, which drives the spinner exactly as a local run does.

    Entirely best-effort and additive. Streaming is off unless the caller is watching (a spinner is
    active) or has asked for verbose output, the queue is created before the workers start so a
    write never races its creation, and any failure here leaves the dispatch untouched. The tailer
    stops on the end sentinels or when this block exits, whichever comes first, which is what keeps
    a deployment too old to emit anything from leaving a thread polling.

    Args:
        configs (list[BaseConfig | None]): Configs about to be dispatched, stamped in place.
        expected_ends (int): Workers that will report, one end sentinel each.
        environment (str | None): Modal environment the dispatch resolves in. There is one queue
            per environment, so opening a different one leaves the tailer waiting on an empty
            queue while the container fills another.
        client (Any | None): Modal client to open and tail as, or ``None`` for the process's own.
    """
    present = [one for one in configs if one is not None]
    wanted = has_active_progress_bar() or verbose_level_from_env() > 0 or any(one.verbose > 0 for one in present)
    if not present or not wanted:
        yield
        return

    # Said before anything is dispatched, because the container reports nothing until it has
    # started, and on a cold start that is several seconds of a motionless spinner.
    set_substatus(remote_connecting_status("modal"))

    partition = uuid.uuid4().hex
    try:
        open_progress_queue(create=True, environment=environment, client=client)
    except Exception:  # a workspace that cannot host the queue simply gets no progress
        _client_logger.debug("progress queue unavailable; live updates disabled", exc_info=True)
        yield
        return

    # Debug only when the caller asked for detail, matching what the same verbosity shows locally.
    level = logging.DEBUG if max(one.verbose for one in present) >= 2 else logging.INFO
    for one in present:
        one._progress_partition = partition
        one._progress_level = level

    stop = threading.Event()
    tailer = threading.Thread(
        target=stream_modal_progress,
        args=(partition, expected_ends, stop),
        kwargs={"environment": environment, "client": client},
        name=f"proto-tools-progress-tail-{partition[:8]}",
        daemon=True,
    )
    tailer.start()
    try:
        yield
    finally:
        stop.set()
        tailer.join(timeout=2.0)
        for one in present:
            one._progress_partition = None


def dispatch_to_modal(
    key: str,
    inputs: BaseToolInput,
    config: BaseConfig | None = None,
    *,
    environment: str | None = None,
    scaledown_window: int | None = None,
    client: Any | None = None,
) -> BaseToolOutput:
    """Run a tool on Modal apps deployed in the active workspace.

    Args:
        key (str): Registry key (e.g. ``"esm2-embedding"``).
        inputs (BaseToolInput): Tool input payload.
        config (BaseConfig | None): Tool configuration. A logical
            ``device`` (``"proto"``/``"modal"``) or the ``"cpu"`` default is
            replaced with the physical device of the target container.
        environment (str | None): Modal environment. Defaults to ``MODAL_ENVIRONMENT``
            when set, otherwise to ``proto-env``.
        scaledown_window (int | None): Seconds an idle container stays alive,
            overriding the deployed value. Use the same value for every call
            in a session; see :func:`_bound_method`.
        client (Any | None): Modal client to dispatch as, for a caller acting on behalf of
            someone else. ``None`` lets Modal resolve the process's own credentials.

    Returns:
        BaseToolOutput: The validated tool output.

    Raises:
        ToolNotShippedError: If ``key`` has no deployment.
        ModalCredentialsError: If Modal credentials are missing or rejected.
        ToolNotDeployedError: If the tool's app is not deployed here.
        TypeError: If the result doesn't match the tool's output model.
    """
    app_name, service_class, method_name = resolve_tool(key)
    _require_modal_credentials(client)
    # Always pinned, never inherited. Left to the Modal profile this resolves to production,
    # where an app of the same name may belong to an entirely different project. Passed to every
    # Modal lookup below rather than published into os.environ, which two concurrent calls share.
    env = resolve_environment(environment)
    _warn_once_on_drift(key, service_class, environment=env, client=client)
    _warn_once_if_long_running(key, service_class)

    config = _resolve_device(config, service_class)
    method = _bound_method(app_name, service_class, method_name, key, scaledown_window, environment=env, client=client)
    with remote_progress("modal"), _live_progress([config], expected_ends=1, environment=env, client=client):
        result: dict[str, Any] = method.remote(
            input_dict=inputs.model_dump(mode="json"),
            config_dict=config.to_transport_dict() if config is not None else {},
        )
    return _validated_output(key, result)


def dispatch_batch_to_modal(
    key: str,
    inputs: list[BaseToolInput],
    config: BaseConfig | list[BaseConfig] | None = None,
    *,
    environment: str | None = None,
    scaledown_window: int | None = None,
    client: Any | None = None,
) -> list[BaseToolOutput | Exception]:
    """Run one tool over many inputs via Modal's ``starmap``, fanning out across containers.

    Args:
        key (str): Registry key.
        inputs (list[BaseToolInput]): One payload per run.
        config (BaseConfig | list[BaseConfig] | None): Shared configuration, or one per input
            when they differ. proto-tools sends one per chunk so each carries its position in the
            split batch, which a stochastic tool needs to keep its sampling from colliding.
        environment (str | None): Modal environment. Defaults to ``MODAL_ENVIRONMENT``
            when set, otherwise to ``proto-env``.
        scaledown_window (int | None): Seconds an idle container stays alive,
            overriding the deployed value.
        client (Any | None): Modal client to dispatch as, for a caller acting on behalf of
            someone else. ``None`` lets Modal resolve the process's own credentials.

    Returns:
        list[BaseToolOutput | Exception]: One entry per input, in input order. A chunk that failed
            contributes the exception it hit rather than aborting the batch, leaving the caller to
            decide what a partial result is worth.
    """
    app_name, service_class, method_name = resolve_tool(key)
    _require_modal_credentials(client)
    # Always pinned, never inherited. Left to the Modal profile this resolves to production,
    # where an app of the same name may belong to an entirely different project. Passed to every
    # Modal lookup below rather than published into os.environ, which two concurrent calls share.
    env = resolve_environment(environment)
    _warn_once_on_drift(key, service_class, environment=env, client=client)
    _warn_once_if_long_running(key, service_class)

    configs = list(config) if isinstance(config, list) else [config] * len(inputs)
    if len(configs) != len(inputs):
        raise ValueError(f"{key}: {len(configs)} config(s) for {len(inputs)} input(s)")
    configs = [_resolve_device(one, service_class) for one in configs]
    method = _bound_method(app_name, service_class, method_name, key, scaledown_window, environment=env, client=client)

    # return_exceptions keeps a failed chunk from discarding the chunks that already succeeded and
    # were already billed; the caller decides what to do with a partial batch. order_outputs keeps
    # each entry lined up with the input that produced it, which is what makes that decision
    # possible.
    outputs: list[BaseToolOutput | Exception] = []
    with (
        remote_progress("modal"),
        _live_progress(configs, expected_ends=len(inputs), environment=env, client=client),
    ):
        args = [
            (item.model_dump(mode="json"), one.to_transport_dict() if one is not None else {})
            for item, one in zip(inputs, configs, strict=True)
        ]
        results = list(method.starmap(args, return_exceptions=True, order_outputs=True))
    for result in results:
        if isinstance(result, Exception):
            outputs.append(result)
            continue
        try:
            outputs.append(_validated_output(key, result))
        except TypeError as exc:
            # A malformed result is this chunk's failure, not the batch's; raising here would
            # discard the chunks that came back intact.
            outputs.append(exc)
    return outputs
