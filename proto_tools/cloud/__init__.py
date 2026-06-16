"""Cloud-dispatch support for the ``device="cloud"`` option.

When a tool is called with ``config.device == "cloud"``, the registry
routes the call to Proto's hosted execution service via
:func:`dispatch_to_cloud`.

The dispatcher reads ``PROTO_API_KEY`` from the environment (or accepts
an explicit ``api_key`` kwarg). No setup ceremony is required — cloud is
available whenever ``proto-client`` is installed and a key is configured.

Example::

    from proto_tools import run_esmfold

    result = run_esmfold(inputs, Config(device="cloud"))  # uses PROTO_API_KEY
"""

import functools
import logging
import os
import sys
import threading
import time
from typing import Any

from pydantic import ValidationError

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.base_config import BaseConfig
from proto_tools.utils.logging_config import verbose_level_from_env
from proto_tools.utils.progress import _in_notebook, progress_bar, set_substatus
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)

# Replayed server records land here so the user's configured handlers render them like local logs.
_remote_logger = logging.getLogger("proto_tools.cloud.remote")


def _status_box(title: str, body: str) -> str:
    """Wrap a status message in a double-ruled box so it stands out in the terminal."""
    lines = body.strip("\n").splitlines()
    w = max(len(title) + 4, max((len(line) for line in lines), default=0) + 4)
    border = "═" * w
    boxed = [
        "",
        f"╔{border}╗",
        f"║{title:^{w}}║",
        f"║{'':{w}}║",
        *[f"║  {line:<{w - 2}}║" for line in lines],
        f"╚{border}╝",
        "",
    ]
    return "\n".join(boxed)


_CLOUD_STATUS = _status_box(
    "Proto Cloud: request access",
    "No API key was detected.\n"
    "\n"
    "device='cloud' is coming soon! We're rolling out access in the coming\n"
    "weeks; only approved users will have API keys for now.\n"
    "\n"
    "Request access: (request link coming soon)\n"
    "\n"
    "Once approved, set PROTO_API_KEY in your environment and re-run;\n"
    "device='cloud' will dispatch automatically.\n"
    "\n"
    "In the meantime, run locally with device='cpu' or device='cuda'.",
)

_INVALID_KEY = _status_box(
    "Proto Cloud: This key appears to be invalid",
    "We couldn't validate your Proto API key.\n"
    "\n"
    "Double-check the PROTO_API_KEY value and confirm your account is\n"
    "approved for cloud access. If you believe this is in error, contact\n"
    "the Proto team: (contact link coming soon)",
)


def is_cloud_hostable(key: str) -> bool:
    """True iff the tool's license permits running on Proto's hosted cloud service.

    Mirrors the hosted service's deploy gate: a tool is hostable iff its
    ``license.yaml`` declares ``redistribution: true``. Fails closed; an
    unknown key, a missing/unreadable/malformed ``license.yaml``, or a
    missing/false ``redistribution`` field all read as "not hostable", so we
    never promise cloud support the hosted service can't honor.

    Note: the hosted service additionally licenses a few non-redistributable
    tools by allowlist; that exception is not mirrored here yet, so those tools
    report ``False`` even though the server can run them.
    """
    try:
        license_data = ToolRegistry.get_license(key)
    except Exception as exc:
        # Fail closed on any license-read error (unknown key, unreadable / malformed YAML).
        logger.debug("Cloud hostability check for %r failed closed: %s", key, exc, exc_info=True)
        return False
    return bool(license_data and license_data.get("redistribution"))


def cloud_unhostable_message(key: str) -> str:
    """Boxed error shown when ``device='cloud'`` targets a tool the cloud can't host."""
    return _status_box(
        "Proto Cloud: tool not hostable",
        f"Tool {key!r} can't run with device='cloud'.\n"
        "\n"
        "Its license prohibits redistribution, so Proto's hosted service\n"
        "can't run it remotely.\n"
        "\n"
        "Run it locally instead with device='cpu' or device='cuda'.",
    )


_DEFAULT_POLL_INTERVAL = 1.0
_LOG_THREAD_JOIN_TIMEOUT = 2.0

# RFC 5424 severity → Python logging level.
_RFC5424_TO_PY_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}

# Map verbose (0=quiet, 1=info, 2=debug, 3=raw) to server-side level/stream filters; None = unfiltered.
_VERBOSE_LEVEL_FILTER: dict[int, list[str] | None] = {
    0: ["warning", "error", "critical", "alert", "emergency"],
    1: ["info", "notice", "warning", "error", "critical", "alert", "emergency"],
    2: None,
    3: None,
}
_VERBOSE_STREAM_FILTER: dict[int, list[str] | None] = {
    0: ["system"],
    1: ["system", "stdout"],
    2: ["system", "stdout"],
    3: None,
}


def _cloud_spinner() -> tuple[str, str | None]:
    """Cloud-mode spinner as a ``(spinner_style, prefix)`` pair for ``progress_bar``.

    On emoji-capable terminals (and notebooks, which render in the browser) it's the
    ``cloud`` style — a pulse bouncing between a 💻 and a ☁️. Non-UTF terminals fall
    back to the plain dotted spinner with a ``[cloud]`` badge.
    """
    if _in_notebook():
        return "cloud", None
    enc = (getattr(sys.stderr, "encoding", "") or "").lower()
    if "utf" in enc:
        return "cloud", None
    return "dots", "[cloud]"


@functools.lru_cache(maxsize=4)
def _get_client(api_key: str) -> Any:
    from proto_client import ProtoClient

    return ProtoClient(api_key=api_key)


def _is_output_asset_ref(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("id"), str) and value.get("kind") == "output"


def _decode_output_assets(value: Any, assets: Any) -> Any:
    if _is_output_asset_ref(value):
        try:
            return assets.decode(value)
        except Exception as exc:
            raise RuntimeError(f"Failed to decode cloud output asset {value.get('id')!r}: {exc}") from exc
    if isinstance(value, dict):
        return {k: _decode_output_assets(v, assets) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_output_assets(v, assets) for v in value]
    return value


def _effective_verbose(config: BaseConfig | None) -> int:
    """Mirror local execution: max of ``config.verbose`` and ``PROTO_WORKER_VERBOSE``."""
    cfg_verbose = int(config.verbose) if config is not None else 0
    return max(cfg_verbose, verbose_level_from_env())


def _strip_logger_prefix(msg: str) -> str:
    """Drop the leading ``<logger.name>: `` the server prepends to logging records, so cloud output reads like local.

    Only strips a dotted, space-free head (a logger name like ``proto_tools.worker.esmfold``); a message that
    merely starts with ``word: `` (e.g. ``Done: 5``) or real tool stdout (``Epoch 1: loss=0.5``) is left untouched.
    """
    head, sep, tail = msg.partition(": ")
    return tail if sep and "." in head and " " not in head else msg


def _stream_remote_logs(client: Any, key: str, job_id: str, verbose: int) -> None:
    """Stream NDJSON job logs from the server and replay each through the local logger.

    Best-effort: any failure (network blip, missing endpoint on an older
    server) downgrades to a debug log and lets the main thread finish the run
    normally. Exit conditions, in order: the server emits a ``LogsEnd``
    terminator (normal path), or the connection closes / errors out, or the
    main thread reaches its join timeout and abandons the daemon thread
    (slow-server safety net).
    """
    level_filter = _VERBOSE_LEVEL_FILTER.get(verbose)
    stream_filter = _VERBOSE_STREAM_FILTER.get(verbose)
    try:
        records = client.tools.iter_job_logs(
            key,
            job_id,
            follow=True,
            level=level_filter,
            stream=stream_filter,
        )
        for rec in records:
            if getattr(rec, "type", None) == "end":
                break
            msg = getattr(rec, "msg", None)
            if not msg:
                continue
            update_status = getattr(rec, "update_status", False)
            py_level = _RFC5424_TO_PY_LEVEL.get(getattr(rec, "level", "info"), logging.INFO)
            # Replay the phase flag so the local SpinnerFromLogsHandler drives the bar, exactly like a local run.
            # Strip the worker logger-name prefix from every line so cloud output reads identically to local.
            _remote_logger.log(py_level, "%s", _strip_logger_prefix(msg), extra={"update_status": update_status})
    except Exception as exc:
        logger.debug("Remote log stream for job %s ended: %s", job_id, exc, exc_info=True)


# Cloud job status → spinner phase shown while polling; unmapped statuses display verbatim.
_CLOUD_STATUS_PHASE = {"pending": "queued", "running": "running"}
_CLOUD_INITIAL_PHASE = "queued"


def _poll_until_terminal(client: Any, key: str, job_id: str, poll_interval: float, timeout: float | None) -> Any:
    """Block until the job reaches a terminal status; return the final job envelope.

    ``timeout=None`` polls with no client-side deadline, honoring
    ``config.timeout=None`` ("wait indefinitely").
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    last_status: str | None = None
    while True:
        status = client.tools.get(key, job_id)
        status_value = getattr(status.status, "value", status.status)
        if status_value == "completed":
            return status
        if status_value == "failed":
            raise RuntimeError(f"Cloud job {job_id} failed: {status.error}")
        if status_value == "cancelled":
            raise RuntimeError(f"Cloud job {job_id} was cancelled")
        if status_value != last_status:
            # Coarse phase from job status; streamed system records (if any) refine it.
            set_substatus(_CLOUD_STATUS_PHASE.get(status_value, status_value))
            last_status = status_value
        if deadline is None:
            time.sleep(poll_interval)
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Cloud job {job_id} did not complete within {timeout}s")
        time.sleep(min(poll_interval, remaining))


def dispatch_to_cloud(
    key: str,
    inputs: BaseToolInput,
    config: BaseConfig | None,
    *,
    api_key: str | None = None,
) -> BaseToolOutput:
    """Submit a tool call to Proto's hosted execution service.

    Args:
        key (str): Registry key (e.g. ``"esmfold-prediction"``).
        inputs (BaseToolInput): Tool input payload.
        config (BaseConfig | None): Tool configuration. ``config.device``
            is stripped before sending — the server picks its own
            physical device.
        api_key (str | None): Overrides ``PROTO_API_KEY`` env var.

    Returns:
        BaseToolOutput: The validated tool output.

    Raises:
        ImportError: If ``proto-client`` is not installed. Install with
            ``pip install proto-client``.
        NotImplementedError: If no API key is configured. Surfaces the
            cloud-access status message.
        PermissionError: If the server does not accept the configured key.
        TypeError: If the server response doesn't match the tool's
            ``output_model`` schema.
    """
    try:
        from proto_client.errors import ProtoAuthError
    except ImportError as exc:
        raise ImportError("device='cloud' requires proto-client. Install with `pip install proto-client`.") from exc

    resolved_key = api_key if api_key is not None else os.environ.get("PROTO_API_KEY")
    if not resolved_key:
        raise NotImplementedError(_CLOUD_STATUS)

    client = _get_client(resolved_key)
    output_class = ToolRegistry.get(key).output_model

    # device='cloud' is the client-side routing signal; the server picks its own physical device, so strip it before sending.
    config_payload = config.model_dump(exclude_none=True) if config is not None else {}
    config_payload.pop("device", None)

    tool_timeout: float | None = config.effective_timeout() if config is not None else None

    verbose = _effective_verbose(config)

    # Status-only spinner across the round-trip; its substatus tracks job status and streamed phases.
    spinner_style, spinner_prefix = _cloud_spinner()
    with progress_bar(desc=_CLOUD_INITIAL_PHASE, prefix=spinner_prefix, spinner_style=spinner_style, show_bar=False):
        try:
            job_id = client.tools.submit(
                key,
                inputs=inputs.model_dump(exclude_none=True),
                config=config_payload,
            )
        except ProtoAuthError as exc:
            # Auth fails on submit, before we have a job_id.
            logger.debug("Cloud auth error for %r: %s", key, exc, exc_info=True)
            raise PermissionError(_INVALID_KEY) from exc

        log_thread: threading.Thread | None = None
        if hasattr(client.tools, "iter_job_logs"):
            log_thread = threading.Thread(
                target=_stream_remote_logs,
                args=(client, key, job_id, verbose),
                name=f"proto-cloud-logs-{job_id[:8]}",
                daemon=True,
            )
            log_thread.start()

        try:
            response = _poll_until_terminal(
                client, key, job_id, _DEFAULT_POLL_INTERVAL, None if tool_timeout is None else float(tool_timeout)
            )
        finally:
            # Brief join to drain the trailing log stream; the daemon thread is abandoned if it stalls.
            if log_thread is not None:
                log_thread.join(timeout=_LOG_THREAD_JOIN_TIMEOUT)

    decoded_result = _decode_output_assets(response.result, client.assets)
    try:
        return output_class.model_validate(decoded_result)
    except ValidationError as exc:
        raise TypeError(f"Tool {key!r} result does not conform to {output_class.__name__}: {exc}") from exc


__all__ = ["cloud_unhostable_message", "dispatch_to_cloud", "is_cloud_hostable"]
