"""Cloud-dispatch support for the ``device="cloud"`` option.

When a tool is called with ``config.device == "cloud"``,
``ToolRegistry._try_dispatch`` routes the call to Proto's remote
execution service via ``proto-client`` before local preprocessing or
worker setup runs.

Enable with :func:`use_api_backend` (requires the ``cloud`` extra:
``pip install proto-tools[cloud]``).

Example::

    from proto_tools.cloud import use_api_backend

    use_api_backend()  # reads PROTO_API_KEY from env
    result = run_esmfold(inputs, Config(device="cloud"))
"""

import logging
import os
from typing import Any

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.base_config import BaseConfig
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)

# TODO: remove once cloud is generally available — drop _NOT_READY_MSG, the API-key precheck in use_api_backend(), and the ProtoAuthError catch in _route_to_cloud, so all SDK exceptions propagate.
_NOT_READY_MSG = (
    "\n"
    "  ┌─────────────────────────────────────────────────────────────┐\n"
    "  │  Proto Cloud (device='cloud') is coming soon!               │\n"
    "  │                                                             │\n"
    "  │  Our hosted execution service is not yet generally          │\n"
    "  │  available. For now, please run tools locally using         │\n"
    "  │  device='cpu' or device='cuda'.                             │\n"
    "  └─────────────────────────────────────────────────────────────┘\n"
)

_enabled: bool = False


def use_api_backend(
    *,
    poll_interval: float = 1.0,
    timeout: float = 600.0,
    **client_kwargs: Any,
) -> None:
    """Enable ``device="cloud"`` by routing tool runs through ``proto-client``.

    Constructs a :class:`proto_client.ProtoClient` and installs a dispatch
    hook on :class:`ToolRegistry`. The SDK reads ``PROTO_API_KEY`` (and
    optionally ``PROTO_TOOLS_BASE_URL``) from the environment unless
    overridden via ``client_kwargs``.

    Args:
        poll_interval (float): Seconds between job-status polls.
        timeout (float): Max seconds to wait for a single tool to complete.
        client_kwargs (Any): Forwarded to :class:`proto_client.ProtoClient`
            (e.g. ``api_key``, ``tools_base_url``).

    Raises:
        ImportError: If ``proto-client`` is not installed. Install with
            ``pip install proto-tools[cloud]``.
        NotImplementedError: If no API key is configured (neither
            ``api_key`` kwarg nor ``PROTO_API_KEY`` env var). The
            placeholder message documents that cloud isn't generally
            available yet.
    """
    try:
        from proto_client import ProtoClient
        from proto_client.errors import ProtoAuthError
    except ImportError as exc:
        raise ImportError(
            "device='cloud' requires proto-client. Install with `pip install proto-tools[cloud]`."
        ) from exc

    # API-key gate: no key configured → user-facing placeholder. Auth failures from a configured-but-invalid key are caught below.
    if not client_kwargs.get("api_key") and not os.environ.get("PROTO_API_KEY"):
        raise NotImplementedError(_NOT_READY_MSG)

    client = ProtoClient(**client_kwargs)

    def _route_to_cloud(
        _cls: type,
        key: str,
        inputs: BaseToolInput,
        config: BaseConfig | None,
    ) -> BaseToolOutput | None:
        if config is None or config.device != "cloud":
            return None
        output_class = ToolRegistry.get(key).output_model
        # device='cloud' is the routing signal for this client; the server picks its own physical device, so strip it before sending.
        config_payload = config.model_dump(exclude_none=True)
        config_payload.pop("device", None)
        try:
            response = client.tools.run(
                key,
                inputs=inputs.model_dump(exclude_none=True),
                config=config_payload,
                poll_interval=poll_interval,
                timeout=timeout,
                output_model=output_class,
            )
        except ProtoAuthError as exc:
            # Invalid / unauthorized key — surface the user-facing placeholder. All other SDK exceptions propagate.
            logger.debug("Cloud auth error for %r: %s", key, exc, exc_info=True)
            raise NotImplementedError(_NOT_READY_MSG) from exc

        # SDK swaps in the validated instance when output_model is passed; widen for the type checker.
        result: Any = response.result
        if not isinstance(result, output_class):
            raise TypeError(f"Tool {key!r} returned {type(result).__name__}, expected {output_class.__name__}")
        return result

    setattr(ToolRegistry, "_try_dispatch", classmethod(_route_to_cloud))  # noqa: B010
    global _enabled  # noqa: PLW0603 — module-level on/off flag
    _enabled = True


def disable_api_backend() -> None:
    """Restore default local dispatch. Primarily for tests."""

    def _noop(
        _cls: type,
        _key: str,
        _inputs: BaseToolInput,
        _config: BaseConfig | None,
    ) -> BaseToolOutput | None:
        return None

    setattr(ToolRegistry, "_try_dispatch", classmethod(_noop))  # noqa: B010
    global _enabled  # noqa: PLW0603 — module-level on/off flag
    _enabled = False


def is_api_backend_enabled() -> bool:
    """Return True iff :func:`use_api_backend` has been called (and not since disabled)."""
    return _enabled


__all__ = ["disable_api_backend", "is_api_backend_enabled", "use_api_backend"]
