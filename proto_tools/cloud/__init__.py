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
from typing import Any

from pydantic import ValidationError

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.base_config import BaseConfig
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput

logger = logging.getLogger(__name__)

# TODO: replace "(request link coming soon)" / "(contact link coming soon)" with the real signup + support URLs once they're live.
# TODO: drop _CLOUD_STATUS / _INVALID_KEY and the ProtoAuthError catch once cloud is generally available — SDK exceptions should propagate as-is.

_CLOUD_STATUS = (
    "\n"
    "No API key was detected.\n"
    "\n"
    "device='cloud' is coming soon! We're rolling out to beta-testers in the\n"
    "coming weeks — only approved users will have API keys for now.\n"
    "\n"
    "Request access: (request link coming soon)\n"
    "\n"
    "Once approved, set PROTO_API_KEY in your environment and re-run;\n"
    "device='cloud' will dispatch automatically.\n"
    "\n"
    "In the meantime, run locally with device='cpu' or device='cuda'.\n"
)

_INVALID_KEY = (
    "\n"
    "Your Proto API key was rejected.\n"
    "\n"
    "Double-check the PROTO_API_KEY value and confirm your account is\n"
    "approved for beta access. If you believe this is in error, contact\n"
    "the Proto team: (contact link coming soon)\n"
)

_DEFAULT_POLL_INTERVAL = 1.0
_DEFAULT_FALLBACK_TIMEOUT = 600.0


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
            beta-access status message.
        PermissionError: If the configured key is rejected by the server.
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
    if tool_timeout is None:
        logger.warning(
            "No timeout configured for cloud run of %r; capping at %ss.",
            key,
            _DEFAULT_FALLBACK_TIMEOUT,
        )
        tool_timeout = _DEFAULT_FALLBACK_TIMEOUT

    try:
        response = client.tools.run(
            key,
            inputs=inputs.model_dump(exclude_none=True),
            config=config_payload,
            poll_interval=_DEFAULT_POLL_INTERVAL,
            timeout=float(tool_timeout),
            output_model=None,
        )
    except ProtoAuthError as exc:
        # Distinct from the no-key path, which raises NotImplementedError(_CLOUD_STATUS) above. All other SDK exceptions propagate.
        logger.debug("Cloud auth error for %r: %s", key, exc, exc_info=True)
        raise PermissionError(_INVALID_KEY) from exc

    decoded_result = _decode_output_assets(response.result, client.assets)
    try:
        return output_class.model_validate(decoded_result)
    except ValidationError as exc:
        raise TypeError(f"Tool {key!r} result does not conform to {output_class.__name__}: {exc}") from exc


__all__ = ["dispatch_to_cloud"]
