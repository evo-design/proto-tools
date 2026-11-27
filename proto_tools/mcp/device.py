"""Selection of the backend an MCP server talks to, resolved once at startup."""

from __future__ import annotations

import os
from typing import Literal

from proto_tools.utils.device import is_remote_device

Device = Literal["modal", "proto", "local"]

DEVICE_ENV = "PROTO_MCP_DEVICE"
API_KEY_ENV = "PROTO_API_KEY"

# The registry's own guard, which derives its set from ``RemoteDevice`` so the type and the
# runtime check cannot drift. ``local`` is the one Device it rejects, which is the question here.
is_remote = is_remote_device


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested backend cannot be used as configured."""


def resolve_device(requested: str | None = None) -> Device:
    """Return the backend this server will use.

    Args:
        requested (str | None): An explicit choice, from ``--device`` or
            :data:`DEVICE_ENV`. ``None`` takes the default.

    Returns:
        Device: ``"modal"``, ``"proto"``, or ``"local"``.

    Raises:
        DeviceUnavailableError: If ``proto`` is requested without an API key, or
            the name is not a known device.
    """
    choice = requested or os.environ.get(DEVICE_ENV)
    if not choice:
        return "modal"

    if choice == "modal":
        return "modal"

    if choice == "local":
        return "local"

    if choice == "proto":
        if not os.environ.get(API_KEY_ENV):
            raise DeviceUnavailableError(
                f"device='proto' needs {API_KEY_ENV}; see https://proto.evodesign.org for more "
                "information. Or omit --device to run tools in your own Modal workspace."
            )
        return "proto"

    raise DeviceUnavailableError(f"unknown device {choice!r}; expected 'modal', 'proto', or 'local'")


def is_deployable(device: Device) -> bool:
    """Report whether the caller can deploy tools to ``device`` themselves."""
    return device == "modal"
