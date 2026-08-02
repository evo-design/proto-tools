"""Registry mapping each tool key to the service class and method that serve it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, tuple[str, str]] = {}


# Tools with redistribution=false that this workspace is licensed to host.
#
# Intentionally EMPTY here. Only tools whose upstream
# license.yaml sets ``redistribution: true``, so the registry filters the
# deployable set on its own and no user needs to attest to a license they
# may not hold. A deployer who has independently obtained the rights to a
# non-redistributable tool can add its key locally, but that is their
# licensing decision to make, not a default shipped in this repo.
LICENSE_GATED_TOOL_KEYS: frozenset[str] = frozenset()


def _is_redistributable(tool_key: str) -> bool:
    """Return True iff proto-tools' license metadata permits redistribution.

    Reads ``license.yaml`` for the tool via :meth:`ToolRegistry.get_license`.
    Fails closed in three cases — an unknown tool key (proto-tools rejects
    it), a missing license file, or a missing ``redistribution`` field —
    since hosting a tool we can't verify is riskier than blocking deployment
    of one with incomplete metadata.

    Args:
        tool_key (str): Tool identifier (e.g. ``"esmfold-prediction"``).

    Returns:
        bool: True iff ``tool_key`` is a known tool, ``license.yaml`` exists,
            and ``redistribution: true``.
    """
    from proto_tools.tools import ToolRegistry

    try:
        license_data = ToolRegistry.get_license(tool_key)
    except ValueError:
        return False
    if license_data is None:
        return False
    return bool(license_data.get("redistribution"))


def _is_deployable(tool_key: str) -> bool:
    """Return True iff ``tool_key`` is redistributable or on :data:`LICENSE_GATED_TOOL_KEYS`."""
    return _is_redistributable(tool_key) or tool_key in LICENSE_GATED_TOOL_KEYS


def register_tools(mapping: dict[str, str]) -> Callable[[type[Any]], type[Any]]:
    """Class decorator that registers tool keys for a service class.

    Place **below** ``@app.cls()`` so it sees the raw class for method validation::

        @app.cls(image=image, gpu="H100:1", ...)
        @register_tools({"esm2-sample": "sample", "esm2-score": "score"})
        class ESM2Service:
            ...

    Raises ``ValueError`` on duplicate keys or on keys blocked by deployment
    license policy. Raises ``AttributeError`` on missing methods.
    """

    def decorator(cls: type[Any]) -> type[Any]:
        class_name = cls.__name__
        for tool_key, method_name in mapping.items():
            if not _is_deployable(tool_key):
                msg = (
                    f"Tool '{tool_key}' has redistribution=false in proto-tools and is not "
                    f"on LICENSE_GATED_TOOL_KEYS (attempted by {class_name}.{method_name}). "
                    f"Either flip redistribution upstream, or add the key to the allowlist "
                    f"if this workspace holds the necessary license."
                )
                raise ValueError(msg)
            if tool_key in _REGISTRY:
                existing_cls, existing_method = _REGISTRY[tool_key]
                msg = (
                    f"Duplicate tool key '{tool_key}': already registered to "
                    f"{existing_cls}.{existing_method}, cannot register to "
                    f"{class_name}.{method_name}"
                )
                raise ValueError(msg)
            if not hasattr(cls, method_name):
                msg = f"Tool '{tool_key}' references method '{method_name}' which does not exist on {class_name}"
                raise AttributeError(msg)
            _REGISTRY[tool_key] = (class_name, method_name)
        return cls

    return decorator


def get_registry() -> dict[str, tuple[str, str]]:
    """Return a snapshot of the current registry."""
    return dict(_REGISTRY)


def import_all_services() -> None:
    """Import every service module, so the registry reflects the whole manifest.

    Registration happens as a side effect of ``@register_tools`` at class
    definition, so a service nobody imported is absent from the registry. Driven
    from the manifest rather than a generated aggregate module, so a service
    added there needs no other file to exist.
    """
    import importlib

    from proto_tools.modal.manifest import SERVICE_TO_MODULE

    for module in SERVICE_TO_MODULE.values():
        importlib.import_module(module)
