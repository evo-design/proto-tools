"""Registry mapping each tool key to the service class and method that serve it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, tuple[str, str]] = {}


def register_tools(mapping: dict[str, str]) -> Callable[[type[Any]], type[Any]]:
    """Class decorator that registers tool keys for a service class.

    Place **below** ``@app.cls()`` so it sees the raw class for method validation::

        @app.cls(image=image, gpu="H100:1", ...)
        @register_tools({"esm2-sample": "sample", "esm2-score": "score"})
        class ESM2Service:
            ...

    Raises ``ValueError`` on duplicate keys. Raises ``AttributeError`` on missing methods.
    """

    def decorator(cls: type[Any]) -> type[Any]:
        class_name = cls.__name__
        for tool_key, method_name in mapping.items():
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
