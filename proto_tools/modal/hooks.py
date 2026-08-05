"""Extension points for code that runs inside a deployed worker.

Whoever operates a deployment may need to adapt a call or observe it without forking every
service class. Two extension points cover that, distinguished by what they can still see:

- A :data:`PayloadHook` runs on the raw mappings, before validation. The only place a value can
  still be rewritten.
- A :data:`CallMiddleware` wraps the call, and may transform what it returns.

Both are process-wide and applied in registration order. Register during import, before any call
is served: registration is not synchronized.

:data:`PLUGINS_ENV` names the modules a worker imports to perform that registration, since a
deployed container imports only what a service module reaches.
"""

import functools
import importlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

#: Modules a worker imports before serving its first call, separated by commas or whitespace.
#: Each is imported for its side effects, which is where a deployment registers its hooks. Read
#: in the environment a deploy runs from, and baked into the image so the container sees it.
PLUGINS_ENV = "PROTO_MODAL_WORKER_PLUGINS"

#: Adjusts a call's raw input and config mappings in place, before either is validated.
PayloadHook = Callable[[dict[str, Any], dict[str, Any]], None]


@dataclass(frozen=True)
class CallContext:
    """What a middleware is told about the call it is wrapping.

    Attributes:
        run_fn (Callable[..., Any]): The tool's ``run_*`` function.
    """

    run_fn: Callable[..., Any]

    @property
    def tool_key(self) -> str | None:
        """Registry key for this tool, or ``None``. A middleware's only handle on which call it is."""
        return _tool_key_for(self.run_fn)


#: Wraps one tool call. Receives its :class:`CallContext` and a zero-argument callable that
#: performs it; must call that and return a mapping.
CallMiddleware = Callable[[CallContext, Callable[[], dict[str, Any]]], dict[str, Any]]

_payload_hooks: list[PayloadHook] = []
_call_middleware: list[CallMiddleware] = []


@functools.cache
def _tool_key_for(run_fn: Callable[..., Any]) -> str | None:
    """Return the registry key whose tool is ``run_fn``, or ``None``. Cached: the map is static."""
    from proto_tools.tools import ToolRegistry

    return next((spec.key for spec in ToolRegistry.list_all() if spec.function is run_fn), None)


def register_payload_hook(hook: PayloadHook) -> None:
    """Register ``hook`` to run on every call's raw mappings before validation.

    Args:
        hook (PayloadHook): Callable taking ``(input_dict, config_dict)``. Mutates in place;
            its return value is ignored.
    """
    _payload_hooks.append(hook)


def register_call_middleware(middleware: CallMiddleware) -> None:
    """Register ``middleware`` to wrap every tool call in this process.

    Registration order is outermost first: the first registered sees the call before the second,
    and sees the second's result.

    Args:
        middleware (CallMiddleware): Callable taking the next step and returning a result mapping.
    """
    _call_middleware.append(middleware)


def _plugin_modules(raw: str) -> tuple[str, ...]:
    """Return the module names in ``raw``.

    Args:
        raw (str): Raw variable value, comma- or whitespace-separated.

    Returns:
        tuple[str, ...]: Module names in the order given, empty when ``raw`` names none.
    """
    return tuple(name for name in re.split(r"[,\s]+", raw) if name)


@functools.cache
def load_plugins() -> tuple[str, ...]:
    """Import every module named in :data:`PLUGINS_ENV`.

    An unimportable name raises rather than being skipped. A worker that quietly served calls
    without a deployment's extensions would drop whatever they were responsible for.

    Cached to keep this off the hot path; ``sys.modules`` is what makes a module body run once.

    Returns:
        tuple[str, ...]: The module names imported, in the order given.

    Raises:
        ImportError: If a named module cannot be imported.
    """
    names = _plugin_modules(os.environ.get(PLUGINS_ENV, ""))
    for name in names:
        _import_plugin(name)
    return names


def _import_plugin(name: str) -> None:
    """Import one plugin module, naming the variable it came from when that fails.

    Args:
        name (str): Module to import.

    Raises:
        ImportError: If the module cannot be imported.
    """
    try:
        importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(f"{PLUGINS_ENV} names {name!r}, which this worker cannot import") from exc


def plugin_env() -> dict[str, str]:
    """Return :data:`PLUGINS_ENV` as image env, or empty when unset.

    Returns:
        dict[str, str]: Mapping to merge into a service image's runtime environment.
    """
    raw = os.environ.get(PLUGINS_ENV)
    return {PLUGINS_ENV: raw} if raw else {}


def clear_hooks() -> None:
    """Remove every registered hook and forget which plugins were loaded.

    Intended for tests, which must not leak into each other.
    """
    _payload_hooks.clear()
    _call_middleware.clear()
    load_plugins.cache_clear()


def apply_payload_hooks(input_dict: dict[str, Any], config_dict: dict[str, Any]) -> None:
    """Run every registered payload hook, in registration order.

    Args:
        input_dict (dict[str, Any]): The call's raw input mapping, mutated in place.
        config_dict (dict[str, Any]): The call's raw config mapping, mutated in place.
    """
    for hook in _payload_hooks:
        hook(input_dict, config_dict)


def run_with_middleware(context: CallContext, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Invoke ``call`` through every registered middleware, outermost first.

    Args:
        context (CallContext): Describes the call, passed to each middleware.
        call (Callable[[], dict[str, Any]]): Performs the call and returns its result mapping.

    Returns:
        dict[str, Any]: The result, as returned by the outermost middleware. With none
            registered, exactly what ``call`` returned.

    Raises:
        TypeError: If a middleware returns a non-mapping — usually one that forgot to return at
            all, whose ``None`` would otherwise surface as a client-side error naming no middleware.
    """
    wrapped = call
    # Reversed so the first registered ends up outermost, matching the documented order.
    for middleware in reversed(_call_middleware):
        wrapped = _bind(middleware, context, wrapped)
    result = wrapped()
    if not isinstance(result, Mapping):
        raise TypeError(
            f"A registered call middleware returned {type(result).__name__}, not a mapping. "
            f"Middleware must return the result of the step it wraps."
        )
    return result


def _bind(
    middleware: CallMiddleware, context: CallContext, next_step: Callable[[], dict[str, Any]]
) -> Callable[[], dict[str, Any]]:
    """Bind this layer's arguments, so building the chain in a loop cannot late-bind them."""
    return lambda: middleware(context, next_step)


__all__ = [
    "PLUGINS_ENV",
    "CallContext",
    "CallMiddleware",
    "PayloadHook",
    "apply_payload_hooks",
    "clear_hooks",
    "load_plugins",
    "plugin_env",
    "register_call_middleware",
    "register_payload_hook",
    "run_with_middleware",
]
