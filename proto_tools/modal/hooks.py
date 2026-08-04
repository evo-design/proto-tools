"""Extension points for code that runs inside a deployed worker.

A deployment is not always just the tool. Whoever operates one may need to adapt a call or
observe it — resolving a reference the caller passed instead of a value, recording timings,
moving a large result somewhere the transport is happier with — without forking every service
class to do it.

Two extension points cover that, distinguished by what they can still see:

- A :data:`PayloadHook` runs on the raw mappings, before they are validated into models. This is
  the only place a value can still be rewritten, because validation may reject or normalize it.
- A :data:`CallMiddleware` wraps the call itself, and may transform what it returns.

Both are process-wide and applied in registration order. Register them at import time, from the
module that defines a deployment's entry point, so every call through that worker sees them.
"""

from collections.abc import Callable
from typing import Any

#: Adjusts a call's raw input and config mappings in place, before either is validated.
PayloadHook = Callable[[dict[str, Any], dict[str, Any]], None]

#: Wraps one tool call. Receives a zero-argument callable that performs the call and returns its
#: result mapping; must call it and return a mapping. Wrapping it in a context manager, timing it,
#: or transforming the result are all ordinary uses.
CallMiddleware = Callable[[Callable[[], dict[str, Any]]], dict[str, Any]]

_payload_hooks: list[PayloadHook] = []
_call_middleware: list[CallMiddleware] = []


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


def clear_hooks() -> None:
    """Remove every registered hook. Intended for tests, which must not leak into each other."""
    _payload_hooks.clear()
    _call_middleware.clear()


def apply_payload_hooks(input_dict: dict[str, Any], config_dict: dict[str, Any]) -> None:
    """Run every registered payload hook, in registration order.

    Args:
        input_dict (dict[str, Any]): The call's raw input mapping, mutated in place.
        config_dict (dict[str, Any]): The call's raw config mapping, mutated in place.
    """
    for hook in _payload_hooks:
        hook(input_dict, config_dict)


def run_with_middleware(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Invoke ``call`` through every registered middleware, outermost first.

    Args:
        call (Callable[[], dict[str, Any]]): Performs the call and returns its result mapping.

    Returns:
        dict[str, Any]: The result, as returned by the outermost middleware. With none
            registered, exactly what ``call`` returned.
    """
    wrapped = call
    # Reversed so the first registered ends up outermost, matching the documented order.
    for middleware in reversed(_call_middleware):
        wrapped = _bind(middleware, wrapped)
    return wrapped()


def _bind(middleware: CallMiddleware, next_step: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
    """Bind ``next_step`` into ``middleware``, so the chain can be built without late binding."""
    return lambda: middleware(next_step)


__all__ = [
    "CallMiddleware",
    "PayloadHook",
    "apply_payload_hooks",
    "clear_hooks",
    "register_call_middleware",
    "register_payload_hook",
    "run_with_middleware",
]
