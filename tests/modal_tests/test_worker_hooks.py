"""Extension points a worker exposes, and the guarantees they carry."""

from __future__ import annotations

from typing import Any

import pytest

from proto_tools.modal.hooks import (
    apply_payload_hooks,
    clear_hooks,
    register_call_middleware,
    register_payload_hook,
    run_with_middleware,
)


@pytest.fixture(autouse=True)
def _isolate_hooks():
    """Hooks are process-wide, so a test that registers one must not leak into the next."""
    clear_hooks()
    yield
    clear_hooks()


def test_nothing_registered_is_a_pass_through() -> None:
    """The default must cost nothing and change nothing: most deployments register no hooks."""
    payload: dict[str, Any] = {"sequences": ["ACGT"]}
    config: dict[str, Any] = {"batch_size": 2}
    apply_payload_hooks(payload, config)
    assert payload == {"sequences": ["ACGT"]}
    assert config == {"batch_size": 2}
    assert run_with_middleware(lambda: {"ok": True}) == {"ok": True}


def test_a_payload_hook_sees_both_mappings() -> None:
    """Config carries file references as often as inputs do, so a hook must reach both."""
    seen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    register_payload_hook(lambda i, c: seen.append((i, c)))

    apply_payload_hooks({"a": 1}, {"b": 2})
    assert seen == [({"a": 1}, {"b": 2})]


def test_a_payload_hook_can_rewrite_a_value() -> None:
    """The reason this hook runs before validation: a value may need resolving to pass it."""

    def resolve(input_dict: dict[str, Any], _config: dict[str, Any]) -> None:
        if input_dict.get("path") == "ref://x":
            input_dict["path"] = "/staged/x"

    register_payload_hook(resolve)
    payload = {"path": "ref://x"}
    apply_payload_hooks(payload, {})
    assert payload == {"path": "/staged/x"}


def test_payload_hooks_run_in_registration_order() -> None:
    """Later hooks see earlier ones' edits, which is what makes them composable."""
    register_payload_hook(lambda i, _c: i.__setitem__("order", "first"))
    register_payload_hook(lambda i, _c: i.__setitem__("order", i["order"] + "-second"))

    payload: dict[str, Any] = {}
    apply_payload_hooks(payload, {})
    assert payload["order"] == "first-second"


def test_middleware_wraps_the_call() -> None:
    """The common case: do something before and after, and return the result untouched."""
    events: list[str] = []

    def timing(next_step):
        events.append("before")
        result = next_step()
        events.append("after")
        return result

    register_call_middleware(timing)
    assert run_with_middleware(lambda: {"value": 1}) == {"value": 1}
    assert events == ["before", "after"]


def test_middleware_can_transform_the_result() -> None:
    """A large field may need moving elsewhere before the transport sees it."""
    register_call_middleware(lambda nxt: {**nxt(), "added": True})
    assert run_with_middleware(lambda: {"value": 1}) == {"value": 1, "added": True}


def test_middleware_may_wrap_the_call_in_a_context() -> None:
    """Capturing output for the duration of a call is the motivating shape."""
    import contextlib

    entered: list[str] = []

    @contextlib.contextmanager
    def capture():
        entered.append("open")
        try:
            yield
        finally:
            entered.append("close")

    def with_capture(next_step):
        with capture():
            return next_step()

    register_call_middleware(with_capture)
    run_with_middleware(lambda: {"ok": True})
    assert entered == ["open", "close"]


def test_the_first_registered_middleware_is_outermost() -> None:
    """Documented ordering. Reversing it would silently invert nesting for anyone relying on it."""
    order: list[str] = []

    def outer(next_step):
        order.append("outer-in")
        result = next_step()
        order.append("outer-out")
        return result

    def inner(next_step):
        order.append("inner-in")
        result = next_step()
        order.append("inner-out")
        return result

    register_call_middleware(outer)
    register_call_middleware(inner)
    run_with_middleware(dict)
    assert order == ["outer-in", "inner-in", "inner-out", "outer-out"]


def test_each_middleware_binds_its_own_next_step() -> None:
    """Building the chain in a loop invites late binding, where every layer calls the last one."""
    calls: list[int] = []

    def make(index: int):
        def middleware(next_step):
            calls.append(index)
            return next_step()

        return middleware

    for index in range(3):
        register_call_middleware(make(index))
    run_with_middleware(dict)
    assert calls == [0, 1, 2]


def test_a_raising_middleware_propagates() -> None:
    """Swallowing an error here would report a failed call as a successful one."""
    register_call_middleware(lambda nxt: nxt())

    def explode(_next_step):
        raise RuntimeError("middleware failed")

    register_call_middleware(explode)
    with pytest.raises(RuntimeError, match="middleware failed"):
        run_with_middleware(dict)


def test_no_service_bypasses_the_hook_point() -> None:
    """Every service method must dispatch through ``run_tool_call``.

    A method that builds its own models and calls ``dispatch_tool_call`` directly still works,
    which is the problem: payload hooks never see that call, and the omission shows up as a
    tool that quietly ignores whatever the operator installed rather than as an error.
    """
    import ast
    import pathlib

    modal_root = pathlib.Path(__file__).resolve().parents[2] / "proto_tools" / "modal"
    offenders: list[str] = []
    for path in sorted(modal_root.rglob("*_service.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "dispatch_tool_call":
                offenders.append(f"{path.relative_to(modal_root)}:{node.lineno}")

    assert not offenders, (
        f"Service methods call dispatch_tool_call directly, bypassing payload hooks: {offenders}. "
        f"Use run_tool_call(run_fn, InputModel, ConfigModel, input_dict, config_dict) instead."
    )


def test_run_tool_call_applies_payload_hooks_before_validation() -> None:
    """The whole point of centralizing: a hook must be able to fix a value validation rejects."""
    from proto_tools.modal.utils import run_tool_call
    from proto_tools.utils.base_config import BaseConfig
    from proto_tools.utils.tool_io import BaseToolInput

    class _Input(BaseToolInput):
        sequence: str

    class _Config(BaseConfig):
        pass

    def resolve(input_dict: dict[str, Any], _config: dict[str, Any]) -> None:
        input_dict["sequence"] = input_dict["sequence"].removeprefix("ref://")

    register_payload_hook(resolve)

    seen: dict[str, Any] = {}

    def fake_run(inputs: _Input, config: _Config, instance: Any = None) -> Any:
        seen["sequence"] = inputs.sequence
        raise RuntimeError("stop here — validation already happened")

    with pytest.raises(RuntimeError, match="stop here"):
        run_tool_call(fake_run, _Input, _Config, {"sequence": "ref://ACGT"}, {})
    assert seen["sequence"] == "ACGT", "the hook must run before the model is constructed"
