"""Extension points a worker exposes, and the guarantees they carry."""

from __future__ import annotations

from typing import Any

import pytest

from proto_tools.modal.hooks import (
    CallContext,
    apply_payload_hooks,
    clear_hooks,
    register_call_middleware,
    register_payload_hook,
    run_with_middleware,
)

_CTX = CallContext(run_fn=lambda: None)


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
    assert run_with_middleware(_CTX, lambda: {"ok": True}) == {"ok": True}


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

    def timing(_ctx, next_step):
        events.append("before")
        result = next_step()
        events.append("after")
        return result

    register_call_middleware(timing)
    assert run_with_middleware(_CTX, lambda: {"value": 1}) == {"value": 1}
    assert events == ["before", "after"]


def test_middleware_can_transform_the_result() -> None:
    """A large field may need moving elsewhere before the transport sees it."""
    register_call_middleware(lambda _ctx, nxt: {**nxt(), "added": True})
    assert run_with_middleware(_CTX, lambda: {"value": 1}) == {"value": 1, "added": True}


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

    def with_capture(_ctx, next_step):
        with capture():
            return next_step()

    register_call_middleware(with_capture)
    run_with_middleware(_CTX, lambda: {"ok": True})
    assert entered == ["open", "close"]


def test_the_first_registered_middleware_is_outermost() -> None:
    """Documented ordering. Reversing it would silently invert nesting for anyone relying on it."""
    order: list[str] = []

    def outer(_ctx, next_step):
        order.append("outer-in")
        result = next_step()
        order.append("outer-out")
        return result

    def inner(_ctx, next_step):
        order.append("inner-in")
        result = next_step()
        order.append("inner-out")
        return result

    register_call_middleware(outer)
    register_call_middleware(inner)
    run_with_middleware(_CTX, dict)
    assert order == ["outer-in", "inner-in", "inner-out", "outer-out"]


def test_each_middleware_binds_its_own_next_step() -> None:
    """Building the chain in a loop invites late binding, where every layer calls the last one."""
    calls: list[int] = []

    def make(index: int):
        def middleware(_ctx, next_step):
            calls.append(index)
            return next_step()

        return middleware

    for index in range(3):
        register_call_middleware(make(index))
    run_with_middleware(_CTX, dict)
    assert calls == [0, 1, 2]


def test_a_raising_middleware_propagates() -> None:
    """Swallowing an error here would report a failed call as a successful one."""
    register_call_middleware(lambda _ctx, nxt: nxt())

    def explode(_ctx, _next_step):
        raise RuntimeError("middleware failed")

    register_call_middleware(explode)
    with pytest.raises(RuntimeError, match="middleware failed"):
        run_with_middleware(_CTX, dict)


def test_no_service_bypasses_the_hook_point() -> None:
    """Every service method must dispatch through ``run_tool_call``.

    Asserted positively rather than by banning ``dispatch_tool_call``: calling the run function
    or the envelope directly bypasses hooks just as effectively, and a bypass shows up as a tool
    quietly ignoring whatever the operator installed rather than as an error.
    """
    import ast
    import pathlib

    modal_root = pathlib.Path(__file__).resolve().parents[2] / "proto_tools" / "modal"
    offenders: list[str] = []
    checked = 0
    for path in sorted(modal_root.rglob("*_service.py")):
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
            if not any(getattr(d, "attr", None) == "method" for d in decorators):
                continue
            checked += 1
            called = {
                inner.func.id if isinstance(inner.func, ast.Name) else getattr(inner.func, "attr", None)
                for inner in ast.walk(node)
                if isinstance(inner, ast.Call)
            }
            if "run_tool_call" not in called:
                offenders.append(f"{path.relative_to(modal_root)}:{node.lineno} ({node.name})")

    assert checked, "found no service methods to check — the scan itself is broken"
    assert not offenders, (
        f"Service methods do not dispatch through run_tool_call, so hooks never see them: {offenders}. "
        f"Use run_tool_call(run_fn, InputModel, ConfigModel, input_dict, config_dict)."
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


def test_middleware_cleanup_runs_when_the_tool_raises() -> None:
    """The guarantee that matters for a context-manager middleware, and the one a success cannot show."""
    import contextlib

    entered: list[str] = []

    @contextlib.contextmanager
    def capture():
        entered.append("open")
        try:
            yield
        finally:
            entered.append("close")

    def with_capture(_ctx, next_step):
        with capture():
            return next_step()

    register_call_middleware(with_capture)

    def explode() -> dict[str, Any]:
        raise RuntimeError("the tool failed")

    with pytest.raises(RuntimeError, match="the tool failed"):
        run_with_middleware(_CTX, explode)
    assert entered == ["open", "close"], "cleanup must run when the wrapped call raises"


def test_a_middleware_that_forgets_to_return_is_named() -> None:
    """Otherwise the None travels to the client and fails somewhere pointing at no middleware."""

    def forgets(_ctx, next_step) -> None:
        next_step()

    register_call_middleware(forgets)
    with pytest.raises(TypeError, match="middleware returned NoneType"):
        run_with_middleware(_CTX, lambda: {"ok": True})


def test_middleware_is_told_which_tool_it_wrapped() -> None:
    """Timings and destinations need naming; the run function alone is not a stable label."""
    from proto_tools.modal.hooks import CallContext as _CallContext
    from proto_tools.tools.masked_models.esm2.esm2_embeddings import run_esm2_embeddings

    assert _CallContext(run_esm2_embeddings).tool_key == "esm2-embedding"
    assert _CallContext(lambda: None).tool_key is None, "an unregistered function resolves to nothing"
