"""Tests for the ToolInstance auto-persist overlay and @tool wrapper integration."""

import threading
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils import BaseConfig, ConfigField
from proto_tools.utils.tool_instance import (
    ToolInstance,
    _auto_persist_overlay,
    _instances,
    _persist_mode,
    _scope_override,
)
from proto_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_fake_instance(toolkit: str = "pyrosetta") -> ToolInstance:
    """Construct a ToolInstance bypassing __init__ so no subprocess is spawned."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.toolkit = toolkit
    inst.device = "cpu"
    inst._instance_lock = threading.RLock()
    inst._worker = None
    inst._cache_keys = set()
    inst._reload_params = {}
    inst._env_ready = True
    inst._tool_env_vars = {"passthrough": [], "set": []}
    return inst


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_contextvars() -> Generator[None, None, None]:
    """Reset all ContextVar + module-level state to a clean baseline per test."""
    _instances.clear()
    ToolInstance._build_failures.clear()
    _auto_persist_overlay.set(None)
    _scope_override.set(None)
    _persist_mode.set(False)
    yield
    _auto_persist_overlay.set(None)
    _scope_override.set(None)
    _persist_mode.set(False)
    for inst in list(_instances.values()):
        inst.shutdown()
    _instances.clear()
    ToolInstance._build_failures.clear()


class _AutoPersistInput(BaseToolInput):
    """Mock input for auto-persist wrapper tests."""

    input_data: str = Field(description="Input data")


class _AutoPersistOutput(MockToolOutputBase):
    """Mock output for auto-persist wrapper tests."""

    result: str = Field(default="", description="Result string")


class _PreprocessConfig(BaseConfig):
    """Config with a custom preprocess hook (triggers auto-persist)."""

    recorded_overlay_has_toolkit: bool = ConfigField(
        default=False,
        description="Set by preprocess() to the presence of the toolkit in the overlay when called",
        include_in_key=False,
    )

    def preprocess(self, inputs: BaseToolInput) -> BaseToolInput:
        """Mark the input with whether the auto-persist overlay was seeded for 'pyrosetta'."""
        overlay = _auto_persist_overlay.get() or {}
        marker = "OVERLAY_HAD_TOOLKIT" if "pyrosetta" in overlay else "NO_OVERLAY"
        assert isinstance(inputs, _AutoPersistInput)
        return inputs.model_copy(update={"input_data": f"{marker}:{inputs.input_data}"})


class _PlainConfig(BaseConfig):
    """Config with no preprocess override."""

    param: str = ConfigField(default="x", description="Dummy")


@pytest.fixture
def _clean_registry() -> Generator[type[ToolRegistry], None, None]:
    """Save/restore the real registry around a test so we can register fakes."""
    original = ToolRegistry._registry.copy()
    ToolRegistry._registry.clear()
    yield ToolRegistry
    ToolRegistry._registry = original


# ── _auto_persist_scope direct tests ─────────────────────────────────────────


def test_auto_persist_scope_noop_when_no_preprocess() -> None:
    """With no existing overlay: None-instance creates+shuts down; provided instance is preserved."""
    # Case A: no instance passed — scope creates a fresh ToolInstance and shuts it down.
    with patch.object(ToolInstance, "__init__", return_value=None) as mock_init:
        with patch.object(ToolInstance, "shutdown") as mock_shutdown:
            with ToolInstance._auto_persist_scope("pyrosetta"):
                overlay = _auto_persist_overlay.get()
                assert overlay is not None
                assert "pyrosetta" in overlay
            # Scope exited: shutdown must have been called on the created instance.
            assert mock_init.call_count == 1
            assert mock_shutdown.call_count == 1
    assert _auto_persist_overlay.get() is None

    # Case B: caller passes own instance — scope seeds it but does NOT shut it down.
    my_inst = _make_fake_instance(toolkit="pyrosetta")
    with patch.object(ToolInstance, "shutdown") as mock_shutdown:
        with ToolInstance._auto_persist_scope("pyrosetta", instance=my_inst):
            overlay = _auto_persist_overlay.get()
            assert overlay is not None
            assert overlay["pyrosetta"] is my_inst
        # Caller-owned instance must NOT be shut down.
        assert mock_shutdown.call_count == 0
    assert _auto_persist_overlay.get() is None


def test_auto_persist_scope_nesting_safe_via_overlay() -> None:
    """If the outer overlay already contains the toolkit, the inner scope is a no-op."""
    outer_inst = _make_fake_instance(toolkit="pyrosetta")
    token = _auto_persist_overlay.set({"pyrosetta": outer_inst})
    try:
        with patch.object(ToolInstance, "__init__", return_value=None) as mock_init:
            with ToolInstance._auto_persist_scope("pyrosetta"):
                overlay = _auto_persist_overlay.get()
                assert overlay is not None
                # Inner scope did NOT overwrite the outer entry.
                assert overlay["pyrosetta"] is outer_inst
            # No new instance was created.
            assert mock_init.call_count == 0
    finally:
        _auto_persist_overlay.reset(token)


def test_auto_persist_scope_nesting_safe_via_shared_cache() -> None:
    """If the shared _active_cache() already contains the toolkit, scope is a no-op."""
    shared_inst = _make_fake_instance(toolkit="pyrosetta")
    _instances["pyrosetta"] = shared_inst
    try:
        with patch.object(ToolInstance, "__init__", return_value=None) as mock_init:
            with ToolInstance._auto_persist_scope("pyrosetta"):
                # The shared cache should still hold the original instance; the overlay
                # should NOT have been seeded (bail-out branch).
                assert _auto_persist_overlay.get() is None
            assert mock_init.call_count == 0
    finally:
        _instances.pop("pyrosetta", None)


def test_auto_persist_scope_mismatch_raises() -> None:
    """Passing an instance whose toolkit != the outer toolkit must raise ValueError."""
    wrong_inst = _make_fake_instance(toolkit="esm2")
    with pytest.raises(ValueError, match=r"toolkit mismatch"):
        with ToolInstance._auto_persist_scope("pyrosetta", instance=wrong_inst):
            pass  # pragma: no cover -- scope body shouldn't run
    # Error message mentions both toolkits.
    try:
        with ToolInstance._auto_persist_scope("pyrosetta", instance=wrong_inst):
            pass  # pragma: no cover
    except ValueError as exc:
        assert "esm2" in str(exc)
        assert "pyrosetta" in str(exc)


# ── dispatch overlay integration ─────────────────────────────────────────────


def test_dispatch_checks_overlay_before_shared_cache() -> None:
    """dispatch() Path 2 should consult the overlay first and invoke its instance's .run()."""
    overlay_inst = _make_fake_instance(toolkit="pyrosetta")
    overlay_inst.run = MagicMock(return_value={"result": "from-overlay"})  # type: ignore[method-assign]

    # Also seed the shared cache with a DIFFERENT instance under the same key — the
    # overlay must win.
    shared_inst = _make_fake_instance(toolkit="pyrosetta")
    shared_inst.run = MagicMock(return_value={"result": "from-shared"})  # type: ignore[method-assign]
    _instances["pyrosetta"] = shared_inst

    token = _auto_persist_overlay.set({"pyrosetta": overlay_inst})
    try:
        result = ToolInstance.dispatch("pyrosetta", {"op": "score"})
        assert result == {"result": "from-overlay"}
        overlay_inst.run.assert_called_once()
        shared_inst.run.assert_not_called()
    finally:
        _auto_persist_overlay.reset(token)
        _instances.pop("pyrosetta", None)


def test_dispatch_overlay_miss_falls_through() -> None:
    """If the overlay holds a different key, dispatch falls through to the one-shot path."""
    other_inst = _make_fake_instance(toolkit="pyrosetta")
    other_inst.run = MagicMock()  # type: ignore[method-assign]

    token = _auto_persist_overlay.set({"pyrosetta": other_inst})
    try:
        with patch.object(ToolInstance, "_oneshot", return_value={"result": "oneshot"}) as mock_oneshot:
            result = ToolInstance.dispatch("esm2", {"op": "score"})
        assert result == {"result": "oneshot"}
        mock_oneshot.assert_called_once()
        # The mismatched overlay entry must not have been used.
        other_inst.run.assert_not_called()
    finally:
        _auto_persist_overlay.reset(token)


# ── Concurrency: ContextVar isolation ────────────────────────────────────────


def test_concurrent_threads_isolated_overlays() -> None:
    """Two threads inside their own _auto_persist_scope must see their own instance only."""
    barrier = threading.Barrier(2)
    results: dict[str, Any] = {}

    def worker(name: str) -> None:
        inst = _make_fake_instance(toolkit="pyrosetta")
        with ToolInstance._auto_persist_scope("pyrosetta", instance=inst):
            # Synchronize so both threads are simultaneously inside the scope.
            barrier.wait(timeout=5)
            overlay = _auto_persist_overlay.get()
            assert overlay is not None
            seen = overlay["pyrosetta"]
            results[name] = (inst, seen)

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(worker, "A"), ex.submit(worker, "B")]
        for f in futures:
            f.result(timeout=10)

    inst_a, seen_a = results["A"]
    inst_b, seen_b = results["B"]
    # Each thread sees its own instance.
    assert seen_a is inst_a
    assert seen_b is inst_b
    # And crucially: the two threads' instances are different objects.
    assert inst_a is not inst_b
    # After both scopes exited the ContextVar in the main thread is still clean.
    assert _auto_persist_overlay.get() is None


# ── @tool wrapper integration ────────────────────────────────────────────────


def _override_wrapper_source_file(wrapped_fn: Any, new_source_file: Path) -> None:
    """Mutate the ``source_file`` closure cell of a @tool-wrapped function.

    The @tool decorator captures ``source_file`` from the call stack at decoration
    time. In tests we want to pretend the tool lives in a different directory (one
    with a real ``standalone/`` dir) so the wrapper's auto-persist branch fires.
    """
    body = wrapped_fn.__closure__[0].cell_contents
    freevars = body.__code__.co_freevars
    idx = freevars.index("source_file")
    body.__closure__[idx].cell_contents = new_source_file


def test_atool_wrapper_triggers_auto_persist_for_custom_preprocess(
    _clean_registry: type[ToolRegistry],
) -> None:
    """When a tool has a custom preprocess AND has_standalone_env=True, the wrapper opens a scope."""

    @_clean_registry.register(
        key="auto-persist-preprocess-test",
        label="Auto Persist Preprocess Test",
        category="test",
        input_class=_AutoPersistInput,
        config_class=_PreprocessConfig,
        output_class=_AutoPersistOutput,
        description="Triggers auto-persist via custom preprocess",
    )
    def run_tool(inputs: _AutoPersistInput, config: _PreprocessConfig, instance: Any = None) -> _AutoPersistOutput:
        # Inside the wrapped function body — overlay should still be seeded.
        overlay = _auto_persist_overlay.get() or {}
        marker = "BODY_OVERLAY_HIT" if "pyrosetta" in overlay else "BODY_OVERLAY_MISS"
        return _AutoPersistOutput(result=f"{inputs.input_data}|{marker}")

    # Point the wrapper's source_file (closure) and the spec's source_file at the real
    # pyrosetta directory, which has a ``standalone/`` subdir — so both
    # ``source_file.parent.name`` -> "pyrosetta" AND ``spec.has_standalone_env`` -> True
    # without needing to patch a pydantic computed property.
    pyrosetta_source = (
        Path(__file__).resolve().parents[2]
        / "proto_tools"
        / "tools"
        / "structure_scoring"
        / "pyrosetta"
        / "fake_tool_for_test.py"
    )
    assert (pyrosetta_source.parent / "standalone").is_dir(), "pyrosetta/standalone/ must exist for this test"

    spec = _clean_registry.get("auto-persist-preprocess-test")
    spec.source_file = pyrosetta_source
    _override_wrapper_source_file(run_tool, pyrosetta_source)

    # Stub out ToolInstance construction — the scope will create + shut down one
    # since we pass no instance.
    with patch.object(ToolInstance, "__init__", return_value=None):
        with patch.object(ToolInstance, "shutdown"):
            result = run_tool(_AutoPersistInput(input_data="hello"), _PreprocessConfig())

    # preprocess saw the overlay (the scope was opened around it).
    assert "OVERLAY_HAD_TOOLKIT:hello" in result.result
    # Function body also saw the overlay (scope still active during dispatch).
    assert "BODY_OVERLAY_HIT" in result.result


def test_atool_wrapper_skips_scope_without_preprocess_or_instance(
    _clean_registry: type[ToolRegistry],
) -> None:
    """Default preprocess + no instance -> overlay stays None throughout the call."""
    observed: dict[str, Any] = {}

    @_clean_registry.register(
        key="auto-persist-noop-test",
        label="Auto Persist Noop Test",
        category="test",
        input_class=_AutoPersistInput,
        config_class=_PlainConfig,
        output_class=_AutoPersistOutput,
        description="No preprocess override, no explicit instance — scope should be skipped",
    )
    def run_tool(inputs: _AutoPersistInput, config: _PlainConfig, instance: Any = None) -> _AutoPersistOutput:
        observed["overlay"] = _auto_persist_overlay.get()
        return _AutoPersistOutput(result=inputs.input_data)

    # Even with has_standalone_env=True, a tool that has no custom preprocess AND no
    # instance= argument must NOT open a scope. Point the spec+closure at the real
    # pyrosetta/ dir (which has standalone/) so has_standalone_env=True — proving
    # the short-circuit depends on the preprocess/instance gate, not env presence.
    pyrosetta_source = (
        Path(__file__).resolve().parents[2]
        / "proto_tools"
        / "tools"
        / "structure_scoring"
        / "pyrosetta"
        / "fake_tool_for_test.py"
    )
    spec = _clean_registry.get("auto-persist-noop-test")
    spec.source_file = pyrosetta_source
    _override_wrapper_source_file(run_tool, pyrosetta_source)

    result = run_tool(_AutoPersistInput(input_data="ping"), _PlainConfig())

    assert result.result == "ping"
    assert observed["overlay"] is None
