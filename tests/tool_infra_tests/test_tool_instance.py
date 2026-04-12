"""tests/tool_infra_tests/test_tool_instance.py.

Tests for ToolInstance.
"""

import contextlib
import hashlib
import logging
import subprocess
import sys
import tempfile
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_tools.utils.tool_instance import (
    ToolInstance,
    _active_cache,
    _instances,
    _persist_mode,
    _scope_override,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_fake_instance(
    tool_name: str = "esm2",
    device: str = "cpu",
    needs_warmup: bool = False,
    _tmp_dir: Path | None = None,
) -> ToolInstance:
    """Create a ToolInstance with fake paths, bypassing __init__.

    Parameters
    ----------
    tool_name : str
        Name of the tool
    device : str
        Device to use
    needs_warmup : bool
        If False (default), pre-mark the default config as warmed up,
        which disables warmup timeout. Set to True to test warmup behavior.
    _tmp_dir : Path | None
        If provided, use this as env_path (for marker file tests).
        Otherwise creates a temp directory.
    """
    inst = ToolInstance.__new__(ToolInstance)
    inst.tool_name = tool_name
    inst.device = device
    if _tmp_dir is not None:
        inst.env_path = _tmp_dir
    else:
        inst.env_path = Path(tempfile.mkdtemp(prefix="test_toolinstance_"))
    inst.script_path = Path("/fake/inference.py")
    inst._tool_env_vars = {"passthrough": [], "set": []}
    inst._env_ready = True
    inst._cache_keys = set()
    inst._instance_lock = threading.Lock()
    inst._worker = None
    inst._reload_params = {}

    # Pre-mark default config as warmed up unless testing warmup behavior
    if not needs_warmup:
        inst._mark_warmup_complete({})

    return inst


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_instance_cache():
    """Ensure the singleton cache and build-failure set are clean for each test."""
    _instances.clear()
    ToolInstance._build_failures.clear()
    _scope_override.set(None)
    _persist_mode.set(False)
    yield
    _scope_override.set(None)
    _persist_mode.set(False)
    # Also stop any workers that were created
    for inst in list(_instances.values()):
        inst.shutdown()
    _instances.clear()
    ToolInstance._build_failures.clear()


# ── Singleton factory tests ─────────────────────────────────────────────────


@patch.object(ToolInstance, "__init__", return_value=None)
def test_get_returns_same_instance(mock_init: MagicMock):
    """ToolInstance.get() should return the same instance for same args."""
    inst1 = ToolInstance.get("esm2")
    inst2 = ToolInstance.get("esm2")
    assert inst1 is inst2
    assert mock_init.call_count == 1  # Only constructed once


@patch.object(ToolInstance, "__init__", return_value=None)
def test_default_key_is_tool_name(mock_init: MagicMock):
    """Without instance_name, cache key should be tool_name."""
    ToolInstance.get("esm2")
    assert "esm2" in _instances
    assert len(_instances) == 1


@patch.object(ToolInstance, "__init__", return_value=None)
def test_get_different_tool_creates_new(mock_init: MagicMock):
    """Different tool name should create a new instance."""
    inst1 = ToolInstance.get("esm2")
    inst2 = ToolInstance.get("blast")
    assert inst1 is not inst2
    assert mock_init.call_count == 2


@patch.object(ToolInstance, "__init__", return_value=None)
def test_explicit_instance_name_creates_named_instance(mock_init: MagicMock):
    """Explicit instance_name caches under that name."""
    inst = ToolInstance.get("esm2", instance_name="my-esm2")
    assert "my-esm2" in _instances
    assert inst is _instances["my-esm2"]
    # Default key should not exist
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_same_instance_name_returns_same_instance(mock_init: MagicMock):
    """Two calls with the same instance_name return the same object."""
    inst1 = ToolInstance.get("esm2", instance_name="worker-1")
    inst2 = ToolInstance.get("esm2", instance_name="worker-1")
    assert inst1 is inst2
    assert mock_init.call_count == 1


@patch.object(ToolInstance, "__init__", return_value=None)
def test_different_instance_names_create_separate_instances(mock_init: MagicMock):
    """Different instance_name values create separate instances."""
    inst1 = ToolInstance.get("esm2", instance_name="worker-1")
    inst2 = ToolInstance.get("esm2", instance_name="worker-2")
    assert inst1 is not inst2
    assert mock_init.call_count == 2
    assert "worker-1" in _instances
    assert "worker-2" in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_named_and_default_are_independent(mock_init: MagicMock):
    """A named instance and the default instance are separate objects."""
    default = ToolInstance.get("esm2")
    named = ToolInstance.get("esm2", instance_name="my-esm2")
    assert default is not named
    assert "esm2" in _instances
    assert "my-esm2" in _instances


def test_clear_all():
    """clear_all() should empty the cache."""
    with patch.object(ToolInstance, "__init__", return_value=None):
        ToolInstance.get("esm2")
        assert len(_instances) == 1
        ToolInstance.clear_all()
        assert len(_instances) == 0


# ── Tool name validation tests ──────────────────────────────────────────────


def test_valid_tool_name():
    """Known tool names should validate."""
    assert ToolInstance._validate_tool_name("esm2") == "esm2"
    assert ToolInstance._validate_tool_name("blast") == "blast"


def test_invalid_tool_name():
    """Unknown tool names should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid tool name"):
        ToolInstance._validate_tool_name("nonexistent_tool_xyz")


# ── Script discovery tests ──────────────────────────────────────────────────


def test_find_setup_script():
    """Should find setup.sh for known tools."""
    path = ToolInstance._find_setup_script("esm2")
    assert path.name == "setup.sh"
    assert path.exists()


def test_find_script_inference():
    """Tools with inference.py should find it."""
    path = ToolInstance._find_script("esm2")
    assert path.name == "inference.py"
    assert path.exists()


def test_find_script_run():
    """Tools with run.py should find it."""
    path = ToolInstance._find_script("blast")
    assert path.name == "run.py"
    assert path.exists()


def test_find_script_nonexistent():
    """Should raise for nonexistent tools."""
    with pytest.raises(ValueError, match="No standalone script found"):
        ToolInstance._find_script("nonexistent_tool_xyz")


# ── run() method tests ──────────────────────────────────────────────────────


@patch.object(ToolInstance, "_run_persistent")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_run_uses_persistent(mock_init: MagicMock, mock_persistent: MagicMock):
    """run() should delegate to _run_persistent."""
    mock_persistent.return_value = {"result": "ok"}

    inst = ToolInstance.get("esm2")
    inst.script_path = Path("/fake/inference.py")
    inst._instance_lock = threading.Lock()
    result = inst.run({"operation": "score"})

    assert result == {"result": "ok"}
    mock_persistent.assert_called_once()


# ── dispatch() tests ────────────────────────────────────────────────────────


@patch.object(ToolInstance, "_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_runs_oneshot_when_no_cache(mock_init: MagicMock, mock_oneshot: MagicMock):
    """dispatch() should use _oneshot when no cached instance exists."""
    mock_oneshot.return_value = {"result": "ok"}
    result = ToolInstance.dispatch("esm2", {"op": "score", "device": "cuda"})
    assert result == {"result": "ok"}
    mock_oneshot.assert_called_once_with(
        "esm2",
        {"op": "score", "device": "cuda"},
        script_path=None,
        verbose=False,
        timeout=600,
    )


@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_uses_cached_instance(mock_init: MagicMock):
    """dispatch() should use a cached persistent instance when available."""
    inst = ToolInstance.get("esm2")
    inst.run = MagicMock(return_value={"result": "cached"})

    result = ToolInstance.dispatch("esm2", {"op": "score", "device": "cuda"})
    assert result == {"result": "cached"}
    inst.run.assert_called_once_with(
        {"op": "score", "device": "cuda"},
        script_path=None,
        verbose=False,
        timeout=600,
        reload_on=None,
    )


@patch.object(ToolInstance, "_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_respects_instance_string_key(mock_init: MagicMock, mock_oneshot: MagicMock):
    """dispatch() should look up by string instance key when provided."""
    mock_oneshot.return_value = {"result": "oneshot"}

    # Cache under "esm2" (default key), but dispatch with custom key
    inst = ToolInstance.get("esm2")
    inst.run = MagicMock()

    result = ToolInstance.dispatch("esm2", {"op": "score"}, instance="other-key")
    # Should NOT use cached "esm2" — key mismatch
    assert result == {"result": "oneshot"}
    inst.run.assert_not_called()


def test_dispatch_with_tool_instance_object():
    """dispatch() should use a ToolInstance object directly when passed."""
    inst = _make_fake_instance()
    inst.run = MagicMock(return_value={"result": "direct"})

    result = ToolInstance.dispatch("esm2", {"op": "score"}, instance=inst)
    assert result == {"result": "direct"}
    inst.run.assert_called_once_with(
        {"op": "score"},
        script_path=None,
        verbose=False,
        timeout=600,
        reload_on=None,
    )


@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_passes_script_path(mock_init: MagicMock):
    """dispatch() should forward script_path and config-derived args to cached instance."""
    from proto_tools.utils.base_config import BaseConfig

    inst = ToolInstance.get("esm2")
    inst.run = MagicMock(return_value={"result": "ok"})

    cfg = BaseConfig(verbose=True)
    ToolInstance.dispatch("esm2", {}, script_path="/custom/script.py", config=cfg)
    inst.run.assert_called_once_with(
        {},
        script_path="/custom/script.py",
        verbose=True,
        timeout=600,
        reload_on=set(),
    )


# ── _oneshot() tests ────────────────────────────────────────────────────────


@patch.object(ToolInstance, "_run_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_oneshot_does_not_cache(mock_init: MagicMock, mock_run: MagicMock):
    """_oneshot() should not leave anything in _instances."""
    mock_run.return_value = {"result": "ok"}
    # Need script_path set for _oneshot to work
    with patch.object(ToolInstance, "script_path", Path("/fake/inference.py"), create=True):
        ToolInstance._oneshot("esm2", {"op": "score"})
    assert len(_instances) == 0


@patch.object(ToolInstance, "_run_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_oneshot_calls_run_oneshot(mock_init: MagicMock, mock_run: MagicMock):
    """_oneshot() should call _run_oneshot, not _run_persistent."""
    mock_run.return_value = {"result": "ephemeral"}
    with patch.object(ToolInstance, "script_path", Path("/fake/inference.py"), create=True):
        result = ToolInstance._oneshot("esm2", {"op": "score"})
    assert result == {"result": "ephemeral"}
    mock_run.assert_called_once()


def test_oneshot_injects_tool_env_path():
    """_run_oneshot() should set TOOL_VENV_PATH in the subprocess env."""
    inst = _make_fake_instance()

    with patch(
        "proto_tools.utils.tool_instance.subprocess.run",
    ) as mock_run:
        mock_run.return_value = MagicMock()
        # Will fail on output read, but we only care about the env arg
        with contextlib.suppress(Exception):
            inst._run_oneshot(
                {"op": "score"},
                script_path=Path("/fake/inference.py"),
            )

        env = mock_run.call_args.kwargs["env"]
        assert env["TOOL_VENV_PATH"] == str(inst.env_path)


# ── persist_tool() tests ───────────────────────────────────────────────────


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_creates_cached_instance(mock_init: MagicMock):
    """Instance should be in _instances during the block."""
    with ToolInstance.persist_tool("esm2") as inst:
        assert "esm2" in _instances
        assert _instances["esm2"] is inst


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_cleans_up_on_exit(mock_init: MagicMock):
    """Instance should be removed from _instances after the block."""
    with ToolInstance.persist_tool("esm2"):
        pass
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_cleans_up_on_exception(mock_init: MagicMock):
    """Instance should be removed even if an exception occurs."""
    with pytest.raises(RuntimeError, match="boom"), ToolInstance.persist_tool("esm2"):
        raise RuntimeError("boom")
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_calls_shutdown(mock_init: MagicMock):
    """Worker stop() should be called on exit."""
    with ToolInstance.persist_tool("esm2") as inst:
        mock_worker = MagicMock()
        inst._worker = mock_worker
    mock_worker.stop.assert_called_once()


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_evicts_from_cache(mock_init: MagicMock):
    """shutdown() should remove itself from _instances."""
    inst = ToolInstance.get("esm2")
    assert "esm2" in _instances
    inst.shutdown()
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_with_instance_name(mock_init: MagicMock):
    """Custom instance_name should be used as cache key."""
    with ToolInstance.persist_tool("esm2", instance_name="my-esm2") as inst:
        assert "my-esm2" in _instances
        assert "esm2" not in _instances
        assert _instances["my-esm2"] is inst
    assert "my-esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_anonymous_caches_when_slot_open(mock_init: MagicMock):
    """Single anonymous persistent should cache under tool_name."""
    with ToolInstance.persist_tool("esm2") as inst:
        assert "esm2" in _instances
        assert _instances["esm2"] is inst
        # Implicit dispatch should find it
        inst.run = MagicMock(return_value={"result": "cached"})
        result = ToolInstance.dispatch("esm2", {"op": "score"})
        assert result == {"result": "cached"}
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_anonymous_skips_cache_when_slot_taken(mock_init: MagicMock):
    """Second anonymous persistent for same tool should not cache."""
    with ToolInstance.persist_tool("esm2") as inst_a:
        assert "esm2" in _instances
        assert _instances["esm2"] is inst_a

        with ToolInstance.persist_tool("esm2") as inst_b:
            # inst_b should NOT be in the cache
            assert _instances["esm2"] is inst_a  # first one still there
            assert inst_b is not inst_a

    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persistent_anonymous_second_is_different_object(mock_init: MagicMock):
    """Two anonymous persistent instances are distinct objects."""
    with ToolInstance.persist_tool("esm2") as inst_a, ToolInstance.persist_tool("esm2") as inst_b:
        assert inst_a is not inst_b


@patch.object(ToolInstance, "_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_tool_does_not_cache_other_tools(mock_init: MagicMock, mock_oneshot: MagicMock):
    """dispatch() for a different tool should use one-shot, not auto-cache."""
    mock_oneshot.return_value = {"result": "oneshot"}

    with ToolInstance.persist_tool("esm2"):
        assert "esm2" in _instances

        # Dispatch a *different* tool — should NOT be cached
        result = ToolInstance.dispatch("blast", {"op": "search"})
        assert result == {"result": "oneshot"}
        mock_oneshot.assert_called_once()
        assert "blast" not in _instances

        # The named tool is still the only cached one
        assert list(_instances.keys()) == ["esm2"]


# ── scope() tests ──────────────────────────────────────────────────────────


@patch.object(ToolInstance, "__init__", return_value=None)
def test_scope_isolates_cache(mock_init: MagicMock):
    """Instances created inside scope() should not leak out."""
    ToolInstance.get("esm2")
    assert "esm2" in _instances

    with ToolInstance.scope():
        assert len(_active_cache()) == 0  # clean scoped cache
        ToolInstance.get("blast")
        assert "blast" in _active_cache()

    # After scope: blast gone, esm2 still in global
    assert "blast" not in _instances
    assert "esm2" in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_scope_calls_shutdown_on_exit(mock_init: MagicMock):
    """scope() should shutdown instances created inside on exit."""
    with ToolInstance.scope():
        inst = ToolInstance.get("esm2")
        assert "esm2" in _active_cache()
        mock_worker = MagicMock()
        inst._worker = mock_worker

    mock_worker.stop.assert_called_once()
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_scope_restores_on_exception(mock_init: MagicMock):
    """scope() should restore cache even if an exception occurs inside."""
    ToolInstance.get("esm2")

    with pytest.raises(RuntimeError, match="boom"), ToolInstance.scope():
        ToolInstance.get("blast")
        raise RuntimeError("boom")

    assert "esm2" in _instances
    assert "blast" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_scope_nestable(mock_init: MagicMock):
    """Nested scope() calls should work correctly."""
    ToolInstance.get("esm2")

    with ToolInstance.scope():
        ToolInstance.get("blast")

        with ToolInstance.scope():
            assert len(_active_cache()) == 0
            ToolInstance.get("chai1")
            assert "chai1" in _active_cache()

        # Inner scope exited: chai1 gone, blast restored
        assert "blast" in _active_cache()
        assert "chai1" not in _active_cache()

    # Outer scope exited: blast gone, esm2 still in global
    assert "esm2" in _instances
    assert "blast" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_scope_does_not_affect_other_threads(mock_init: MagicMock):
    """scope() on one thread should not affect another thread's cache."""
    barrier = threading.Barrier(2)
    other_thread_saw_global = [False]

    def other_thread():
        barrier.wait()
        # This thread is not inside any scope — should see _instances
        other_thread_saw_global[0] = _active_cache() is _instances

    ToolInstance.get("esm2")

    with ToolInstance.scope():
        t = threading.Thread(target=other_thread)
        t.start()
        barrier.wait()
        t.join()

    assert other_thread_saw_global[0]


# ── persist() tests (auto-cache everything mode) ───────────────────────────


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_auto_caches_on_dispatch(mock_init: MagicMock):
    """dispatch() inside persist() should auto-cache instead of one-shot."""
    with ToolInstance.persist():
        cache = _active_cache()
        assert len(cache) == 0

        with patch.object(ToolInstance, "run", return_value={"result": "ok"}):
            result = ToolInstance.dispatch("esm2", {"op": "score"})

        assert result == {"result": "ok"}
        assert "esm2" in cache


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_reuses_cached_instance(mock_init: MagicMock):
    """Second dispatch to same tool reuses the auto-cached instance."""
    with ToolInstance.persist():
        with patch.object(ToolInstance, "run", return_value={"r": 1}):
            ToolInstance.dispatch("esm2", {"op": "score"})
            ToolInstance.dispatch("esm2", {"op": "score"})

        cache = _active_cache()
        assert "esm2" in cache
        assert mock_init.call_count == 1


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_cleans_up_on_exit(mock_init: MagicMock):
    """All auto-cached instances should be shut down on block exit."""
    mock_workers = []
    with ToolInstance.persist():
        with patch.object(ToolInstance, "run", return_value={"r": 1}):
            ToolInstance.dispatch("esm2", {"op": "score"})
            ToolInstance.dispatch("blast", {"op": "search"})

        cache = _active_cache()
        for inst in cache.values():
            w = MagicMock()
            inst._worker = w
            mock_workers.append(w)

    for w in mock_workers:
        w.stop.assert_called_once()
    assert len(_instances) == 0


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_cleans_up_on_exception(mock_init: MagicMock):
    """Auto-cached instances should be cleaned up even on exception."""
    with pytest.raises(RuntimeError, match="boom"), ToolInstance.persist():
        with patch.object(ToolInstance, "run", return_value={"r": 1}):
            ToolInstance.dispatch("esm2", {"op": "score"})
        raise RuntimeError("boom")
    assert len(_instances) == 0
    assert not _persist_mode.get()


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_does_not_pollute_global_cache(mock_init: MagicMock):
    """Auto-cached instances should not appear in the global cache."""
    ToolInstance.get("blast")

    with ToolInstance.persist():
        with patch.object(ToolInstance, "run", return_value={"r": 1}):
            ToolInstance.dispatch("esm2", {"op": "score"})
        assert "esm2" in _active_cache()
        assert "blast" not in _active_cache()

    assert "blast" in _instances
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_nestable(mock_init: MagicMock):
    """Nested persist() blocks should each get their own scope."""
    with ToolInstance.persist(), patch.object(ToolInstance, "run", return_value={"r": 1}):
        ToolInstance.dispatch("esm2", {"op": "score"})

        with ToolInstance.persist():
            ToolInstance.dispatch("blast", {"op": "search"})
            assert "blast" in _active_cache()
            assert "esm2" not in _active_cache()

        assert "esm2" in _active_cache()
        assert "blast" not in _active_cache()

    assert len(_instances) == 0


@patch.object(ToolInstance, "_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_uses_oneshot_outside_persist(mock_init: MagicMock, mock_oneshot: MagicMock):
    """Without persist(), dispatch() should still use one-shot."""
    mock_oneshot.return_value = {"result": "ok"}
    ToolInstance.dispatch("esm2", {"op": "score"})
    mock_oneshot.assert_called_once()


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_mode_flag_reset_on_exit(mock_init: MagicMock):
    """_persist_mode should be False after the block exits."""
    assert not _persist_mode.get()
    with ToolInstance.persist():
        assert _persist_mode.get()
    assert not _persist_mode.get()


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_thread_isolation(mock_init: MagicMock):
    """persist() in one thread should not affect another thread."""
    barrier = threading.Barrier(2)
    other_thread_persist = [None]

    def other_thread():
        barrier.wait()
        other_thread_persist[0] = _persist_mode.get()

    with ToolInstance.persist():
        t = threading.Thread(target=other_thread)
        t.start()
        barrier.wait()
        t.join()

    assert other_thread_persist[0] is False


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_multiple_tools(mock_init: MagicMock):
    """Multiple different tools should each get auto-cached."""
    with ToolInstance.persist():
        with patch.object(ToolInstance, "run", return_value={"r": 1}):
            ToolInstance.dispatch("esm2", {"op": "score"})
            ToolInstance.dispatch("blast", {"op": "search"})
            ToolInstance.dispatch("esm2", {"op": "score"})

        cache = _active_cache()
        assert "esm2" in cache
        assert "blast" in cache
        assert len(cache) == 2
        assert mock_init.call_count == 2


@patch.object(ToolInstance, "__init__", return_value=None)
def test_persist_with_nested_persist_tool(mock_init: MagicMock):
    """persist_tool(tool_name) inside persist() should coexist correctly.

    The persist_tool instance lives in the persist scope's cache. When
    persist_tool() exits, its instance is cleaned up, but the auto-cached
    instance from persist() survives until the outer block exits.
    """
    with ToolInstance.persist(), patch.object(ToolInstance, "run", return_value={"r": 1}):
        # Auto-cached by persist mode
        ToolInstance.dispatch("esm2", {"op": "score"})
        assert "esm2" in _active_cache()

        # Explicit persistent for a different tool
        with ToolInstance.persist_tool("blast"):
            assert "blast" in _active_cache()
            ToolInstance.dispatch("blast", {"op": "search"})

        # persist_tool() exited — blast cleaned up, esm2 still alive
        assert "blast" not in _active_cache()
        assert "esm2" in _active_cache()

        # esm2 still works via persist auto-cache
        ToolInstance.dispatch("esm2", {"op": "score"})

    # persist() exited — everything cleaned up
    assert len(_instances) == 0


# ── shutdown_instance() tests ──────────────────────────────────────────────


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_instance_removes_instance(mock_init: MagicMock):
    """shutdown_instance() should remove the instance from the cache."""
    ToolInstance.get("esm2")
    assert "esm2" in _instances
    ToolInstance.shutdown_instance("esm2")
    assert "esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_instance_calls_shutdown(mock_init: MagicMock):
    """shutdown_instance() should call shutdown() on the evicted instance."""
    inst = ToolInstance.get("esm2")
    mock_worker = MagicMock()
    inst._worker = mock_worker
    ToolInstance.shutdown_instance("esm2")
    mock_worker.stop.assert_called_once()


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_instance_leaves_others(mock_init: MagicMock):
    """Shutting down one instance should not affect others."""
    ToolInstance.get("esm2")
    ToolInstance.get("blast")
    assert len(_instances) == 2
    ToolInstance.shutdown_instance("esm2")
    assert "esm2" not in _instances
    assert "blast" in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_instance_nonexistent_is_noop(mock_init: MagicMock):
    """shutdown_instance() on a missing key should not raise."""
    ToolInstance.shutdown_instance("nonexistent")  # should not raise


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_instance_with_named_key(mock_init: MagicMock):
    """shutdown_instance() should use the explicit cache key."""
    ToolInstance.get("esm2", instance_name="my-esm2")
    assert "my-esm2" in _instances
    ToolInstance.shutdown_instance("my-esm2")
    assert "my-esm2" not in _instances


@patch.object(ToolInstance, "__init__", return_value=None)
def test_shutdown_instance_then_get_creates_fresh(mock_init: MagicMock):
    """After shutdown_instance(), get() should create a fresh instance."""
    inst1 = ToolInstance.get("esm2")
    inst1._worker = None
    ToolInstance.shutdown_instance("esm2")
    inst2 = ToolInstance.get("esm2")
    assert inst1 is not inst2
    assert mock_init.call_count == 2


# ── Device restart tests ───────────────────────────────────────────────────


@patch.object(ToolInstance, "_oneshot")
def test_dispatch_forwards_input_dict_to_oneshot(mock_oneshot: MagicMock):
    """dispatch() should forward input_dict (with device) to _oneshot."""
    mock_oneshot.return_value = {"result": "ok"}
    ToolInstance.dispatch("esm2", {"op": "score", "device": "cuda"})
    mock_oneshot.assert_called_once_with(
        "esm2",
        {"op": "score", "device": "cuda"},
        script_path=None,
        verbose=False,
        timeout=600,
    )


@patch.object(ToolInstance, "_oneshot")
def test_dispatch_without_device_in_input_dict(mock_oneshot: MagicMock):
    """dispatch() without device in input_dict should still forward."""
    mock_oneshot.return_value = {"result": "ok"}
    ToolInstance.dispatch("blast", {"op": "search"})
    mock_oneshot.assert_called_once_with(
        "blast",
        {"op": "search"},
        script_path=None,
        verbose=False,
        timeout=600,
    )


# ── Reload-on-change tests ─────────────────────────────────────────────────


def test_persistent_worker_restarts_on_reload_param_change():
    """Changing a tracked reload param should restart the worker."""
    inst = _make_fake_instance(device="cpu")
    inst._reload_params = {
        "model_checkpoint": "esm2_t33_650M_UR50D",
    }

    mock_worker = MagicMock()
    mock_worker.script_path = inst.script_path
    inst._worker = mock_worker

    with patch("proto_tools.utils.tool_instance.PersistentWorker") as MockPW:
        new_worker = MagicMock()
        new_worker.send.return_value = {"result": "ok"}
        MockPW.return_value = new_worker

        result = inst._run_persistent(
            {"device": "cpu", "model_checkpoint": "esm2_t36_3B_UR50D"},
            reload_on={"model_checkpoint"},
        )

    mock_worker.stop.assert_called_once()
    MockPW.assert_called_once()
    assert inst._reload_params == {
        "model_checkpoint": "esm2_t36_3B_UR50D",
    }
    assert result == {"result": "ok"}


def test_persistent_worker_no_restart_same_reload_params():
    """Same reload params should reuse existing worker."""
    inst = _make_fake_instance(device="cpu")
    inst._reload_params = {
        "model_checkpoint": "esm2_t33_650M_UR50D",
    }

    mock_worker = MagicMock()
    mock_worker.script_path = inst.script_path
    mock_worker.send.return_value = {"result": "ok"}
    inst._worker = mock_worker

    result = inst._run_persistent(
        {"device": "cpu", "model_checkpoint": "esm2_t33_650M_UR50D"},
        reload_on={"model_checkpoint"},
    )

    mock_worker.stop.assert_not_called()
    assert result == {"result": "ok"}


@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_derives_reload_on_from_config(mock_init: MagicMock):
    """dispatch() should derive reload_on from config's reload_fields()."""
    from proto_tools.utils.base_config import BaseConfig, ConfigField

    class TestConfig(BaseConfig):
        model_checkpoint: str = ConfigField(
            default="default",
            description="model",
            reload_on_change=True,
        )

    inst = ToolInstance.get("esm2")
    inst.run = MagicMock(return_value={"result": "ok"})

    cfg = TestConfig()
    ToolInstance.dispatch(
        "esm2",
        {"op": "score", "device": "cpu"},
        config=cfg,
    )
    inst.run.assert_called_once_with(
        {"op": "score", "device": "cpu"},
        script_path=None,
        verbose=False,
        timeout=600,
        reload_on={"model_checkpoint"},
    )


# ── BaseConfig.reload_fields() tests ──────────────────────────────────────


def test_base_config_reload_fields_empty():
    """BaseConfig.reload_fields() should return empty set."""
    from proto_tools.utils.base_config import BaseConfig

    assert BaseConfig.reload_fields() == set()


def test_subclass_without_reload_fields():
    """Subclass without extra reload fields has empty reload_fields."""
    from proto_tools.utils.base_config import BaseConfig, ConfigField

    class MyConfig(BaseConfig):
        param: int = ConfigField(default=1, description="test")

    assert MyConfig.reload_fields() == set()


def test_subclass_with_reload_on_change():
    """Subclass with reload_on_change=True includes those fields."""
    from proto_tools.utils.base_config import BaseConfig, ConfigField

    class MyConfig(BaseConfig):
        model_checkpoint: str = ConfigField(
            default="default",
            description="model",
            reload_on_change=True,
        )

    assert MyConfig.reload_fields() == {"model_checkpoint"}


def test_reload_fields_excludes_non_reload():
    """Fields without reload_on_change are excluded."""
    from proto_tools.utils.base_config import BaseConfig, ConfigField

    class MyConfig(BaseConfig):
        reload_me: str = ConfigField(
            default="a",
            description="r",
            reload_on_change=True,
        )
        leave_me: str = ConfigField(default="b", description="l")

    fields = MyConfig.reload_fields()
    assert "reload_me" in fields
    assert "leave_me" not in fields


# ── Timeout tests ──────────────────────────────────────────────────────────


def test_oneshot_timeout_raises():
    """_run_oneshot() should convert subprocess.TimeoutExpired to TimeoutError."""
    inst = _make_fake_instance()

    with (
        patch(
            "proto_tools.utils.tool_instance.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=10),
        ),
        pytest.raises(TimeoutError, match="timed out after 10s"),
    ):
        inst._run_oneshot(
            {"op": "score"},
            script_path=Path("/fake/inference.py"),
            timeout=10,
        )


def test_persistent_timeout_raises():
    """_run_persistent() should propagate TimeoutError from the worker."""
    inst = _make_fake_instance()
    inst._reload_params = {}

    mock_worker = MagicMock()
    mock_worker.script_path = inst.script_path
    mock_worker.send.side_effect = TimeoutError("timed out")
    inst._worker = mock_worker

    with pytest.raises(TimeoutError, match="timed out"):
        inst._run_persistent({"device": "cpu"}, timeout=30)


@patch.object(ToolInstance, "_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_reads_timeout_from_config(mock_init: MagicMock, mock_oneshot: MagicMock):
    """dispatch() should use timeout from config object."""
    from proto_tools.utils.base_config import BaseConfig

    mock_oneshot.return_value = {"result": "ok"}
    cfg = BaseConfig(timeout=60)
    ToolInstance.dispatch("esm2", {"op": "score"}, config=cfg)
    mock_oneshot.assert_called_once_with(
        "esm2",
        {"op": "score"},
        script_path=None,
        verbose=False,
        timeout=60,
    )


@patch.object(ToolInstance, "_oneshot")
@patch.object(ToolInstance, "__init__", return_value=None)
def test_dispatch_defaults_timeout_to_600(mock_init: MagicMock, mock_oneshot: MagicMock):
    """dispatch() should default timeout to 600 when not in input_dict."""
    mock_oneshot.return_value = {"result": "ok"}
    ToolInstance.dispatch("esm2", {"op": "score"})
    mock_oneshot.assert_called_once_with(
        "esm2",
        {"op": "score"},
        script_path=None,
        verbose=False,
        timeout=600,
    )


def test_send_timeout_kills_worker():
    """PersistentWorker.send() should kill the worker on timeout."""
    from proto_tools.utils.persistent_worker import PersistentWorker

    worker = PersistentWorker.__new__(PersistentWorker)
    worker.tool_name = "esm2"
    worker._lock = __import__("threading").Lock()

    mock_process = MagicMock()
    mock_process.poll.return_value = None  # alive
    mock_process.pid = 999999  # safe fake PID (avoid MagicMock defaulting to int 1)
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stdout.fileno.return_value = 99
    worker._process = mock_process
    worker._stderr_lines = []

    with patch("proto_tools.utils.persistent_worker.select") as mock_sel:
        mock_sel.select.return_value = ([], [], [])  # timeout — nothing ready

        with pytest.raises(TimeoutError, match="timed out after 5s"):
            worker.send({"op": "score"}, timeout=5)

    # Worker should have been stopped (process set to None)
    assert worker._process is None


# ── Warmup timeout tests ──────────────────────────────────────────────────


def test_first_config_gets_warmup_timeout():
    """A never-seen config combination should get extended warmup timeout."""
    inst = _make_fake_instance(needs_warmup=True)
    params = {"model_name": "protenix_mini_esm_v0.5.0"}

    result = inst._apply_warmup_timeout(1200, params)
    assert result == 3600  # warmup timeout (60 min)


def test_seen_config_gets_normal_timeout():
    """A previously-completed config should get the normal timeout."""
    inst = _make_fake_instance(needs_warmup=True)
    params = {"model_name": "protenix_mini_esm_v0.5.0"}

    # Mark this config as completed
    inst._mark_warmup_complete(params)

    result = inst._apply_warmup_timeout(1200, params)
    assert result == 1200  # normal timeout


def test_different_configs_independent():
    """Different config combos should have independent warmup markers."""
    inst = _make_fake_instance(needs_warmup=True)
    params_a = {"model_name": "protenix_base_default_v1.0.0"}
    params_b = {"model_name": "protenix_mini_esm_v0.5.0"}

    # Mark config A as completed
    inst._mark_warmup_complete(params_a)

    # Config A should get normal timeout
    assert inst._apply_warmup_timeout(1200, params_a) == 1200
    # Config B should still get warmup timeout
    assert inst._apply_warmup_timeout(1200, params_b) == 3600


def test_empty_params_warmup():
    """Empty reload_params (no reload_on_change fields) should still track warmup."""
    inst = _make_fake_instance(needs_warmup=True)

    # First run with empty params should get warmup
    assert inst._apply_warmup_timeout(600, {}) == 3600

    # After marking complete, should get normal timeout
    inst._mark_warmup_complete({})
    assert inst._apply_warmup_timeout(600, {}) == 600


def test_warmup_timeout_is_at_least_configured():
    """Warmup timeout should be max(WARMUP, configured), not always WARMUP."""
    inst = _make_fake_instance(needs_warmup=True)
    params = {"model_name": "big_model"}

    # If configured timeout is larger than warmup, use the larger one
    result = inst._apply_warmup_timeout(5000, params)
    assert result == 5000


def test_warmup_with_none_timeout():
    """When timeout=None, warmup should still apply the WARMUP_TIMEOUT."""
    inst = _make_fake_instance(needs_warmup=True)
    params = {"model_name": "some_model"}

    result = inst._apply_warmup_timeout(None, params)
    assert result == 3600


def test_no_params_defaults_to_empty():
    """Calling _apply_warmup_timeout without reload_params should use empty dict."""
    inst = _make_fake_instance(needs_warmup=True)

    # No reload_params arg -> uses {} -> first run
    result = inst._apply_warmup_timeout(600)
    assert result == 3600


def test_marker_deterministic():
    """Same config params should produce the same marker path."""
    inst = _make_fake_instance()
    params = {"model_name": "esm_v1", "checkpoint": "ckpt_a"}
    p1 = inst._config_marker_path(params)
    p2 = inst._config_marker_path(params)
    assert p1 == p2


def test_marker_different_for_different_configs():
    """Different config params should produce different marker paths."""
    inst = _make_fake_instance()
    p1 = inst._config_marker_path({"model_name": "model_a"})
    p2 = inst._config_marker_path({"model_name": "model_b"})
    assert p1 != p2


def test_persistent_warmup_on_config_change():
    """Switching to a new config in _run_persistent should trigger warmup timeout."""
    inst = _make_fake_instance(device="cpu")
    # Pre-mark old config as complete
    old_params = {"model_checkpoint": "esm2_t33_650M_UR50D"}
    inst._mark_warmup_complete(old_params)
    inst._reload_params = dict(old_params)

    mock_worker = MagicMock()
    mock_worker.script_path = inst.script_path
    inst._worker = mock_worker

    with patch("proto_tools.utils.tool_instance.PersistentWorker") as MockPW:
        new_worker = MagicMock()
        new_worker.send.return_value = {"result": "ok"}
        MockPW.return_value = new_worker

        inst._run_persistent(
            {"device": "cpu", "model_checkpoint": "esm2_t36_3B_UR50D"},
            reload_on={"model_checkpoint"},
        )

        # New config should have used warmup timeout (3600)
        call_args = new_worker.send.call_args
        assert call_args.kwargs.get("timeout") == 3600 or call_args[1].get("timeout") == 3600


# ── Thread safety tests ───────────────────────────────────────────────────


@patch.object(ToolInstance, "__init__", return_value=None)
def test_concurrent_get_returns_same_instance(mock_init: MagicMock):
    """Many threads calling get() should all receive the same instance."""
    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results: list[ToolInstance] = [None] * num_threads

    def worker(idx: int):
        barrier.wait()  # all threads start together
        results[idx] = ToolInstance.get("esm2")

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(num_threads)]
        for f in futures:
            f.result()

    # All threads should have gotten the same object
    assert all(r is results[0] for r in results)
    assert len(_instances) == 1
    # __init__ called at most twice (double-check loser creates then discards)
    assert mock_init.call_count <= 2


@patch.object(ToolInstance, "__init__", return_value=None)
def test_concurrent_dispatch_with_cached_instance(mock_init: MagicMock):
    """Many threads dispatching to the same cached instance should succeed."""
    inst = ToolInstance.get("esm2")
    inst.run = MagicMock(return_value={"result": "ok"})

    num_threads = 10
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []

    def worker():
        barrier.wait()
        try:
            result = ToolInstance.dispatch("esm2", {"op": "score"})
            assert result == {"result": "ok"}
        except Exception as e:
            errors.append(e)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(worker) for _ in range(num_threads)]
        for f in futures:
            f.result()

    assert errors == [], f"Dispatch errors: {errors}"
    assert inst.run.call_count == num_threads


# ── _create_env() edge case tests ─────────────────────────────────────────


def test_failure_writes_status_and_raises(tmp_path: Path):
    """_create_env() should write FAILED status and raise on setup.sh failure."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.tool_name = "fake_tool"
    inst.device = "cpu"
    inst.env_path = tmp_path / "fake_env"
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\nexit 1\n")
    (tmp_path / "python_version.txt").write_text("default: 3.12\n")
    inst._tool_env_vars = {"passthrough": [], "set": []}

    mock_proc = MagicMock()
    mock_proc.returncode = 42
    mock_proc.communicate.return_value = ("", "setup failed!")

    def _create_env_dir(*args, **kwargs):
        """Simulate 'python -m venv' creating the directory."""
        inst.env_path.mkdir(parents=True, exist_ok=True)

    with (
        patch.object(
            inst,
            "_ensure_micromamba",
            return_value=Path("/fake/micromamba"),
        ),
        patch(
            "proto_tools.utils.tool_instance.subprocess.run",
            side_effect=_create_env_dir,
        ),
        patch(
            "proto_tools.utils.tool_instance.subprocess.Popen",
            return_value=mock_proc,
        ),
        pytest.raises(RuntimeError, match="may not be compatible"),
    ):
        inst._create_env()

    status = (inst.env_path / "STATUS.txt").read_text()
    assert status.startswith("FAILED")
    assert "42" in status
    assert "Setup hash:" in status


def test_build_failure_prevents_retry_in_process(tmp_path: Path):
    """After _create_env fails, _ensure_env raises immediately on retry."""
    inst = _make_fake_instance()
    inst._env_ready = False
    inst.env_path = tmp_path / "fake_env"
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\nexit 1\n")

    with (
        patch.object(ToolInstance, "_is_env_ok", return_value=False),
        patch.object(
            ToolInstance,
            "_create_env",
            side_effect=RuntimeError("'fake_tool' may not be compatible with your system. setup.sh failed (exit 1)."),
        ) as mock_create,
    ):
        with pytest.raises(RuntimeError, match=r"setup\.sh failed"):
            inst._ensure_env()

        mock_create.assert_called_once()

        # Second call should fail fast without calling _create_env again
        with pytest.raises(RuntimeError, match="may not be compatible"):
            inst._ensure_env()

        assert mock_create.call_count == 1


def test_stale_failure_warns_and_retries(tmp_path: Path, caplog):
    """A FAILED STATUS.txt with matching hash logs a warning and retries."""
    inst = _make_fake_instance()
    inst._env_ready = False
    inst.env_path = tmp_path / "fake_env"
    inst.env_path.mkdir()
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho hi\n")

    setup_hash = hashlib.sha256(inst.setup_script.read_bytes()).hexdigest()[:16]
    status_file = inst.env_path / "STATUS.txt"
    status_file.write_text(
        f"FAILED\n\n"
        f"Return code: 1\n"
        f"Command: {inst.setup_script}\n"
        f"Setup hash: {setup_hash}\n"
        f"Timestamp: 2025-01-01\n\n"
        f"STDERR:\npip install exploded\n"
    )

    with (
        patch.object(ToolInstance, "_is_env_ok", return_value=False),
        patch.object(ToolInstance, "_create_env") as mock_create,
        caplog.at_level(logging.WARNING),
    ):
        inst._ensure_env()

    mock_create.assert_called_once()
    assert "previously failed to build" in caplog.text
    assert setup_hash in caplog.text


def test_changed_setup_hash_logs_info_and_retries(tmp_path: Path, caplog):
    """A FAILED STATUS.txt with a different hash logs info and retries."""
    inst = _make_fake_instance()
    inst._env_ready = False
    inst.env_path = tmp_path / "fake_env"
    inst.env_path.mkdir()
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho fixed\n")

    status_file = inst.env_path / "STATUS.txt"
    status_file.write_text(
        "FAILED\n\n"
        "Return code: 1\n"
        "Command: setup.sh\n"
        "Setup hash: 0000000000000000\n"
        "Timestamp: 2025-01-01\n\n"
        "STDERR:\n\n"
    )

    with (
        patch.object(ToolInstance, "_is_env_ok", return_value=False),
        patch.object(ToolInstance, "_create_env") as mock_create,
        caplog.at_level(logging.INFO),
    ):
        inst._ensure_env()

    mock_create.assert_called_once()
    assert "Setup files changed" in caplog.text


def test_success_status_with_stale_hash_rebuilds(tmp_path: Path, caplog):
    """A SUCCESS STATUS.txt with a stale hash triggers a rebuild."""
    inst = _make_fake_instance()
    inst._env_ready = False
    inst.env_path = tmp_path / "fake_env"
    inst.env_path.mkdir()
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho updated\n")

    status_file = inst.env_path / "STATUS.txt"
    status_file.write_text("SUCCESS\nSetup hash: 0000000000000000\n")

    with patch.object(ToolInstance, "_create_env") as mock_create, caplog.at_level(logging.INFO):
        inst._ensure_env()

    mock_create.assert_called_once()
    assert "Setup files changed" in caplog.text


def test_build_failures_cleared_between_tests():
    """The autouse fixture clears _build_failures so tests are isolated."""
    # The fixture already ran — _build_failures should be empty
    assert len(ToolInstance._build_failures) == 0
    ToolInstance._build_failures["some_tool"] = "test error"
    # Next test will see it cleared by the fixture


# ── Setup hash tests ──────────────────────────────────────────────────────


def test_hash_includes_requirements_txt(tmp_path: Path):
    """_setup_hash() should incorporate requirements.txt when present."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho hi\n")

    hash_without_req = inst._setup_hash()

    # Add a requirements.txt — hash should change
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26\n")

    hash_with_req = inst._setup_hash()
    assert hash_without_req != hash_with_req


def test_hash_changes_when_requirements_change(tmp_path: Path):
    """Changing requirements.txt should change the hash."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho hi\n")

    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26\n")
    hash_v1 = inst._setup_hash()

    req.write_text("numpy==2.0\n")
    hash_v2 = inst._setup_hash()

    assert hash_v1 != hash_v2


def test_hash_changes_when_setup_changes(tmp_path: Path):
    """Changing setup.sh should change the hash."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho v1\n")
    hash_v1 = inst._setup_hash()

    inst.setup_script.write_text("#!/bin/bash\necho v2\n")
    hash_v2 = inst._setup_hash()

    assert hash_v1 != hash_v2


def test_hash_includes_env_vars_txt(tmp_path: Path):
    """_setup_hash() should incorporate env_vars.txt when present."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho hi\n")

    hash_without = inst._setup_hash()

    env_vars = tmp_path / "env_vars.txt"
    env_vars.write_text("[passthrough]\nHF_TOKEN\n")

    hash_with = inst._setup_hash()
    assert hash_without != hash_with


def test_hash_changes_when_env_vars_change(tmp_path: Path):
    """Changing env_vars.txt should change the hash."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho hi\n")

    env_vars = tmp_path / "env_vars.txt"
    env_vars.write_text("[passthrough]\nHF_TOKEN\n")
    hash_v1 = inst._setup_hash()

    env_vars.write_text("[passthrough]\nHF_TOKEN\nHF_HOME\n")
    hash_v2 = inst._setup_hash()

    assert hash_v1 != hash_v2


# ── _is_env_ok() tests ───────────────────────────────────────────────────


def test_returns_false_when_hash_mismatches(tmp_path: Path):
    """_is_env_ok() should return False when setup files changed."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.env_path = tmp_path / "fake_env"
    inst.env_path.mkdir()
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho updated\n")

    status_file = inst.env_path / "STATUS.txt"
    status_file.write_text("SUCCESS\nSetup hash: 0000000000000000\n")

    # Create a fake python executable that succeeds
    bin_dir = inst.env_path / "bin"
    bin_dir.mkdir()
    python_exe = bin_dir / "python"
    python_exe.write_text("#!/bin/bash\necho 'Python 3.12'\n")
    python_exe.chmod(0o755)

    with patch(
        "proto_tools.utils.tool_instance.subprocess.run",
        return_value=MagicMock(returncode=0),
    ):
        assert inst._is_env_ok() is False


def test_returns_true_when_hash_matches(tmp_path: Path):
    """_is_env_ok() should return True when hash matches and python works."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.env_path = tmp_path / "fake_env"
    inst.env_path.mkdir()
    inst.setup_script = tmp_path / "setup.sh"
    inst.setup_script.write_text("#!/bin/bash\necho hi\n")

    current_hash = inst._setup_hash()
    status_file = inst.env_path / "STATUS.txt"
    status_file.write_text(f"SUCCESS\nSetup hash: {current_hash}\n")

    bin_dir = inst.env_path / "bin"
    bin_dir.mkdir()
    python_exe = bin_dir / "python"
    python_exe.write_text("#!/bin/bash\necho 'Python 3.12'\n")
    python_exe.chmod(0o755)

    with patch(
        "proto_tools.utils.tool_instance.subprocess.run",
        return_value=MagicMock(returncode=0),
    ):
        assert inst._is_env_ok() is True


# ── _run_oneshot() output tests ──────────────────────────────────────────


def test_run_oneshot_reads_output(tmp_path: Path):
    """_run_oneshot() should return the parsed output JSON."""
    script = tmp_path / "echo_script.py"
    script.write_text(
        textwrap.dedent("""\
        import json, sys
        input_path, output_path = sys.argv[1], sys.argv[2]
        with open(input_path) as f:
            data = json.load(f)
        result = {"echo": data}
        with open(output_path, "w") as f:
            json.dump(result, f)
    """)
    )

    inst = ToolInstance.__new__(ToolInstance)
    inst.tool_name = "test"
    inst.device = "cpu"
    # Use current Python as the "venv" python
    python_dir = Path(sys.executable).parent
    inst.env_path = python_dir.parent
    inst.script_path = script
    inst._tool_env_vars = {"passthrough": [], "set": []}
    inst._env_ready = True

    result = inst._run_oneshot(
        {"hello": "world"},
        script_path=script,
    )
    assert result == {"echo": {"hello": "world"}}


# ── env_vars.txt loading tests ─────────────────────────────────────────────


def test_init_loads_env_vars():
    """ToolInstance should parse env_vars.txt from the standalone dir."""
    inst = ToolInstance("alphagenome")
    # alphagenome has an env_vars.txt with [set] LD_LIBRARY_PATH
    assert any("LD_LIBRARY_PATH" in entry for entry in inst._tool_env_vars["set"])


def test_init_empty_env_vars_for_tool_without_file():
    """Tools without env_vars.txt should get empty lists."""
    inst = ToolInstance("tmalign")
    assert inst._tool_env_vars == {"passthrough": [], "set": []}
