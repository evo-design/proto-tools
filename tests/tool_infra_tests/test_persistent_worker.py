"""Tests for PersistentWorker and _worker_bootstrap."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from bio_programming_tools.utils.persistent_worker import (
    PersistentWorker,
    _clean_env,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def echo_script(tmp_path: Path) -> Path:
    """A trivial standalone script that echoes input back."""
    script = tmp_path / "echo_script.py"
    script.write_text(textwrap.dedent("""\
        import json, sys

        def dispatch(input_dict):
            return {"echo": input_dict}

        if __name__ == "__main__":
            input_path, output_path = sys.argv[1], sys.argv[2]
            with open(input_path) as f:
                data = json.load(f)
            result = dispatch(data)
            with open(output_path, "w") as f:
                json.dump(result, f)
        """))
    return script


@pytest.fixture
def adder_script(tmp_path: Path) -> Path:
    """A standalone script that adds two numbers, simulating stateful work."""
    script = tmp_path / "adder_script.py"
    script.write_text(textwrap.dedent("""\
        import json, sys

        _call_count = 0

        def dispatch(input_dict):
            global _call_count
            _call_count += 1
            a = input_dict["a"]
            b = input_dict["b"]
            return {"sum": a + b, "call_count": _call_count}

        if __name__ == "__main__":
            input_path, output_path = sys.argv[1], sys.argv[2]
            with open(input_path) as f:
                data = json.load(f)
            result = dispatch(data)
            with open(output_path, "w") as f:
                json.dump(result, f)
        """))
    return script


@pytest.fixture
def error_script(tmp_path: Path) -> Path:
    """A standalone script that raises an error."""
    script = tmp_path / "error_script.py"
    script.write_text(textwrap.dedent("""\
        def dispatch(input_dict):
            raise ValueError("intentional test error")
        """))
    return script


@pytest.fixture
def legacy_script(tmp_path: Path) -> Path:
    """A standalone script without dispatch(), using the legacy __main__ pattern."""
    script = tmp_path / "legacy_script.py"
    script.write_text(textwrap.dedent("""\
        import json, sys

        def run_greet(input_dict):
            name = input_dict.get("name", "world")
            return {"greeting": f"hello {name}"}

        def run_farewell(input_dict):
            name = input_dict.get("name", "world")
            return {"farewell": f"goodbye {name}"}

        if __name__ == "__main__":
            input_path, output_path = sys.argv[1], sys.argv[2]
            with open(input_path) as f:
                data = json.load(f)
            op = data["operation"]
            if op == "greet":
                result = run_greet(data)
            elif op == "farewell":
                result = run_farewell(data)
            else:
                raise ValueError(f"Unknown operation: {op}")
            with open(output_path, "w") as f:
                json.dump(result, f)
        """))
    return script


def _make_worker(script_path: Path) -> PersistentWorker:
    """Create a PersistentWorker using the current Python (no venv needed)."""
    # Use the current Python's directory as a fake venv
    python_dir = Path(sys.executable).parent
    fake_venv = python_dir.parent  # e.g. /usr → /usr/bin/python
    return PersistentWorker(
        tool_name="test",
        venv_path=fake_venv,
        script_path=script_path,
        device="cpu",
    )


# ============================================================================
# Tests
# ============================================================================
class TestPersistentWorkerBasic:
    """Basic send/receive tests."""

    def test_echo(self, echo_script: Path):
        worker = _make_worker(echo_script)
        try:
            result = worker.send({"foo": "bar"})
            assert result == {"echo": {"foo": "bar"}}
        finally:
            worker.stop()

    def test_tool_venv_path_injected(self, tmp_path: Path):
        """TOOL_VENV_PATH should be set in the subprocess environment."""
        script = tmp_path / "env_script.py"
        script.write_text(textwrap.dedent("""\
            import os

            def dispatch(input_dict):
                return {"venv_path": os.environ.get("TOOL_VENV_PATH", "")}
            """))
        worker = _make_worker(script)
        try:
            result = worker.send({})
            assert result["venv_path"] == str(worker.venv_path)
        finally:
            worker.stop()

    def test_multiple_calls(self, adder_script: Path):
        worker = _make_worker(adder_script)
        try:
            r1 = worker.send({"a": 1, "b": 2})
            assert r1["sum"] == 3
            assert r1["call_count"] == 1

            r2 = worker.send({"a": 10, "b": 20})
            assert r2["sum"] == 30
            assert r2["call_count"] == 2  # Same process, counter incremented
        finally:
            worker.stop()

    def test_error_handling(self, error_script: Path):
        worker = _make_worker(error_script)
        try:
            with pytest.raises(RuntimeError, match="intentional test error"):
                worker.send({"anything": True})
        finally:
            worker.stop()


class TestLegacyDispatch:
    """Tests for the legacy dispatch pattern (run_{operation} functions)."""

    def test_legacy_greet(self, legacy_script: Path):
        worker = _make_worker(legacy_script)
        try:
            result = worker.send({"operation": "greet", "name": "Alice"})
            assert result == {"greeting": "hello Alice"}
        finally:
            worker.stop()

    def test_legacy_farewell(self, legacy_script: Path):
        worker = _make_worker(legacy_script)
        try:
            result = worker.send({"operation": "farewell", "name": "Bob"})
            assert result == {"farewell": "goodbye Bob"}
        finally:
            worker.stop()

    def test_legacy_unknown_operation(self, legacy_script: Path):
        worker = _make_worker(legacy_script)
        try:
            with pytest.raises(RuntimeError, match="Cannot dispatch operation"):
                worker.send({"operation": "nonexistent"})
        finally:
            worker.stop()


class TestWorkerLifecycle:
    """Tests for worker start/stop/restart."""

    def test_stop_and_restart(self, echo_script: Path):
        worker = _make_worker(echo_script)
        try:
            result = worker.send({"x": 1})
            assert result == {"echo": {"x": 1}}
            assert worker.alive

            worker.stop()
            assert not worker.alive

            # Should auto-restart on next send
            result = worker.send({"x": 2})
            assert result == {"echo": {"x": 2}}
            assert worker.alive
        finally:
            worker.stop()

    def test_alive_property(self, echo_script: Path):
        worker = _make_worker(echo_script)
        assert not worker.alive
        worker.start()
        assert worker.alive
        worker.stop()
        assert not worker.alive


class TestSerialize:
    """Test that _worker_bootstrap._serialize() handles special types."""

    def test_serialize_tensor_like(self, tmp_path: Path):
        """_serialize() should handle objects with .detach(), .cpu(), .tolist()."""
        script = tmp_path / "tensor_script.py"
        script.write_text(textwrap.dedent("""\
            class FakeTensor:
                def __init__(self, data):
                    self._data = data
                def detach(self):
                    return self
                def cpu(self):
                    return self
                def tolist(self):
                    return self._data

            class FakeScalar:
                def __init__(self, val):
                    self._val = val
                def detach(self):
                    return self
                def cpu(self):
                    return self
                def item(self):
                    return self._val

            def dispatch(input_dict):
                return {
                    "tensor": FakeTensor([1.0, 2.0, 3.0]),
                    "scalar": FakeScalar(42),
                    "nested": {"arr": FakeTensor([[1, 2], [3, 4]])},
                    "plain": "hello",
                }
        """))
        worker = _make_worker(script)
        try:
            result = worker.send({})
            assert result == {
                "tensor": [1.0, 2.0, 3.0],
                "scalar": 42,
                "nested": {"arr": [[1, 2], [3, 4]]},
                "plain": "hello",
            }
        finally:
            worker.stop()


class TestCleanEnv:
    """Tests for subprocess environment construction."""

    def test_clean_env_preserves_parent_ld_library_path(self, monkeypatch):
        monkeypatch.setenv("LD_LIBRARY_PATH", "/base/a:/base/b")

        env = _clean_env(device="cpu")

        assert env["LD_LIBRARY_PATH"] == "/base/a:/base/b"

    def test_clean_env_omits_ld_library_path_when_unset(self, monkeypatch):
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

        env = _clean_env(device="cpu")

        assert "LD_LIBRARY_PATH" not in env

    def test_clean_env_merges_tool_venv_library_paths(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("LD_LIBRARY_PATH", "/base/a:/base/b")

        cuda_runtime_lib = tmp_path / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
        cuda_runtime_lib.mkdir(parents=True)
        cuda_env_lib = tmp_path / "cuda_env" / "lib"
        cuda_env_lib.mkdir(parents=True)

        env = _clean_env(device="cuda", tool_venv_path=tmp_path)

        merged_parts = env["LD_LIBRARY_PATH"].split(":")
        assert merged_parts[0] == str(cuda_env_lib)
        assert str(cuda_runtime_lib) in merged_parts
        assert "/base/a" in merged_parts
        assert "/base/b" in merged_parts

    def test_clean_env_dedupes_merged_library_paths(self, monkeypatch, tmp_path: Path):
        cuda_runtime_lib = tmp_path / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
        cuda_runtime_lib.mkdir(parents=True)
        monkeypatch.setenv("LD_LIBRARY_PATH", f"{cuda_runtime_lib}:/base/a:{cuda_runtime_lib}")

        env = _clean_env(device="cuda", tool_venv_path=tmp_path)

        merged_parts = env["LD_LIBRARY_PATH"].split(":")
        assert merged_parts.count(str(cuda_runtime_lib)) == 1

    def test_clean_env_sets_jax_platforms_for_cpu(self, monkeypatch):
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)

        env = _clean_env(device="cpu")

        assert env["JAX_PLATFORMS"] == "cpu"

    def test_clean_env_unsets_jax_platforms_for_cuda(self, monkeypatch):
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        env = _clean_env(device="cuda")

        assert "JAX_PLATFORMS" not in env
