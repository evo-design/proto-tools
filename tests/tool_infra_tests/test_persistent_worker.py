"""Tests for PersistentWorker and _worker_bootstrap."""

from __future__ import annotations

import logging
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bio_programming_tools.utils.persistent_worker import (
    PersistentWorker,
    _build_subprocess_env,
    _parse_env_vars_file,
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
        env_path=fake_venv,
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

    def test_tool_env_path_injected(self, tmp_path: Path):
        """TOOL_VENV_PATH should be set in the subprocess environment."""
        script = tmp_path / "env_script.py"
        script.write_text(textwrap.dedent("""\
            import os

            def dispatch(input_dict):
                return {"env_path": os.environ.get("TOOL_VENV_PATH", "")}
            """))
        worker = _make_worker(script)
        try:
            result = worker.send({})
            assert result["env_path"] == str(worker.env_path)
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


class TestProcessGroupCleanup:
    """Tests for process-group-based cleanup in stop()."""

    def test_stop_signals_process_group(self):
        """stop() should send SIGTERM to the process group, not just the process."""
        worker = PersistentWorker.__new__(PersistentWorker)
        worker.tool_name = "test"

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 99999
        mock_process.stdin = MagicMock()
        worker._process = mock_process

        with patch("bio_programming_tools.utils.persistent_worker.os.killpg") as mock_killpg:
            worker.stop()

        # SIGTERM should be sent to the process group, not process.terminate()
        mock_killpg.assert_any_call(99999, signal.SIGTERM)
        mock_process.terminate.assert_not_called()
        assert worker._process is None

    def test_stop_escalates_to_sigkill(self):
        """stop() should SIGKILL the group if SIGTERM + wait fails."""
        worker = PersistentWorker.__new__(PersistentWorker)
        worker.tool_name = "test"

        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.pid = 99999
        mock_process.stdin = MagicMock()
        # First wait (after SIGTERM) times out, second wait (after SIGKILL) succeeds
        mock_process.wait.side_effect = [Exception("timed out"), None]
        worker._process = mock_process

        with patch("bio_programming_tools.utils.persistent_worker.os.killpg") as mock_killpg:
            worker.stop()

        calls = [c.args for c in mock_killpg.call_args_list]
        assert (99999, signal.SIGTERM) in calls
        assert (99999, signal.SIGKILL) in calls
        assert worker._process is None

    def test_stop_kills_child_processes(self, tmp_path: Path):
        """stop() should kill subprocesses spawned by the worker script."""
        # Script that forks a long-lived child and reports its PID
        script = tmp_path / "forking_script.py"
        script.write_text(textwrap.dedent("""\
            import subprocess, sys

            def dispatch(input_dict):
                # Spawn a long-lived child process
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(3600)"],
                )
                return {"child_pid": child.pid}
            """))

        worker = _make_worker(script)
        try:
            result = worker.send({})
            child_pid = result["child_pid"]

            # Child should be alive
            os.kill(child_pid, 0)  # raises ProcessLookupError if dead

            worker.stop()

            # Both the worker and its child should be dead
            assert not worker.alive
            # Poll briefly — the child may take a moment to be reaped after
            # the process group receives SIGTERM.
            for _ in range(50):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            worker.stop()


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


# ============================================================================
# Tests for _parse_env_vars_file
# ============================================================================
class TestParseEnvVarsFile:
    """Tests for the env_vars.txt parser."""

    def test_none_path(self):
        result = _parse_env_vars_file(None)
        assert result == {"passthrough": [], "set": []}

    def test_missing_file(self, tmp_path: Path):
        result = _parse_env_vars_file(tmp_path / "nonexistent.txt")
        assert result == {"passthrough": [], "set": []}

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "env_vars.txt"
        f.write_text("")
        result = _parse_env_vars_file(f)
        assert result == {"passthrough": [], "set": []}

    def test_passthrough_section(self, tmp_path: Path):
        f = tmp_path / "env_vars.txt"
        f.write_text("[passthrough]\nHF_TOKEN\nHF_HOME\n")
        result = _parse_env_vars_file(f)
        assert result["passthrough"] == ["HF_TOKEN", "HF_HOME"]
        assert result["set"] == []

    def test_set_section(self, tmp_path: Path):
        f = tmp_path / "env_vars.txt"
        f.write_text("[set]\nMY_VAR=${VENV_PATH}/data\n")
        result = _parse_env_vars_file(f)
        assert result["set"] == ["MY_VAR=${VENV_PATH}/data"]
        assert result["passthrough"] == []

    def test_both_sections(self, tmp_path: Path):
        f = tmp_path / "env_vars.txt"
        f.write_text(
            "[passthrough]\nHF_TOKEN\n\n"
            "[set]\nFOO=${VENV_PATH}/bar\n"
        )
        result = _parse_env_vars_file(f)
        assert result["passthrough"] == ["HF_TOKEN"]
        assert result["set"] == ["FOO=${VENV_PATH}/bar"]

    def test_comments_and_blank_lines(self, tmp_path: Path):
        f = tmp_path / "env_vars.txt"
        f.write_text(
            "# This is a comment\n"
            "\n"
            "[passthrough]\n"
            "# Another comment\n"
            "HF_TOKEN\n"
            "\n"
            "HF_HOME\n"
        )
        result = _parse_env_vars_file(f)
        assert result["passthrough"] == ["HF_TOKEN", "HF_HOME"]

    def test_unknown_section_warns(self, tmp_path: Path, caplog):
        f = tmp_path / "env_vars.txt"
        f.write_text("[bogus]\nFOO\n")
        with caplog.at_level(logging.WARNING):
            result = _parse_env_vars_file(f)
        assert "Unknown section" in caplog.text
        assert result == {"passthrough": [], "set": []}



# ============================================================================
# Tests for _build_subprocess_env (whitelist-based)
# ============================================================================
class TestCleanEnv:
    """Tests for whitelist-based subprocess environment construction."""

    def test_non_whitelisted_vars_are_absent(self, monkeypatch):
        """Jupyter, mamba root, and arbitrary vars must not leak.

        Note: CONDA_PREFIX, CONDA_DEFAULT_ENV, and CONDA_SHLVL are now
        intentionally passed through to support micromamba-created environments.
        """
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        monkeypatch.setenv("CONDA_SHLVL", "4")
        monkeypatch.setenv("MAMBA_ROOT_PREFIX", "/opt/mamba")
        monkeypatch.setenv("JPY_PARENT_PID", "12345")
        monkeypatch.setenv("RDBASE", "/opt/rdkit")
        monkeypatch.setenv("SOME_RANDOM_VAR", "leaked")

        env = _build_subprocess_env(device="cpu")

        # CONDA_PREFIX, CONDA_DEFAULT_ENV, CONDA_SHLVL are now whitelisted
        assert env.get("CONDA_PREFIX") == "/opt/conda"
        assert env.get("CONDA_DEFAULT_ENV") == "base"
        assert env.get("CONDA_SHLVL") == "4"
        # But these should still be filtered out
        assert "MAMBA_ROOT_PREFIX" not in env
        assert "JPY_PARENT_PID" not in env
        assert "RDBASE" not in env
        assert "SOME_RANDOM_VAR" not in env

    def test_base_whitelist_vars_present(self, monkeypatch):
        """Whitelisted vars should be passed through when set."""
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
        monkeypatch.setenv("CONDA_SHLVL", "2")

        env = _build_subprocess_env(device="cpu")

        assert env["HOME"] == "/home/test"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["HTTP_PROXY"] == "http://proxy:8080"
        # Conda variables needed for micromamba environments
        assert env["CONDA_PREFIX"] == "/opt/conda"
        assert env["CONDA_DEFAULT_ENV"] == "base"
        assert env["CONDA_SHLVL"] == "2"

    def test_missing_whitelist_vars_not_added(self, monkeypatch):
        """Whitelisted vars not in parent env should not appear."""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
        monkeypatch.delenv("CONDA_SHLVL", raising=False)

        env = _build_subprocess_env(device="cpu")

        assert "HTTP_PROXY" not in env
        assert "HTTPS_PROXY" not in env
        assert "CONDA_PREFIX" not in env
        assert "CONDA_DEFAULT_ENV" not in env
        assert "CONDA_SHLVL" not in env

    def test_path_reconstructed_without_conda(self, monkeypatch, tmp_path: Path):
        """PATH should be reconstructed from venv/bin + system dirs, no conda."""
        monkeypatch.setenv("PATH", "/opt/conda/bin:/usr/bin:/bin")

        env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)

        path_parts = env["PATH"].split(":")
        assert path_parts[0] == str(tmp_path / "bin")
        assert "/opt/conda/bin" not in path_parts
        assert "/usr/bin" in path_parts
        assert "/bin" in path_parts

    def test_path_includes_cuda_for_gpu(self, monkeypatch, tmp_path: Path):
        """GPU device should add /usr/local/cuda/bin to PATH."""
        env = _build_subprocess_env(device="cuda", tool_env_path=tmp_path)

        path_parts = env["PATH"].split(":")
        assert "/usr/local/cuda/bin" in path_parts

    def test_path_excludes_cuda_for_cpu(self, monkeypatch, tmp_path: Path):
        """CPU device should not have /usr/local/cuda/bin in PATH."""
        env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)

        path_parts = env["PATH"].split(":")
        assert "/usr/local/cuda/bin" not in path_parts

    def test_parent_ld_library_path_not_inherited(self, monkeypatch):
        """Parent LD_LIBRARY_PATH must not leak into subprocess."""
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/conda/lib:/some/other/lib")

        env = _build_subprocess_env(device="cpu")

        assert "LD_LIBRARY_PATH" not in env

    def test_ld_library_path_not_auto_set(self, monkeypatch, tmp_path: Path):
        """LD_LIBRARY_PATH should NOT be auto-discovered from venv CUDA libs."""
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/conda/lib")

        # Even with cuda_env/lib present, no auto-discovery should happen
        cuda_env_lib = tmp_path / "cuda_env" / "lib"
        cuda_env_lib.mkdir(parents=True)

        env = _build_subprocess_env(device="cuda", tool_env_path=tmp_path)

        assert "LD_LIBRARY_PATH" not in env

    def test_ld_library_path_via_set_directive(self, tmp_path: Path):
        """Tools can explicitly set LD_LIBRARY_PATH via [set] in env_vars.txt."""
        tool_env_vars = {
            "passthrough": [],
            "set": [
                "LD_LIBRARY_PATH=${VENV_PATH}/cuda_env/lib:${VENV_PATH}/cuda_env/lib64"
            ],
        }
        env = _build_subprocess_env(
            device="cuda",
            tool_env_path=tmp_path,
            tool_env_vars=tool_env_vars,
        )

        expected = f"{tmp_path}/cuda_env/lib:{tmp_path}/cuda_env/lib64"
        assert env["LD_LIBRARY_PATH"] == expected

    def test_torch_home_set_per_venv(self, tmp_path: Path):
        """TORCH_HOME should be set to {venv}/cache/torch."""
        env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)

        assert env["TORCH_HOME"] == str(tmp_path / "cache" / "torch")

    def test_torch_home_not_set_without_venv(self):
        """Without a venv path, TORCH_HOME should not be set."""
        env = _build_subprocess_env(device="cpu", tool_env_path=None)

        assert "TORCH_HOME" not in env

    def test_jax_platforms_cpu(self, monkeypatch):
        """CPU device should set JAX_PLATFORMS=cpu."""
        env = _build_subprocess_env(device="cpu")

        assert env["JAX_PLATFORMS"] == "cpu"

    def test_jax_platforms_not_set_for_cuda(self, monkeypatch):
        """GPU device should not set JAX_PLATFORMS."""
        env = _build_subprocess_env(device="cuda")

        assert "JAX_PLATFORMS" not in env

    def test_passthrough_vars_present_when_set(self, monkeypatch):
        """Tool-specific passthrough vars should appear when in parent env."""
        monkeypatch.setenv("HF_TOKEN", "secret-token")

        tool_env_vars = {"passthrough": ["HF_TOKEN"], "set": []}
        env = _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

        assert env["HF_TOKEN"] == "secret-token"

    def test_passthrough_vars_absent_when_not_set(self, monkeypatch):
        """Tool-specific passthrough vars missing in parent should not appear."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        tool_env_vars = {"passthrough": ["HF_TOKEN"], "set": []}
        env = _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

        assert "HF_TOKEN" not in env

    def test_passthrough_missing_var_warns(self, monkeypatch, caplog):
        """Passthrough var not in parent env should emit a debug message."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        tool_env_vars = {"passthrough": ["HF_TOKEN"], "set": []}
        with caplog.at_level(logging.DEBUG):
            _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

        assert "HF_TOKEN" in caplog.text
        assert "not set in the parent environment" in caplog.text

    def test_set_vars_with_venv_interpolation(self, tmp_path: Path):
        """[set] entries should interpolate ${VENV_PATH}."""
        tool_env_vars = {
            "passthrough": [],
            "set": ["MY_DATA=${VENV_PATH}/data"],
        }
        env = _build_subprocess_env(
            device="cpu",
            tool_env_path=tmp_path,
            tool_env_vars=tool_env_vars,
        )

        assert env["MY_DATA"] == f"{tmp_path}/data"

    def test_set_vars_literal(self, tmp_path: Path):
        """[set] entries without interpolation should be literal."""
        tool_env_vars = {
            "passthrough": [],
            "set": ["FOO=bar"],
        }
        env = _build_subprocess_env(
            device="cpu",
            tool_env_path=tmp_path,
            tool_env_vars=tool_env_vars,
        )

        assert env["FOO"] == "bar"


# ============================================================================
# Tests for compute environment injection
# ============================================================================
class TestComputeEnvInjection:
    """Test that compute environment vars are injected into subprocess env."""

    @pytest.fixture(autouse=True)
    def clear_caches(self):
        """Clear LRU caches before each test to ensure mocks work correctly."""
        from bio_programming_tools.utils.compute_deps import detect_compute_environment
        from bio_programming_tools.utils.system_info import get_gpu_info

        get_gpu_info.cache_clear()
        detect_compute_environment.cache_clear()
        yield
        get_gpu_info.cache_clear()
        detect_compute_environment.cache_clear()

    def test_compute_env_vars_present_gpu(self, monkeypatch):
        """On GPU systems, compute env vars should be present."""
        from bio_programming_tools.utils.system_info import GPUDevice, GPUInfo

        fake_gpu_info = GPUInfo(
            available=True,
            count=1,
            driver_version="550.127",
            cuda_version="12.4",
            devices=[
                GPUDevice(
                    index=0,
                    name="NVIDIA A100",
                    compute_capability="8.0",
                    vram_gb=40.0,
                )
            ],
        )

        with monkeypatch.context() as m:
            m.setattr(
                "bio_programming_tools.utils.system_info.get_gpu_info",
                lambda: fake_gpu_info,
            )
            env = _build_subprocess_env(device="cuda")

        # Should have all compute env vars
        assert "DETECTED_COMPUTE_PLATFORM" in env
        assert env["DETECTED_COMPUTE_PLATFORM"] == "cuda"
        assert "DETECTED_DRIVER_VERSION" in env
        assert env["DETECTED_DRIVER_VERSION"] == "550"
        assert "DETECTED_CUDA_VERSION" in env
        assert env["DETECTED_CUDA_VERSION"] == "12"
        assert "RECOMMENDED_TORCH_SPEC" in env
        assert "torch>=" in env["RECOMMENDED_TORCH_SPEC"]
        assert "RECOMMENDED_JAX_SPEC" in env
        assert "jax[cuda" in env["RECOMMENDED_JAX_SPEC"]
        assert "RECOMMENDED_JAX_VARIANT" in env
        assert env["RECOMMENDED_JAX_VARIANT"].startswith("cuda")

    def test_compute_env_vars_present_cpu(self, monkeypatch):
        """On CPU systems, compute env vars should be present (simplified)."""
        from bio_programming_tools.utils.system_info import GPUInfo

        fake_gpu_info = GPUInfo(
            available=False,
            count=0,
            driver_version=None,
            cuda_version=None,
            devices=[],
        )

        with monkeypatch.context() as m:
            m.setattr(
                "bio_programming_tools.utils.system_info.get_gpu_info",
                lambda: fake_gpu_info,
            )
            env = _build_subprocess_env(device="cpu")

        # CPU systems should have basic vars
        assert "DETECTED_COMPUTE_PLATFORM" in env
        assert env["DETECTED_COMPUTE_PLATFORM"] == "cpu"
        assert "RECOMMENDED_TORCH_SPEC" in env
        assert env["RECOMMENDED_TORCH_SPEC"] == "torch"
        assert "RECOMMENDED_JAX_SPEC" in env
        assert env["RECOMMENDED_JAX_SPEC"] == "jax"

    def test_compute_env_vars_can_be_overridden_by_tool(self, monkeypatch, tmp_path: Path):
        """Tool-specific env vars can override compute env recommendations."""
        from bio_programming_tools.utils.system_info import GPUDevice, GPUInfo

        fake_gpu_info = GPUInfo(
            available=True,
            count=1,
            driver_version="550.127",
            cuda_version="12.4",
            devices=[
                GPUDevice(
                    index=0,
                    name="NVIDIA A100",
                    compute_capability="8.0",
                    vram_gb=40.0,
                )
            ],
        )

        # Tool overrides torch spec
        tool_env_vars = {
            "passthrough": [],
            "set": ["RECOMMENDED_TORCH_SPEC=torch==2.6.0"],
        }

        with monkeypatch.context() as m:
            m.setattr(
                "bio_programming_tools.utils.system_info.get_gpu_info",
                lambda: fake_gpu_info,
            )
            env = _build_subprocess_env(
                device="cuda",
                tool_env_path=tmp_path,
                tool_env_vars=tool_env_vars,
            )

        # Tool override should win
        assert env["RECOMMENDED_TORCH_SPEC"] == "torch==2.6.0"
