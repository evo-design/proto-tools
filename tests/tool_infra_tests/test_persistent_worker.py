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


# Path to standalone_standalone_helpers.py source
STANDALONE_HELPERS_SOURCE = Path(__file__).parent.parent.parent / "bio_programming_tools" / "utils" / "standalone_standalone_helpers.py"


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

    @pytest.mark.parametrize(
        "op, name, expected",
        [
            ("greet", "Alice", {"greeting": "hello Alice"}),
            ("farewell", "Bob", {"farewell": "goodbye Bob"}),
        ],
    )
    def test_legacy_dispatch(self, legacy_script: Path, op, name, expected):
        worker = _make_worker(legacy_script)
        try:
            result = worker.send({"operation": op, "name": name})
            assert result == expected
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
        """Conda activation vars, jupyter, mamba, and arbitrary vars must not leak."""
        leaked = {
            "CONDA_PREFIX": "/opt/conda",
            "CONDA_DEFAULT_ENV": "base",
            "CONDA_SHLVL": "4",
            "MAMBA_ROOT_PREFIX": "/opt/mamba",
            "JPY_PARENT_PID": "12345",
            "RDBASE": "/opt/rdkit",
            "SOME_RANDOM_VAR": "leaked",
        }
        for k, v in leaked.items():
            monkeypatch.setenv(k, v)

        # Without tool_env_path, none of these should appear
        env = _build_subprocess_env(device="cpu")
        for var in leaked:
            assert var not in env, f"{var} leaked into subprocess env"

    def test_base_whitelist_vars_present(self, monkeypatch):
        """Whitelisted vars should be passed through when set."""
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")

        env = _build_subprocess_env(device="cpu")

        assert env["HOME"] == "/home/test"
        assert env["LANG"] == "en_US.UTF-8"
        assert env["HTTP_PROXY"] == "http://proxy:8080"

    def test_missing_whitelist_vars_not_added(self, monkeypatch):
        """Whitelisted vars not in parent env should not appear."""
        monkeypatch.delenv("HTTP_PROXY", raising=False)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)

        env = _build_subprocess_env(device="cpu")

        assert "HTTP_PROXY" not in env
        assert "HTTPS_PROXY" not in env

    # -- PATH construction --

    @pytest.mark.parametrize("device, expect_cuda", [("cpu", False), ("cuda", True)])
    def test_path_ordering(self, monkeypatch, tmp_path: Path, device, expect_cuda):
        """PATH: venv/bin > (cuda if GPU) > parent PATH > system dirs."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        monkeypatch.setenv("PATH", "/opt/conda/bin:/opt/module/bin:/usr/bin:/bin")

        env = _build_subprocess_env(device=device, tool_env_path=tmp_path)

        path_parts = env["PATH"].split(":")
        assert path_parts[0] == str(tmp_path / "bin")
        # Parent PATH entries carried over (including conda/bin from parent)
        assert "/opt/conda/bin" in path_parts
        assert "/opt/module/bin" in path_parts
        assert ("/usr/local/cuda/bin" in path_parts) == expect_cuda
        if expect_cuda:
            cuda_idx = path_parts.index("/usr/local/cuda/bin")
            conda_idx = path_parts.index("/opt/conda/bin")
            assert cuda_idx < conda_idx  # cuda/bin before parent PATH

    def test_path_without_conda_prefix(self, monkeypatch, tmp_path: Path):
        """Without CONDA_PREFIX, no conda bin in PATH; parent PATH still carried over."""
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.setenv("PATH", "/opt/hpc/bin:/usr/bin:/bin")

        env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)

        path_parts = env["PATH"].split(":")
        assert path_parts[0] == str(tmp_path / "bin")
        assert "/opt/hpc/bin" in path_parts  # parent PATH entry carried over

    # -- CONDA_PREFIX / VIRTUAL_ENV --

    @pytest.mark.parametrize(
        "has_tool_env",
        [True, False],
        ids=["with_tool_env", "without_tool_env"],
    )
    def test_conda_prefix_and_virtual_env(self, monkeypatch, tmp_path: Path, has_tool_env):
        """CONDA_PREFIX/VIRTUAL_ENV point to tool env when provided, absent otherwise."""
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")

        tool_env_path = tmp_path if has_tool_env else None
        env = _build_subprocess_env(device="cpu", tool_env_path=tool_env_path)

        if has_tool_env:
            assert env["CONDA_PREFIX"] == str(tmp_path)
            assert env["VIRTUAL_ENV"] == str(tmp_path)
        else:
            assert "CONDA_PREFIX" not in env
            assert "VIRTUAL_ENV" not in env

    # -- LD_LIBRARY_PATH --

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_parent_ld_library_path_inherited(self, monkeypatch, device):
        """Parent LD_LIBRARY_PATH entries are carried over for all devices."""
        monkeypatch.setenv(
            "LD_LIBRARY_PATH",
            "/usr/local/cuda-12.4/lib64:/usr/lib64:/opt/nvidia/lib64"
        )
        monkeypatch.delenv("CONDA_PREFIX", raising=False)

        env = _build_subprocess_env(device=device)

        ld_parts = env["LD_LIBRARY_PATH"].split(":")
        assert "/usr/local/cuda-12.4/lib64" in ld_parts
        assert "/opt/nvidia/lib64" in ld_parts
        assert "/usr/lib64" in ld_parts

    @pytest.mark.parametrize(
        "conda_prefix, expect_ld",
        [("/opt/conda", "/opt/conda/lib"), (None, None)],
        ids=["with_conda", "without_conda"],
    )
    def test_ld_library_path_conda_auto(self, monkeypatch, conda_prefix, expect_ld):
        """LD_LIBRARY_PATH includes $CONDA_PREFIX/lib when set, absent otherwise (CPU)."""
        if conda_prefix:
            monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
        else:
            monkeypatch.delenv("CONDA_PREFIX", raising=False)
        # Clear parent LD so we isolate the conda auto-set behavior
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

        env = _build_subprocess_env(device="cpu")

        if expect_ld:
            assert env["LD_LIBRARY_PATH"] == expect_ld
        else:
            assert "LD_LIBRARY_PATH" not in env

    @pytest.mark.parametrize(
        "conda_prefix, expect_conda_lib",
        [("/opt/conda", True), (None, False)],
        ids=["with_conda", "without_conda"],
    )
    def test_ld_library_path_via_set_directive(
        self, monkeypatch, tmp_path: Path, conda_prefix, expect_conda_lib
    ):
        """[set] LD_LIBRARY_PATH + parent LD + conda lib (when set)."""
        if conda_prefix:
            monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
        else:
            monkeypatch.delenv("CONDA_PREFIX", raising=False)
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
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

        ld_parts = env["LD_LIBRARY_PATH"].split(":")
        # Tool-specific paths come first
        assert ld_parts[0] == f"{tmp_path}/cuda_env/lib"
        assert ld_parts[1] == f"{tmp_path}/cuda_env/lib64"
        # Conda lib present only when CONDA_PREFIX set
        assert ("/opt/conda/lib" in ld_parts) == expect_conda_lib

    # -- Per-venv vars --

    @pytest.mark.parametrize(
        "has_venv, expect_torch_home",
        [(True, True), (False, False)],
        ids=["with_venv", "without_venv"],
    )
    def test_torch_home(self, tmp_path: Path, has_venv, expect_torch_home):
        """TORCH_HOME = {venv}/cache/torch when venv provided, absent otherwise."""
        tool_env_path = tmp_path if has_venv else None
        env = _build_subprocess_env(device="cpu", tool_env_path=tool_env_path)

        if expect_torch_home:
            assert env["TORCH_HOME"] == str(tmp_path / "cache" / "torch")
        else:
            assert "TORCH_HOME" not in env

    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda"],
        ids=["cpu", "cuda"],
    )
    def test_jax_platforms_not_set(self, device):
        """JAX_PLATFORMS should NOT be set for any device (allows later GPU access via device_put)."""
        env = _build_subprocess_env(device=device)
        assert "JAX_PLATFORMS" not in env

    # -- Tool-specific env vars (env_vars.txt) --

    def test_xla_preallocation_disabled_for_cpu(self):
        """CPU device should disable JAX preallocation."""
        env = _build_subprocess_env(device="cpu")

        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"

    def test_xla_preallocation_disabled_for_cuda(self):
        """GPU device should also disable JAX preallocation (DeviceManager handles placement)."""
        env = _build_subprocess_env(device="cuda")

        assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"

    @pytest.mark.parametrize(
        "parent_has_var",
        [True, False],
        ids=["present", "absent"],
    )
    def test_passthrough_vars(self, monkeypatch, parent_has_var):
        """Tool-specific passthrough vars appear only when set in parent env."""
        if parent_has_var:
            monkeypatch.setenv("HF_TOKEN", "secret-token")
        else:
            monkeypatch.delenv("HF_TOKEN", raising=False)

        tool_env_vars = {"passthrough": ["HF_TOKEN"], "set": []}
        env = _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

        if parent_has_var:
            assert env["HF_TOKEN"] == "secret-token"
        else:
            assert "HF_TOKEN" not in env

    def test_passthrough_missing_var_warns(self, monkeypatch, caplog):
        """Passthrough var not in parent env should emit a debug message."""
        monkeypatch.delenv("HF_TOKEN", raising=False)

        tool_env_vars = {"passthrough": ["HF_TOKEN"], "set": []}
        with caplog.at_level(logging.DEBUG):
            _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

        assert "HF_TOKEN" in caplog.text
        assert "not set in the parent environment" in caplog.text

    @pytest.mark.parametrize(
        "set_line, var, expected_suffix",
        [
            ("MY_DATA=${VENV_PATH}/data", "MY_DATA", "/data"),
            ("FOO=bar", "FOO", None),
        ],
        ids=["interpolation", "literal"],
    )
    def test_set_vars(self, tmp_path: Path, set_line, var, expected_suffix):
        """[set] entries: interpolate ${VENV_PATH} or pass through literally."""
        tool_env_vars = {"passthrough": [], "set": [set_line]}
        env = _build_subprocess_env(
            device="cpu", tool_env_path=tmp_path, tool_env_vars=tool_env_vars,
        )

        if expected_suffix:
            assert env[var] == f"{tmp_path}{expected_suffix}"
        else:
            assert env[var] == set_line.split("=", 1)[1]


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


# ============================================================================
# Helper File Copy Tests
# ============================================================================


def test_helpers_copied_on_worker_startup(tmp_path: Path, echo_script):
    """Verify standalone_helpers.py is copied to standalone directory on worker startup."""
    # Create a minimal fake tool environment
    fake_env = tmp_path / "fake_env"
    fake_env.mkdir()
    (fake_env / "bin").mkdir()

    # Create a Python executable symlink (points to current Python)
    python_exe = fake_env / "bin" / "python"
    python_exe.symlink_to(sys.executable)

    # Create standalone directory for script
    standalone_dir = tmp_path / "standalone"
    standalone_dir.mkdir()

    # Move echo script to standalone directory
    script_path = standalone_dir / "test_script.py"
    script_path.write_text(echo_script.read_text())

    # Verify standalone_helpers.py doesn't exist yet
    helpers_path = standalone_dir / "standalone_helpers.py"
    assert not helpers_path.exists(), "standalone_helpers.py should not exist before worker starts"

    # Start the worker
    worker = PersistentWorker(
        tool_name="test-tool",
        env_path=fake_env,
        script_path=script_path,
    )

    try:
        worker.start()

        # Call send to ensure worker has fully started
        result = worker.send({"test": "data"})
        assert result["echo"]["test"] == "data", "Worker should be functional"

        # Verify standalone_helpers.py was copied
        assert helpers_path.exists(), "standalone_helpers.py should be copied on worker startup"

        # Verify content matches source exactly
        if STANDALONE_HELPERS_SOURCE.exists():
            source_content = STANDALONE_HELPERS_SOURCE.read_text()
            copied_content = helpers_path.read_text()
            assert copied_content == source_content, \
                "Copied standalone_helpers.py should be identical to source standalone_standalone_helpers.py"

    finally:
        worker.stop()


def test_helpers_not_copied_outside_standalone(tmp_path: Path):
    """Verify standalone_helpers.py is not copied if script is not in a standalone/ directory."""
    # Create a script in a non-standalone location
    script = tmp_path / "script_not_in_standalone.py"
    script.write_text(textwrap.dedent("""\
        def dispatch(input_dict):
            return {"result": "ok"}
        """))

    # Create a minimal fake tool environment
    fake_env = tmp_path / "fake_env"
    fake_env.mkdir()
    (fake_env / "bin").mkdir()

    # Create a Python executable symlink
    python_exe = fake_env / "bin" / "python"
    python_exe.symlink_to(sys.executable)

    # Start worker with script NOT in standalone/ directory
    worker = PersistentWorker(
        tool_name="test-tool",
        env_path=fake_env,
        script_path=script,
    )

    try:
        worker.start()

        # Worker should still function (just without standalone_helpers.py)
        result = worker.send({"test": "data"})
        assert result["result"] == "ok"

        # Verify standalone_helpers.py was NOT copied
        helpers_path = tmp_path / "standalone_helpers.py"
        assert not helpers_path.exists(), \
            "standalone_helpers.py should not be copied for scripts outside standalone/ directories"

    finally:
        worker.stop()
