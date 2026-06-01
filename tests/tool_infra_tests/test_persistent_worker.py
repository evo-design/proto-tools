"""tests/tool_infra_tests/test_persistent_worker.py.

Tests for PersistentWorker and _worker_bootstrap.
"""

import json
import logging
import os
import signal
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from proto_tools.utils.persistent_worker import (
    PersistentWorker,
    _build_subprocess_env,
    _handle_raw_stderr_line,
    _parse_env_vars_file,
)
from proto_tools.utils.proto_home import get_proto_home

_STANDALONE_HELPERS_SOURCE = (
    Path(__file__).parent.parent.parent / "proto_tools" / "utils" / "standalone_helpers_source" / "standalone_helpers"
)


@pytest.fixture(autouse=True)
def _clear_proto_home_cache():
    """Clear cached value computed with patched expanduser."""
    yield
    get_proto_home.cache_clear()


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def echo_script(tmp_path: Path) -> Path:
    """A trivial standalone script that echoes input back."""
    script = tmp_path / "echo_script.py"
    script.write_text(
        textwrap.dedent("""\
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
        """)
    )
    return script


@pytest.fixture
def adder_script(tmp_path: Path) -> Path:
    """A standalone script that adds two numbers, simulating stateful work."""
    script = tmp_path / "adder_script.py"
    script.write_text(
        textwrap.dedent("""\
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
        """)
    )
    return script


@pytest.fixture
def error_script(tmp_path: Path) -> Path:
    """A standalone script that raises an error."""
    script = tmp_path / "error_script.py"
    script.write_text(
        textwrap.dedent("""\
        def dispatch(input_dict):
            raise ValueError("intentional test error")
        """)
    )
    return script


@pytest.fixture
def legacy_script(tmp_path: Path) -> Path:
    """A standalone script without dispatch(), using the legacy __main__ pattern."""
    script = tmp_path / "legacy_script.py"
    script.write_text(
        textwrap.dedent("""\
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
        """)
    )
    return script


def _make_worker(script_path: Path, verbose: int = 0) -> PersistentWorker:
    """Create a PersistentWorker using the current Python (no venv needed)."""
    # Use the current Python's directory as a fake venv
    python_dir = Path(sys.executable).parent
    fake_venv = python_dir.parent  # e.g. /usr → /usr/bin/python
    return PersistentWorker(
        toolkit="test",
        env_path=fake_venv,
        script_path=script_path,
        device="cpu",
        verbose=verbose,
    )


# ── Basic send/receive ───────────────────────────────────────────────────────


def test_echo(echo_script: Path):
    worker = _make_worker(echo_script)
    try:
        result = worker.send({"foo": "bar"})
        assert result == {"echo": {"foo": "bar"}}
    finally:
        worker.stop()


def test_tool_env_path_injected(tmp_path: Path):
    """TOOL_VENV_PATH should be set in the subprocess environment."""
    script = tmp_path / "env_script.py"
    script.write_text(
        textwrap.dedent("""\
        import os

        def dispatch(input_dict):
            return {"env_path": os.environ.get("TOOL_VENV_PATH", "")}
        """)
    )
    worker = _make_worker(script)
    try:
        result = worker.send({})
        assert result["env_path"] == str(worker.env_path)
    finally:
        worker.stop()


def test_multiple_calls(adder_script: Path):
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


def test_error_handling(error_script: Path):
    worker = _make_worker(error_script)
    try:
        with pytest.raises(RuntimeError, match="intentional test error"):
            worker.send({"anything": True})
    finally:
        worker.stop()


# ── _drain_stderr: tagged demux + raw handling ──────────────────────────────


@pytest.fixture
def stderr_emitter_script(tmp_path: Path) -> Path:
    """Standalone whose dispatch() writes a sentinel line to stderr."""
    script = tmp_path / "stderr_emitter.py"
    script.write_text(
        textwrap.dedent("""\
        import sys

        def dispatch(input_dict):
            sys.stderr.write(input_dict["msg"] + "\\n")
            sys.stderr.flush()
            return {"ok": True}
        """)
    )
    return script


def _wait_for_drain(worker: PersistentWorker, timeout: float = 2.0) -> None:
    """Stop the worker and join its drain thread for deterministic capture."""
    worker.stop()
    if worker._stderr_thread is not None:
        worker._stderr_thread.join(timeout=timeout)


def test_drain_handle_raw_line_collapses_progress_bar_to_last_frame():
    r"""``\r``-separated frames in one captured line keep only the last segment."""
    import collections as _c

    buf: _c.deque[str] = _c.deque(maxlen=20)
    line = "\rstep 25%\rstep 50%\rstep 75%\rstep 100%\n"
    _handle_raw_stderr_line(line, buf, raw_tee=False)
    assert list(buf) == ["step 100%"]


def test_drain_handle_raw_line_skips_blank_after_collapse():
    r"""A line that's just a bare ``\r`` should produce no ring buffer entry."""
    import collections as _c

    buf: _c.deque[str] = _c.deque(maxlen=20)
    _handle_raw_stderr_line("\r", buf, raw_tee=False)
    assert list(buf) == []


def test_drain_handle_raw_line_tees_to_parent_when_raw_tee_true(capfd):
    """Raw lines should be teed to ``sys.stderr`` when ``raw_tee`` is True."""
    import collections as _c

    buf: _c.deque[str] = _c.deque(maxlen=20)
    _handle_raw_stderr_line("plain stderr line\n", buf, raw_tee=True)
    captured = capfd.readouterr()
    assert "plain stderr line" in captured.err


def test_drain_handle_raw_line_quiet_when_raw_tee_false(capfd):
    """``raw_tee=False`` must keep the ring buffer populated but not write to stderr."""
    import collections as _c

    buf: _c.deque[str] = _c.deque(maxlen=20)
    sentinel = "QUIET_RAW_LINE_xyz"
    _handle_raw_stderr_line(f"{sentinel}\n", buf, raw_tee=False)
    captured = capfd.readouterr()
    assert sentinel not in captured.err
    assert sentinel in buf[-1]


def test_stderr_lines_buffer_is_bounded(monkeypatch: pytest.MonkeyPatch):
    """The per-worker stderr buffer must respect PROTO_WORKER_STDERR_BUFFER_LINES."""
    monkeypatch.setenv("PROTO_WORKER_STDERR_BUFFER_LINES", "5")
    # Construct via the proper init so the buffer reads the env var.
    worker = PersistentWorker(toolkit="test", env_path=Path("/fake"), script_path=Path("/fake/script.py"), device="cpu")
    for i in range(20):
        _handle_raw_stderr_line(f"line {i}\n", worker._stderr_lines, raw_tee=False)
    assert len(worker._stderr_lines) == 5
    # FIFO eviction: only the last 5 survived.
    assert list(worker._stderr_lines) == [f"line {i}" for i in range(15, 20)]


def test_stderr_lines_buffer_default_size(monkeypatch: pytest.MonkeyPatch):
    """Default buffer size is 20 lines when the env var is unset."""
    monkeypatch.delenv("PROTO_WORKER_STDERR_BUFFER_LINES", raising=False)
    worker = PersistentWorker(toolkit="test", env_path=Path("/fake"), script_path=Path("/fake/script.py"), device="cpu")
    assert worker._stderr_lines.maxlen == 20


def test_stderr_lines_buffer_invalid_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    """Malformed env var should fall back to the default rather than crash."""
    monkeypatch.setenv("PROTO_WORKER_STDERR_BUFFER_LINES", "not_an_int")
    worker = PersistentWorker(toolkit="test", env_path=Path("/fake"), script_path=Path("/fake/script.py"), device="cpu")
    assert worker._stderr_lines.maxlen == 20


def test_drain_reemits_tagged_lines_under_worker_namespace(stderr_emitter_script: Path, monkeypatch):
    """A standalone emitting via the bridge handler should re-emit on proto_tools.worker.{toolkit}.*."""
    # Capture parent-side records on the worker namespace.
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    parent = logging.getLogger("proto_tools.worker.test")
    handler = _Capture()
    parent.addHandler(handler)
    parent.setLevel(logging.DEBUG)

    # Standalone that emits a tagged log line via the bridge. Uses the producer
    # convention: ``from standalone_helpers import get_logger`` (the bootstrap
    # copies ``standalone_helpers/`` into the standalone dir, triggers
    # ``install()`` via the package init, and adds the dir to ``sys.path``).
    # The script must live under a ``standalone/`` directory for the helpers
    # copy to fire — see ``_copy_standalone_helpers``.
    standalone_dir = stderr_emitter_script.parent / "standalone"
    standalone_dir.mkdir(exist_ok=True)
    script_with_log = standalone_dir / "tagged_emitter.py"
    script_with_log.write_text(
        textwrap.dedent("""\
        from standalone_helpers import get_logger

        logger = get_logger("demo.module")

        def dispatch(input_dict):
            logger.info(input_dict["msg"], update_status=True)
            return {"ok": True}
        """)
    )
    # verbose=1 so the toolkit logger admits INFO records (level=0 sets WARNING).
    worker = _make_worker(script_with_log, verbose=1)
    sentinel = "TAGGED_REEMIT_xyz"
    try:
        worker.send({"msg": sentinel})
    finally:
        _wait_for_drain(worker)
        parent.removeHandler(handler)

    matches = [r for r in captured if sentinel in r.getMessage()]
    assert matches, f"sentinel not re-emitted; captured names: {[r.name for r in captured]}"
    record = matches[0]
    assert record.name.startswith("proto_tools.worker.test")
    assert getattr(record, "update_status", False) is True


# ── Legacy dispatch ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "op, name, expected",
    [
        ("greet", "Alice", {"greeting": "hello Alice"}),
        ("farewell", "Bob", {"farewell": "goodbye Bob"}),
    ],
)
def test_legacy_dispatch(legacy_script: Path, op, name, expected):
    worker = _make_worker(legacy_script)
    try:
        result = worker.send({"operation": op, "name": name})
        assert result == expected
    finally:
        worker.stop()


def test_legacy_unknown_operation(legacy_script: Path):
    worker = _make_worker(legacy_script)
    try:
        with pytest.raises(RuntimeError, match="Cannot dispatch operation"):
            worker.send({"operation": "nonexistent"})
    finally:
        worker.stop()


# ── Worker lifecycle ─────────────────────────────────────────────────────────


def test_send_cleans_up_when_response_unmatched_even_if_process_exited():
    """On a failed/garbled response the worker is always stopped, even if the process already exited."""
    worker = PersistentWorker(toolkit="t", env_path=Path("/x"), script_path=Path("/y"))
    proc = MagicMock()
    proc.poll.side_effect = [None] + [0] * 20  # alive at send() entry, then reports exited
    proc.stdin = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline.return_value = ""  # closed stdout → RuntimeError, no matching response
    worker._process = proc
    with patch.object(worker, "stop") as mock_stop, pytest.raises(RuntimeError):
        worker.send({"x": 1})
    mock_stop.assert_called_once()


def test_stop_and_restart(echo_script: Path):
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


def test_alive_property(echo_script: Path):
    worker = _make_worker(echo_script)
    assert not worker.alive
    worker.start()
    assert worker.alive
    worker.stop()
    assert not worker.alive


def test_send_kills_worker_on_stale_frame(tmp_path: Path):
    """Mismatched-id raises AND kills the worker; the next dispatch must spawn fresh.

    Simulates the production scenario where an upstream-cancelled prior ``send()`` left
    a stale response in the pipe. The script writes a stale PROTO_LENGTH frame via
    ``sys.__stdout__`` ahead of the real reply — same byte layout the subprocess
    produces when it finishes after the parent has unwound out of a cancelled call.
    """
    script = tmp_path / "stale.py"
    script.write_text(
        textwrap.dedent("""\
        import json, sys

        def dispatch(input_dict):
            body = json.dumps({"id": "deadbeef", "result": {"data": "stale"}}, separators=(",", ":"))
            sys.__stdout__.write(f"PROTO_LENGTH:{len(body)}\\n{body}")
            sys.__stdout__.flush()
            return {"data": "real"}
        """)
    )

    worker = _make_worker(script)
    try:
        with pytest.raises(RuntimeError, match="mismatched id"):
            worker.send({"x": 1})
        assert not worker.alive

        clean = tmp_path / "clean.py"
        clean.write_text("def dispatch(input_dict):\n    return {'echo': input_dict}\n")
        worker.script_path = clean
        assert worker.send({"x": 2}) == {"echo": {"x": 2}}
    finally:
        worker.stop()


# ── Process group cleanup ───────────────────────────────────────────────────


def test_stop_signals_process_group():
    """stop() should send SIGTERM to the process group, not just the process."""
    worker = PersistentWorker.__new__(PersistentWorker)
    worker.toolkit = "test"

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 99999
    mock_process.stdin = MagicMock()
    worker._process = mock_process

    with patch("proto_tools.utils.persistent_worker.os.killpg") as mock_killpg:
        worker.stop()

    # SIGTERM should be sent to the process group, not process.terminate()
    mock_killpg.assert_any_call(99999, signal.SIGTERM)
    mock_process.terminate.assert_not_called()
    assert worker._process is None


def test_stop_escalates_to_sigkill():
    """stop() should SIGKILL the group if SIGTERM + wait fails."""
    worker = PersistentWorker.__new__(PersistentWorker)
    worker.toolkit = "test"

    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_process.pid = 99999
    mock_process.stdin = MagicMock()
    # First wait (after SIGTERM) times out, second wait (after SIGKILL) succeeds
    mock_process.wait.side_effect = [Exception("timed out"), None]
    worker._process = mock_process

    with patch("proto_tools.utils.persistent_worker.os.killpg") as mock_killpg:
        worker.stop()

    calls = [c.args for c in mock_killpg.call_args_list]
    assert (99999, signal.SIGTERM) in calls
    assert (99999, signal.SIGKILL) in calls
    assert worker._process is None


def test_stop_kills_child_processes(tmp_path: Path):
    """stop() should kill subprocesses spawned by the worker script."""
    # Script that forks a long-lived child and reports its PID
    script = tmp_path / "forking_script.py"
    script.write_text(
        textwrap.dedent("""\
        import subprocess, sys

        def dispatch(input_dict):
            # Spawn a long-lived child process
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(3600)"],
            )
            return {"child_pid": child.pid}
        """)
    )

    worker = _make_worker(script)
    try:
        result = worker.send({})
        child_pid = result["child_pid"]

        # Child should be alive
        os.kill(child_pid, 0)  # raises ProcessLookupError if dead

        worker.stop()

        # Both the worker and its child should be dead
        assert not worker.alive
        # Poll briefly; the child may take a moment to be reaped after
        # the process group receives SIGTERM.
        for _ in range(50):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        with pytest.raises(ProcessLookupError, match="No such process"):
            os.kill(child_pid, 0)
    finally:
        worker.stop()


# ── Serialization ────────────────────────────────────────────────────────────


def test_serialize_tensor_like(tmp_path: Path):
    """_serialize() should handle objects with .detach(), .cpu(), .tolist()."""
    script = tmp_path / "tensor_script.py"
    script.write_text(
        textwrap.dedent("""\
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
    """)
    )
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


# ── _parse_env_vars_file ─────────────────────────────────────────────────────


_EMPTY_RESULT = {"passthrough": [], "set": [], "no_passthrough": []}


def test_parse_env_vars_none_path():
    result = _parse_env_vars_file(None)
    assert result == _EMPTY_RESULT


def test_parse_env_vars_missing_file(tmp_path: Path):
    result = _parse_env_vars_file(tmp_path / "nonexistent.txt")
    assert result == _EMPTY_RESULT


def test_parse_env_vars_empty_file(tmp_path: Path):
    f = tmp_path / "env_vars.txt"
    f.write_text("")
    result = _parse_env_vars_file(f)
    assert result == _EMPTY_RESULT


@pytest.mark.parametrize(
    "section_text, expected",
    [
        ("[passthrough]\nHF_TOKEN\nHF_HOME\n", {"passthrough": ["HF_TOKEN", "HF_HOME"]}),
        ("[set]\nMY_VAR=${VENV_PATH}/data\n", {"set": ["MY_VAR=${VENV_PATH}/data"]}),
        ("[no_passthrough]\nLD_LIBRARY_PATH\nHF_TOKEN\n", {"no_passthrough": ["LD_LIBRARY_PATH", "HF_TOKEN"]}),
    ],
    ids=["passthrough", "set", "no_passthrough"],
)
def test_parse_env_vars_single_section(tmp_path: Path, section_text: str, expected: dict[str, list[str]]):
    """Each section parses correctly in isolation; other sections stay empty."""
    f = tmp_path / "env_vars.txt"
    f.write_text(section_text)
    result = _parse_env_vars_file(f)
    assert result == {**_EMPTY_RESULT, **expected}


def test_parse_env_vars_all_sections(tmp_path: Path):
    """All three sections in one file land in their own buckets."""
    f = tmp_path / "env_vars.txt"
    f.write_text("[passthrough]\nHF_TOKEN\n\n[set]\nFOO=${VENV_PATH}/bar\n\n[no_passthrough]\nLD_LIBRARY_PATH\n")
    result = _parse_env_vars_file(f)
    assert result == {
        "passthrough": ["HF_TOKEN"],
        "set": ["FOO=${VENV_PATH}/bar"],
        "no_passthrough": ["LD_LIBRARY_PATH"],
    }


def test_parse_env_vars_comments_and_blank_lines(tmp_path: Path):
    f = tmp_path / "env_vars.txt"
    f.write_text("# This is a comment\n\n[passthrough]\n# Another comment\nHF_TOKEN\n\nHF_HOME\n")
    result = _parse_env_vars_file(f)
    assert result["passthrough"] == ["HF_TOKEN", "HF_HOME"]


def test_parse_env_vars_unknown_section_warns(tmp_path: Path, caplog):
    f = tmp_path / "env_vars.txt"
    f.write_text("[bogus]\nFOO\n")
    with caplog.at_level(logging.WARNING):
        result = _parse_env_vars_file(f)
    assert "Unknown section" in caplog.text
    assert result == _EMPTY_RESULT


# ── _build_subprocess_env (whitelist-based) ──────────────────────────────────


def test_non_whitelisted_vars_are_absent(monkeypatch):
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


def test_base_whitelist_vars_present(monkeypatch):
    """Whitelisted vars should be passed through when set."""
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")

    env = _build_subprocess_env(device="cpu")

    assert env["HOME"] == "/home/test"
    assert env["LANG"] == "en_US.UTF-8"
    assert env["HTTP_PROXY"] == "http://proxy:8080"


def test_missing_whitelist_vars_not_added(monkeypatch):
    """Whitelisted vars not in parent env should not appear."""
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)

    env = _build_subprocess_env(device="cpu")

    assert "HTTP_PROXY" not in env
    assert "HTTPS_PROXY" not in env


def test_uv_pip_cache_defaults_under_proto_home(monkeypatch, tmp_path: Path):
    """UV_CACHE_DIR and PIP_CACHE_DIR default to PROTO_HOME/{uv,pip}_cache."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
    get_proto_home.cache_clear()

    env = _build_subprocess_env(device="cpu")

    assert env["UV_CACHE_DIR"] == str(tmp_path / "uv_cache")
    assert env["PIP_CACHE_DIR"] == str(tmp_path / "pip_cache")


def test_uv_pip_cache_user_override_preserved(monkeypatch, tmp_path: Path):
    """User-set UV_CACHE_DIR / PIP_CACHE_DIR win over the PROTO_HOME default."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    monkeypatch.setenv("UV_CACHE_DIR", "/custom/uv")
    monkeypatch.setenv("PIP_CACHE_DIR", "/custom/pip")
    get_proto_home.cache_clear()

    env = _build_subprocess_env(device="cpu")

    assert env["UV_CACHE_DIR"] == "/custom/uv"
    assert env["PIP_CACHE_DIR"] == "/custom/pip"


@pytest.mark.parametrize("device, expect_cuda", [("cpu", False), ("cuda", True)])
def test_path_ordering(monkeypatch, tmp_path: Path, device, expect_cuda):
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


def test_path_without_conda_prefix(monkeypatch, tmp_path: Path):
    """Without CONDA_PREFIX, no conda bin in PATH; parent PATH still carried over."""
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setenv("PATH", "/opt/hpc/bin:/usr/bin:/bin")

    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)

    path_parts = env["PATH"].split(":")
    assert path_parts[0] == str(tmp_path / "bin")
    assert "/opt/hpc/bin" in path_parts  # parent PATH entry carried over


@pytest.mark.parametrize(
    "has_tool_env",
    [True, False],
    ids=["with_tool_env", "without_tool_env"],
)
def test_conda_prefix_and_virtual_env(monkeypatch, tmp_path: Path, has_tool_env):
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_parent_ld_library_path_inherited(monkeypatch, device):
    """Parent LD_LIBRARY_PATH entries are carried over for all devices."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda-12.4/lib64:/usr/lib64:/opt/nvidia/lib64")
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
def test_ld_library_path_conda_auto(monkeypatch, conda_prefix, expect_ld):
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
def test_ld_library_path_via_set_directive(monkeypatch, tmp_path: Path, conda_prefix, expect_conda_lib):
    """[set] LD_LIBRARY_PATH + parent LD + conda lib (when set)."""
    if conda_prefix:
        monkeypatch.setenv("CONDA_PREFIX", conda_prefix)
    else:
        monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    tool_env_vars = {
        "passthrough": [],
        "set": ["LD_LIBRARY_PATH=${VENV_PATH}/cuda_env/lib:${VENV_PATH}/cuda_env/lib64"],
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


def test_no_passthrough_blocks_whitelisted_var(monkeypatch):
    """[no_passthrough] also blocks whitelisted vars (e.g. HTTP_PROXY) from parent."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
    tool_env_vars = {"passthrough": [], "set": [], "no_passthrough": ["HTTP_PROXY"]}

    env = _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

    assert "HTTP_PROXY" not in env


def test_no_passthrough_blocks_parent_ld_library_path(monkeypatch, tmp_path: Path):
    """[no_passthrough] LD_LIBRARY_PATH must not let parent's value leak into the subprocess."""
    from proto_tools.utils.persistent_worker import _find_driver_lib_dir

    _find_driver_lib_dir.cache_clear()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/opt/nvidia/lib64")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    tool_env_vars = {"passthrough": [], "set": [], "no_passthrough": ["LD_LIBRARY_PATH"]}

    env = _build_subprocess_env(device="cuda", tool_env_path=tmp_path, tool_env_vars=tool_env_vars)

    ld = env.get("LD_LIBRARY_PATH", "")
    assert "/usr/local/cuda/lib64" not in ld
    assert "/opt/nvidia/lib64" not in ld


@pytest.mark.parametrize(
    "has_venv, expect_torch_home",
    [(True, True), (False, False)],
    ids=["with_venv", "without_venv"],
)
def test_torch_home(monkeypatch, tmp_path: Path, has_venv, expect_torch_home):
    """TORCH_HOME = {venv}/cache/torch when venv provided (IN_ENV mode)."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "IN_ENV")
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
def test_jax_platforms_not_set(device):
    """JAX_PLATFORMS should NOT be set for any device (allows later GPU access via device_put)."""
    env = _build_subprocess_env(device=device)
    assert "JAX_PLATFORMS" not in env


def test_xla_preallocation_disabled_for_cpu():
    """CPU device should disable JAX preallocation."""
    env = _build_subprocess_env(device="cpu")

    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


def test_xla_preallocation_disabled_for_cuda():
    """GPU device should also disable JAX preallocation (DeviceManager handles placement)."""
    env = _build_subprocess_env(device="cuda")

    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


@pytest.mark.parametrize(
    "parent_has_var",
    [True, False],
    ids=["present", "absent"],
)
def test_passthrough_vars(monkeypatch, parent_has_var):
    """Tool-specific passthrough vars appear only when set in parent env."""
    if parent_has_var:
        monkeypatch.setenv("HF_TOKEN", "secret-token")
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        # Prevent resolve_hf_token() from finding file-based tokens
        monkeypatch.setattr(
            "proto_tools.utils.auth.os.path.expanduser",
            lambda p: "/nonexistent" + p,
        )

    tool_env_vars = {"passthrough": ["HF_TOKEN"], "set": []}
    env = _build_subprocess_env(device="cpu", tool_env_vars=tool_env_vars)

    if parent_has_var:
        assert env["HF_TOKEN"] == "secret-token"
    else:
        assert "HF_TOKEN" not in env


def test_passthrough_missing_var_warns(monkeypatch, caplog):
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
def test_set_vars(tmp_path: Path, set_line, var, expected_suffix):
    """[set] entries: interpolate ${VENV_PATH} or pass through literally."""
    tool_env_vars = {"passthrough": [], "set": [set_line]}
    env = _build_subprocess_env(
        device="cpu",
        tool_env_path=tmp_path,
        tool_env_vars=tool_env_vars,
    )

    if expected_suffix:
        assert env[var] == f"{tmp_path}{expected_suffix}"
    else:
        assert env[var] == set_line.split("=", 1)[1]


# ── env_overrides (caller-supplied final layer) ─────────────────────────────


def test_env_overrides_applied_to_subprocess_env(monkeypatch):
    """Caller-supplied env_overrides land in the built env."""
    env = _build_subprocess_env(
        device="cpu",
        env_overrides={
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        },
    )
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["MKL_NUM_THREADS"] == "4"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["NUMEXPR_NUM_THREADS"] == "4"


def test_env_overrides_win_over_passthrough(monkeypatch):
    """env_overrides applied after _BASE_PASSTHROUGH overrides whitelisted vars."""
    monkeypatch.setenv("HOME", "/home/parent")
    env = _build_subprocess_env(device="cpu", env_overrides={"HOME": "/home/override"})
    assert env["HOME"] == "/home/override"


# ── Compute environment injection ───────────────────────────────────────────


@pytest.fixture
def _clear_compute_caches():
    """Clear LRU caches before and after each test to ensure mocks work correctly."""
    from proto_tools.utils.compute_deps import detect_compute_environment
    from proto_tools.utils.system_info import get_gpu_info

    get_gpu_info.cache_clear()
    detect_compute_environment.cache_clear()
    yield
    get_gpu_info.cache_clear()
    detect_compute_environment.cache_clear()


@pytest.mark.usefixtures("_clear_compute_caches")
def test_compute_env_vars_present_gpu(monkeypatch):
    """On GPU systems, compute env vars should be present."""
    from proto_tools.utils.system_info import GPUDevice, GPUInfo

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
            "proto_tools.utils.system_info.get_gpu_info",
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


@pytest.mark.usefixtures("_clear_compute_caches")
def test_compute_env_vars_present_cpu(monkeypatch):
    """On CPU systems, compute env vars should be present (simplified)."""
    from proto_tools.utils.system_info import GPUInfo

    fake_gpu_info = GPUInfo(
        available=False,
        count=0,
        driver_version=None,
        cuda_version=None,
        devices=[],
    )

    with monkeypatch.context() as m:
        m.setattr(
            "proto_tools.utils.system_info.get_gpu_info",
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


@pytest.mark.usefixtures("_clear_compute_caches")
def test_compute_env_vars_can_be_overridden_by_tool(monkeypatch, tmp_path: Path):
    """Tool-specific env vars can override compute env recommendations."""
    from proto_tools.utils.system_info import GPUDevice, GPUInfo

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
            "proto_tools.utils.system_info.get_gpu_info",
            lambda: fake_gpu_info,
        )
        env = _build_subprocess_env(
            device="cuda",
            tool_env_path=tmp_path,
            tool_env_vars=tool_env_vars,
        )

    # Tool override should win
    assert env["RECOMMENDED_TORCH_SPEC"] == "torch==2.6.0"


# ── Helper file copy ────────────────────────────────────────────────────────


def test_helpers_copied_on_worker_startup(tmp_path: Path, echo_script):
    """Verify standalone_helpers package is copied to standalone directory on worker startup."""
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

    # Verify standalone_helpers package doesn't exist yet
    helpers_path = standalone_dir / "standalone_helpers"
    assert not helpers_path.exists(), "standalone_helpers package should not exist before worker starts"

    # Start the worker
    worker = PersistentWorker(
        toolkit="test-tool",
        env_path=fake_env,
        script_path=script_path,
    )

    try:
        worker.start()

        # Call send to ensure worker has fully started
        result = worker.send({"test": "data"})
        assert result["echo"]["test"] == "data", "Worker should be functional"

        # Verify standalone_helpers package was copied
        assert helpers_path.is_dir(), "standalone_helpers package should be copied on worker startup"

        # Verify the package has __init__.py and expected submodules
        assert (helpers_path / "__init__.py").exists(), "standalone_helpers/__init__.py should exist"
        for submodule in ("device.py", "memory.py", "seeding.py", "weights.py", "compression.py"):
            assert (helpers_path / submodule).exists(), f"standalone_helpers/{submodule} should exist"

        # Verify __init__.py content matches source exactly
        assert _STANDALONE_HELPERS_SOURCE.is_dir(), "source standalone_helpers package must exist"
        source_init = (_STANDALONE_HELPERS_SOURCE / "__init__.py").read_text()
        copied_init = (helpers_path / "__init__.py").read_text()
        assert copied_init == source_init, "Copied standalone_helpers/__init__.py should be identical to source"

    finally:
        worker.stop()


def test_helpers_not_copied_outside_standalone(tmp_path: Path):
    """Verify standalone_helpers package is not copied if script is not in a standalone/ directory."""
    # Create a script in a non-standalone location
    script = tmp_path / "script_not_in_standalone.py"
    script.write_text(
        textwrap.dedent("""\
        def dispatch(input_dict):
            return {"result": "ok"}
        """)
    )

    # Create a minimal fake tool environment
    fake_env = tmp_path / "fake_env"
    fake_env.mkdir()
    (fake_env / "bin").mkdir()

    # Create a Python executable symlink
    python_exe = fake_env / "bin" / "python"
    python_exe.symlink_to(sys.executable)

    # Start worker with script NOT in standalone/ directory
    worker = PersistentWorker(
        toolkit="test-tool",
        env_path=fake_env,
        script_path=script,
    )

    try:
        worker.start()

        # Worker should still function (just without standalone_helpers)
        result = worker.send({"test": "data"})
        assert result["result"] == "ok"

        # Verify standalone_helpers package was NOT copied
        helpers_path = tmp_path / "standalone_helpers"
        assert not helpers_path.exists(), (
            "standalone_helpers package should not be copied for scripts outside standalone/ directories"
        )

    finally:
        worker.stop()


# ── File-based fallback for large responses ──────────────────────────────────


def test_send_response_small_payload_uses_length_protocol():
    """Small payloads should use the PROTO_LENGTH: pipe protocol."""
    from io import StringIO

    from proto_tools.utils._worker_bootstrap import _send_response

    buf = StringIO()
    payload = '{"id":"abc","result":{"x":1}}'
    _send_response(buf, payload)

    output = buf.getvalue()
    assert output.startswith("PROTO_LENGTH:")
    header, body = output.split("\n", 1)
    assert header == f"PROTO_LENGTH:{len(payload)}"
    assert body == payload


def test_send_response_large_payload_uses_file_protocol(monkeypatch):
    """Payloads above the threshold should use the PROTO_FILE: protocol."""
    import io

    import proto_tools.utils._worker_bootstrap as _wb

    # Use a tiny threshold so we don't allocate 100MB in tests
    monkeypatch.setattr(_wb, "_FILE_FALLBACK_THRESHOLD", 1000)

    buf = io.StringIO()
    padding = "x" * 1100
    payload = json.dumps({"id": "abc", "result": {"data": padding}}, separators=(",", ":"))
    assert len(payload) > 1000

    _wb._send_response(buf, payload)

    output = buf.getvalue()
    assert output.startswith("PROTO_FILE:")
    file_path = output.strip().split(":", 1)[1]

    # File should exist with the correct content
    assert Path(file_path).exists()
    with open(file_path) as f:
        assert f.read() == payload

    # Clean up
    os.unlink(file_path)


@pytest.mark.slow
def test_file_fallback_end_to_end(tmp_path: Path):
    """End-to-end: worker returns a large payload via the PROTO_FILE: protocol."""
    # Use a tiny threshold (1KB) so we don't allocate 100MB in tests.
    # The threshold is set inside the worker subprocess via the script.
    script = tmp_path / "large_script.py"
    script.write_text(
        textwrap.dedent("""\
        import proto_tools.utils._worker_bootstrap as _wb
        _wb._FILE_FALLBACK_THRESHOLD = 1000

        def dispatch(input_dict):
            return {"data": "x" * 2000}
        """)
    )

    worker = _make_worker(script)
    try:
        result = worker.send({})
        assert "data" in result
        assert len(result["data"]) == 2000
        assert result["data"] == "x" * 2000
    finally:
        worker.stop()


# ── PROTO_MODEL_CACHE and HF_HOME ─────────────────────────────────────────────


def test_default_sets_hf_home_to_proto_model_cache(monkeypatch, tmp_path: Path):
    """Default mode sets HF_HOME to {PROTO_HOME}/proto_model_cache/huggingface/."""
    monkeypatch.delenv("PROTO_MODEL_CACHE", raising=False)
    monkeypatch.setenv("PROTO_HOME", str(tmp_path / "proto_home"))
    # Clear the lru_cache so monkeypatched PROTO_HOME takes effect
    from proto_tools.utils.proto_home import get_proto_home

    get_proto_home.cache_clear()
    try:
        env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
        assert env["HF_HOME"] == str(tmp_path / "proto_home" / "proto_model_cache" / "huggingface")
    finally:
        get_proto_home.cache_clear()


def test_in_env_sets_hf_home(monkeypatch, tmp_path: Path):
    """IN_ENV mode sets HF_HOME to {venv}/cache/huggingface/."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "IN_ENV")
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["HF_HOME"] == str(tmp_path / "cache" / "huggingface")


def test_shared_path_sets_hf_home(monkeypatch, tmp_path: Path):
    """Absolute path mode sets HF_HOME to /path/huggingface/."""
    shared = str(tmp_path / "shared")
    monkeypatch.setenv("PROTO_MODEL_CACHE", shared)
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["HF_HOME"] == str(Path(shared) / "huggingface")


def test_shared_path_overrides_torch_home(monkeypatch, tmp_path: Path):
    """Absolute path mode overrides TORCH_HOME to /path/torch/."""
    shared = str(tmp_path / "shared")
    monkeypatch.setenv("PROTO_MODEL_CACHE", shared)
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["TORCH_HOME"] == str(Path(shared) / "torch")


def test_none_mode_passes_through_hf_home(monkeypatch, tmp_path: Path):
    """NONE mode passes through parent's HF_HOME."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "NONE")
    monkeypatch.setenv("HF_HOME", "/custom/hf/cache")
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["HF_HOME"] == "/custom/hf/cache"


def test_none_mode_no_hf_home_override(monkeypatch, tmp_path: Path):
    """NONE mode without parent HF_HOME does not set it."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "NONE")
    monkeypatch.delenv("HF_HOME", raising=False)
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert "HF_HOME" not in env


def test_proto_weights_dir_passthrough(monkeypatch):
    """PROTO_*_WEIGHTS_DIR vars are passed through to subprocess."""
    monkeypatch.setenv("PROTO_FAMPNN_WEIGHTS_DIR", "/custom/fampnn")
    monkeypatch.setenv("PROTO_ESM_IF1_WEIGHTS_DIR", "/custom/esmif")

    env = _build_subprocess_env(device="cpu")

    assert env["PROTO_FAMPNN_WEIGHTS_DIR"] == "/custom/fampnn"
    assert env["PROTO_ESM_IF1_WEIGHTS_DIR"] == "/custom/esmif"


def test_proto_weights_mode_passthrough(monkeypatch, tmp_path: Path):
    """PROTO_MODEL_CACHE is passed through to subprocess."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "/shared/weights")
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["PROTO_MODEL_CACHE"] == "/shared/weights"


def test_hf_hub_cache_not_in_env(monkeypatch):
    """HF_HUB_CACHE should NOT be passed through (removed from passthrough list)."""
    monkeypatch.setenv("HF_HUB_CACHE", "/old/cache")
    env = _build_subprocess_env(device="cpu")
    assert "HF_HUB_CACHE" not in env


def test_hf_token_resolved_from_file(monkeypatch, tmp_path: Path):
    """HF_TOKEN is set in subprocess env when token exists only as a file."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    get_proto_home.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    token_file = tmp_path / "token"
    token_file.write_text("hf_test_token_from_file")
    monkeypatch.setattr(
        "proto_tools.utils.auth.os.path.expanduser",
        lambda p: str(token_file) if "huggingface/token" in p else p,
    )
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["HF_TOKEN"] == "hf_test_token_from_file"


def test_hf_token_resolved_from_git_credentials(monkeypatch, tmp_path: Path):
    """HF_TOKEN is set in subprocess env when token exists only in git-credentials."""
    monkeypatch.setenv("PROTO_HOME", str(tmp_path))
    get_proto_home.cache_clear()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    git_creds = tmp_path / "git-credentials"
    git_creds.write_text("https://user:hf_git_cred_token@huggingface.co\n")
    monkeypatch.setattr(
        "proto_tools.utils.auth.os.path.expanduser",
        lambda p: (
            str(git_creds)
            if "git-credentials" in p
            else str(tmp_path / "nonexistent")
            if "huggingface/token" in p
            else p
        ),
    )
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["HF_TOKEN"] == "hf_git_cred_token"


def test_hf_token_env_var_takes_precedence(monkeypatch, tmp_path: Path):
    """HF_TOKEN env var is passed through without file resolution."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert env["HF_TOKEN"] == "hf_from_env"


def test_hf_token_not_set_when_absent(monkeypatch, tmp_path: Path):
    """HF_TOKEN is not set when no token exists anywhere."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "proto_tools.utils.auth.os.path.expanduser",
        lambda p: str(tmp_path / "nonexistent"),
    )
    env = _build_subprocess_env(device="cpu", tool_env_path=tmp_path)
    assert "HF_TOKEN" not in env
