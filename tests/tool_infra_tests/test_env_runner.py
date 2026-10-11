"""Tests for run_in_env: executing code inside a tool's isolated environment."""

import json
from pathlib import Path

import pytest

from proto_tools.utils import run_in_env


def test_requires_exactly_one_of_code_or_script() -> None:
    """Passing both or neither of code/script raises before any env is touched."""
    with pytest.raises(ValueError, match="exactly one"):
        run_in_env("esm2", code="print(1)", script="x.py")
    with pytest.raises(ValueError, match="exactly one"):
        run_in_env("esm2")


@pytest.mark.integration
def test_nonzero_exit_raises() -> None:
    """A non-zero exit in the tool env surfaces as a RuntimeError with stderr."""
    with pytest.raises(RuntimeError, match="failed"):
        run_in_env("esm2", code="import sys; sys.exit(3)")


@pytest.mark.integration
def test_logs_demux_to_parent_and_stdout_stays_clean(caplog: pytest.LogCaptureFixture) -> None:
    """In-env proto-logging re-emits on the parent logger; stdout stays pure data."""
    code = (
        "import json\n"
        "from standalone_helpers import get_logger\n"
        "get_logger('rie').info('hello from env')\n"
        "print(json.dumps({'ok': True}))\n"
    )
    with caplog.at_level("INFO", logger="proto_tools.worker.esm2"):
        out = run_in_env("esm2", code=code)
    assert json.loads(out) == {"ok": True}
    assert any("hello from env" in r.getMessage() for r in caplog.records)


@pytest.mark.integration
def test_runs_script_file_in_env(tmp_path: Path) -> None:
    """The script= path runs a file, forwards args as sys.argv[1:], and exports RUN_IN_ENV_DEVICE."""
    script = tmp_path / "prog.py"
    script.write_text(
        "import json, os, sys; print(json.dumps({'argv': sys.argv[1:], 'device': os.environ['RUN_IN_ENV_DEVICE']}))"
    )
    out = run_in_env("esm2", script=script, args=["a", "b"], device="cpu")
    assert json.loads(out) == {"argv": ["a", "b"], "device": "cpu"}


@pytest.mark.integration
def test_timeout_raises() -> None:
    """A run that exceeds timeout is killed and surfaces as TimeoutError."""
    with pytest.raises(TimeoutError, match="timed out"):
        run_in_env("esm2", code="import time; time.sleep(30)", timeout=3)


@pytest.mark.integration
def test_jax_compilation_cache_reaches_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The tool env's JAX reads the parent-chosen cache directory and writes compiled programs to it."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", str(tmp_path))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.delenv("PROTO_JAX_COMPILATION_CACHE", raising=False)
    code = (
        "import jax, jax.numpy as jnp\n"
        "jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)\n"
        "jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)\n"
        "jax.jit(lambda x: jnp.sin(x) @ x.T)(jnp.ones((8, 8))).block_until_ready()\n"
        "print(jax.config.jax_compilation_cache_dir)\n"
    )

    out = run_in_env("mock_jax_tool", code=code)

    cache_dir = tmp_path / "jax_cache" / "mock_jax_tool"
    assert out.strip() == str(cache_dir)
    assert any(cache_dir.iterdir())
