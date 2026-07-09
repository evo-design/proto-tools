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
def test_runs_code_string_in_env() -> None:
    """A code string runs in the tool env and its stdout is returned."""
    out = run_in_env("esm2", code="import torch, json; print(json.dumps({'torch': torch.__version__}))")
    assert json.loads(out)["torch"]


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
    """The script= path runs a file and forwards args as sys.argv[1:]."""
    script = tmp_path / "prog.py"
    script.write_text("import sys, json; print(json.dumps(sys.argv[1:]))")
    out = run_in_env("esm2", script=script, args=["a", "b"])
    assert json.loads(out) == ["a", "b"]


@pytest.mark.integration
def test_exposes_resolved_device_to_program() -> None:
    """The resolved device is exported to the program as RUN_IN_ENV_DEVICE."""
    out = run_in_env("esm2", code="import os; print(os.environ['RUN_IN_ENV_DEVICE'])", device="cpu")
    assert out.strip() == "cpu"


@pytest.mark.integration
def test_timeout_raises() -> None:
    """A run that exceeds timeout is killed and surfaces as TimeoutError."""
    with pytest.raises(TimeoutError, match="timed out"):
        run_in_env("esm2", code="import time; time.sleep(30)", timeout=3)
