"""tests/tool_infra_tests/test_list_incomplete_tests.py.

The pytest log header round-trips through ``scripts/list_incomplete_tests.py``.
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "list_incomplete_tests.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("list_incomplete_tests", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_multi_word_keyword_expression_survives_the_log_header(tmp_path: Path) -> None:
    """The header conftest writes must give back the same arguments, quoting included."""
    argv = ["/env/bin/pytest", "tests/seed_tests", "--all", "-k", "esm2 or (boltz2 and not slow)", "-p", "no:x"]
    log = tmp_path / "pytest_run.log"
    log.write_text(f"Pytest Run Command: `{shlex.join(argv)}`\nRun Started: 00:00:00\n")
    assert _load_script().parse_pytest_args(log) == argv[1:]


def test_collection_runs_under_the_scripts_own_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different pytest on PATH can build other node IDs, which the run's pytest then rejects."""
    script = _load_script()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="RUNNABLE: tests/test_x.py::test_y[seq0]\n", stderr="")

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    assert script.collect_runnable_node_ids(["--all"]) == {"tests/test_x.py::test_y[seq0]"}
    assert calls[0][:3] == [sys.executable, "-m", "pytest"]
