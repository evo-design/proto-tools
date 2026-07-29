"""Tests for the pytest log filename derived from a ``-k`` selection expression.

The filename is built from the expression itself, so a long selection once produced a
name beyond the 255-byte limit that filesystems impose. Every selected test then failed
at setup with ``OSError: [Errno 36] File name too long``, which reported a logging
problem as a test failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import log_filename_for_k_expression

# The limit ext4, XFS, APFS, and NTFS each impose on a single filename.
_FILENAME_LIMIT = 255


def test_short_expression_is_used_verbatim():
    """A selection that fits keeps its readable name."""
    assert log_filename_for_k_expression("test_alpha or test_beta") == "pytest_test_alpha_or_test_beta.log"


def test_special_characters_are_stripped():
    """Characters that are not filename-safe are removed or replaced."""
    assert log_filename_for_k_expression("test_a and not (test_b)") == "pytest_test_a_and_not_test_b.log"


@pytest.mark.parametrize("repeats", [1, 8, 10, 100, 5000], ids=lambda n: f"{n}_terms")
def test_filename_stays_within_the_byte_limit(repeats):
    """No selection, however long, produces a name the filesystem will reject."""
    expression = " or ".join(["test_iterable_tools_found"] * repeats)

    filename = log_filename_for_k_expression(expression)

    assert len(filename.encode()) <= _FILENAME_LIMIT


def test_distinct_long_expressions_get_distinct_filenames():
    """Truncation appends a hash, so two long selections do not share one log file."""
    prefix = " or ".join(["test_iterable_tools_found"] * 20)

    first = log_filename_for_k_expression(f"{prefix} or test_alpha")
    second = log_filename_for_k_expression(f"{prefix} or test_beta")

    assert first != second


def test_truncation_is_deterministic():
    """The same selection resolves to the same log file across runs."""
    expression = " or ".join(["test_iterable_tools_found"] * 20)

    assert log_filename_for_k_expression(expression) == log_filename_for_k_expression(expression)


# ── End to end ──────────────────────────────────────────────────────────────

_TARGET = "test_short_expression_is_used_verbatim"


def _run_pytest_with_selection(expression: str, log_dir: Path) -> subprocess.CompletedProcess:
    """Run pytest in a subprocess under the repository conftest with the given selection."""
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PROTO_LOG_DIR": str(log_dir), "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-k", expression, "-q", "-p", "no:randomly", "--no-header"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.slow
def test_a_long_selection_runs_and_writes_its_log(tmp_path):
    """A selection far beyond the filename limit completes and produces a log file."""
    expression = " or ".join([_TARGET] + ["test_nonexistent_padding_name"] * 40)
    assert len(expression) > _FILENAME_LIMIT

    result = _run_pytest_with_selection(expression, tmp_path)

    assert "File name too long" not in result.stdout + result.stderr
    assert result.returncode == 0, result.stdout[-2000:]
    assert list(tmp_path.glob("pytest_*.log")), "the run produced no log file"


@pytest.mark.slow
def test_a_short_selection_still_names_its_log_after_the_expression(tmp_path):
    """The readable filename is preserved for selections that fit."""
    result = _run_pytest_with_selection(_TARGET, tmp_path)

    assert result.returncode == 0, result.stdout[-2000:]
    assert (tmp_path / f"pytest_{_TARGET}.log").exists()
