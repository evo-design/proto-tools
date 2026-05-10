#!/usr/bin/env python3
"""scripts/list_incomplete_tests.py.

Given a pytest log file, list every test that pytest *would have
actually run* on this machine but for which no ``TEST PASSED`` /
``TEST FAILED`` / ``TEST ERROR`` line was emitted before the run was
killed.

The log's first line records the exact pytest invocation, e.g.::

    Pytest Run Command: `/path/to/pytest --all --ext -sv`

We re-run that command via the inline plugin in
``scripts/_pytest_collect_runnable.py``, which performs full collection
(so the project conftest applies its custom-marker-driven skips:
``uses_gpu``, ``skip_ci``, ``test_on_platforms``, missing assets, …)
and then evaluates every ``skip``/``skipif`` marker. Tests that would
be skipped are excluded; the survivors form the canonical "should have
run" set. Subtracting the IDs with a terminal log line yields the
tests that never ran or were killed mid-flight.

Usage::

    python scripts/list_incomplete_tests.py logs/pytest_20260509_223444.log -o incomplete.txt
    pytest --all @incomplete.txt

With no log argument, the most recently modified ``logs/pytest_*.log``
is used.
"""

from __future__ import annotations

import argparse
import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

# Header line written by the test logging setup, e.g.:
#   Pytest Run Command: `/home/.../bin/pytest --all --ext -sv`
COMMAND_HEADER_RE = re.compile(r"Pytest Run Command:\s*`([^`]+)`")

# Mid-log per-test status lines, e.g.:
#   2026-05-09 22:34:44 | DEBUG    | proto_tools.tests:377 | TEST PASSED: tests/foo/test_bar.py::test_baz
TEST_OUTCOME_RE = re.compile(r"TEST (PASSED|FAILED|ERROR):\s*(.+?)\s*$")

# Flags from the original pytest invocation that don't affect collection
# but make collect-only output noisier (verbosity / capture toggles).
FLAGS_TO_STRIP = {"-v", "-vv", "-vvv", "-q", "-qq", "-s", "-sv", "-vs"}

logger = logging.getLogger(__name__)


def find_latest_log() -> Path:
    """Return the most recently modified ``logs/pytest_*.log`` file."""
    candidates = sorted(LOGS_DIR.glob("pytest_*.log"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No pytest_*.log files found in {LOGS_DIR}")
    return candidates[-1]


def parse_pytest_args(log_path: Path) -> list[str]:
    """Pull the pytest command from the log header and return its args.

    The leading ``pytest`` (or absolute pytest path) is stripped; only
    the flags / paths / markers passed to it are returned.
    """
    with log_path.open() as f:
        # Header is on the first line, but tolerate a few lines of slack.
        for line in (next(f, "") for _ in range(5)):
            match = COMMAND_HEADER_RE.search(line)
            if match:
                tokens = shlex.split(match.group(1))
                if not tokens:
                    raise ValueError(f"Empty pytest command in {log_path}")
                # Drop the pytest binary itself; keep everything after.
                args = tokens[1:]
                # Strip verbosity/capture toggles; collect-only doesn't need them.
                return [a for a in args if a not in FLAGS_TO_STRIP]
    raise ValueError(f"Could not find a 'Pytest Run Command:' header in the first lines of {log_path}")


def collect_runnable_node_ids(pytest_args: list[str]) -> set[str]:
    """Return the set of nodeids that would actually run (skips excluded).

    Loads the inline plugin in ``scripts/_pytest_collect_runnable.py``,
    which runs after the project conftest's collection hooks (so
    ``uses_gpu`` / ``skip_ci`` / etc. have been translated into skip
    markers), evaluates skip markers, and prints ``RUNNABLE: <nodeid>``
    for every survivor before clearing ``items`` so nothing executes.
    """
    cmd = [
        "pytest",
        # Clear project addopts (which forces -v / coverage / etc.) so
        # output stays minimal and our RUNNABLE: lines are easy to grep.
        "-o",
        "addopts=",
        # Silence proto-tools' custom logging plugin during collection.
        "-p",
        "no:logging",
        "-p",
        "scripts._pytest_collect_runnable",
        *pytest_args,
    ]
    print(f"Collecting tests via: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 5):  # 5 == "no tests collected"; treat as empty
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"pytest collection exited {result.returncode}")
    return {
        line.removeprefix("RUNNABLE: ").strip() for line in result.stdout.splitlines() if line.startswith("RUNNABLE: ")
    }


def collect_finished_node_ids(log_path: Path) -> set[str]:
    """Return the set of node IDs with a PASSED/FAILED/ERROR line in the log."""
    finished: set[str] = set()
    with log_path.open() as f:
        for line in f:
            match = TEST_OUTCOME_RE.search(line)
            if match:
                finished.add(match.group(2))
    return finished


def main() -> int:
    """CLI entry point: parse args, dump incomplete node IDs to stdout or a file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        help="Path to the pytest log. Defaults to the latest logs/pytest_*.log.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write IDs to this file instead of stdout.",
    )
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    pytest_args = parse_pytest_args(log_path)
    runnable = collect_runnable_node_ids(pytest_args)
    finished = collect_finished_node_ids(log_path)
    incomplete = sorted(runnable - finished)

    print(
        f"{log_path}: runnable={len(runnable)} finished={len(finished)} incomplete={len(incomplete)}",
        file=sys.stderr,
    )

    payload = "\n".join(incomplete) + ("\n" if incomplete else "")
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
