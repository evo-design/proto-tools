"""tests/style_consistency_tests/test_module_docstrings.py

Tests that every .py file has a module docstring starting with its relative path.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from .helpers import collect_py_files

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCE_DIRS = ["proto_tools", "tests"]

_EXCLUDE_PATTERNS = [
    # Auto-copied at runtime; path header would be wrong after copy
    "proto_tools/tools/*/standalone/*",
    # Gitignored runtime directories
    "tool_envs/*",
]


def _get_py_files() -> list[tuple[Path, str]]:
    return collect_py_files(_REPO_ROOT, _SOURCE_DIRS, _EXCLUDE_PATTERNS)


_PY_FILES = _get_py_files()


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "py_file, rel_path",
    _PY_FILES,
    ids=[rel for _, rel in _PY_FILES],
)
def test_module_docstring_exists(py_file: Path, rel_path: str):
    """Every .py file must have a module-level docstring."""
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        pytest.skip(f"Could not parse {rel_path}")

    docstring = ast.get_docstring(tree)
    assert docstring is not None and docstring.strip(), (
        f"{rel_path} is missing a module-level docstring."
    )


@pytest.mark.parametrize(
    "py_file, rel_path",
    _PY_FILES,
    ids=[rel for _, rel in _PY_FILES],
)
def test_module_docstring_path_header(py_file: Path, rel_path: str):
    """The first line of the module docstring must be the relative path from repo root."""
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        pytest.skip(f"Could not parse {rel_path}")

    docstring = ast.get_docstring(tree)
    if not docstring:
        pytest.skip(f"{rel_path} has no docstring (caught by test_module_docstring_exists)")

    first_line = docstring.strip().split("\n")[0].strip()
    assert first_line == rel_path, (
        f"{rel_path}: module docstring first line should be '{rel_path}', "
        f"got '{first_line}'"
    )


@pytest.mark.parametrize(
    "py_file, rel_path",
    _PY_FILES,
    ids=[rel for _, rel in _PY_FILES],
)
def test_module_docstring_format(py_file: Path, rel_path: str):
    """If the module docstring is multi-line, the second line must be blank."""
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        pytest.skip(f"Could not parse {rel_path}")

    docstring = ast.get_docstring(tree)
    if not docstring:
        pytest.skip(f"{rel_path} has no docstring")

    lines = docstring.split("\n")
    if len(lines) <= 1:
        # Single-line docstring (just the path) — acceptable
        return

    assert lines[1].strip() == "", (
        f"{rel_path}: multi-line module docstring must have a blank line after "
        f"the path header. Got: '{lines[1].strip()}'"
    )
