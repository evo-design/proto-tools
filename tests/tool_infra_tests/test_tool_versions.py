"""tests/tool_infra_tests/test_tool_versions.py.

Behaviour of the toolkit source hash and the version stamped from it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from proto_tools.utils import tool_versions
from proto_tools.utils.tool_versions import load_versions, source_hash, toolkit_version

_TOOL_SOURCE = 'def run(x):\n    return x + 1  # add one\n\n\nLABEL = "# not a comment"\n'


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    yield
    for cached in (source_hash, toolkit_version, tool_versions._repo_root, tool_versions._git_files):
        cached.cache_clear()


@pytest.fixture
def toolkit(tmp_path: Path) -> Path:
    """A minimal toolkit outside any git checkout."""
    root = tmp_path / "fake_toolkit"
    (root / "standalone").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "fake_toolkit_run.py").write_text(_TOOL_SOURCE)
    (root / "standalone" / "requirements.txt").write_text("numpy==2.0.0\n")
    (root / "README.md").write_text("# Fake\n")
    return root


def _rehash(root: Path) -> str:
    source_hash.cache_clear()
    return source_hash(root)


def test_comments_blank_lines_and_line_endings_do_not_move_the_hash(toolkit: Path) -> None:
    before = _rehash(toolkit)
    edited = _TOOL_SOURCE.replace("  # add one", "  # increment").replace("\n\n\n", "\n\n# note\n\n")
    (toolkit / "fake_toolkit_run.py").write_bytes(edited.replace("\n", "\r\n").encode())

    assert _rehash(toolkit) == before


def test_a_hash_inside_a_string_is_code(toolkit: Path) -> None:
    before = _rehash(toolkit)
    (toolkit / "fake_toolkit_run.py").write_text(_TOOL_SOURCE.replace('"# not a comment"', '"# changed"'))

    assert _rehash(toolkit) != before


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("fake_toolkit_run.py", _TOOL_SOURCE.replace("x + 1", "x + 2")),
        ("standalone/requirements.txt", "numpy==2.1.0\n"),
    ],
)
def test_code_and_environment_edits_move_the_hash(toolkit: Path, relative: str, content: str) -> None:
    before = _rehash(toolkit)
    (toolkit / relative).write_text(content)

    assert _rehash(toolkit) != before


@pytest.mark.parametrize(
    "relative",
    ["README.md", "cite.bib", "links.yaml", "license.yaml", "versions.yaml", "examples/out.json", ".gitignore"],
)
def test_docs_examples_and_metadata_do_not_move_the_hash(toolkit: Path, relative: str) -> None:
    before = _rehash(toolkit)
    (toolkit / relative).write_text("edited\n")

    assert _rehash(toolkit) == before


def test_hash_does_not_depend_on_location(toolkit: Path, tmp_path: Path) -> None:
    moved = shutil.copytree(toolkit, tmp_path / "elsewhere" / "fake_toolkit")

    assert _rehash(moved) == _rehash(toolkit)


def test_git_ignored_artifacts_do_not_move_the_hash(toolkit: Path) -> None:
    subprocess.run(["git", "init", "-q", str(toolkit.parent)], check=True)
    (toolkit.parent / ".gitignore").write_text("*.db\n")
    before = _rehash(toolkit)
    (toolkit / "standalone" / "index.db").write_text("built at runtime\n")
    tool_versions._git_files.cache_clear()

    assert _rehash(toolkit) == before


def test_all_digit_hash_stays_a_string(toolkit: Path) -> None:
    (toolkit / "versions.yaml").write_text("1.0.0: 000000000012\n")

    assert load_versions(toolkit) == {"1.0.0": "000000000012"}


def test_version_is_the_newest_entry(toolkit: Path) -> None:
    (toolkit / "versions.yaml").write_text(f"1.1.0: {source_hash(toolkit)}\n1.0.0: 4b7e02a91c3f\n")

    assert toolkit_version(toolkit) == "1.1.0"


def test_unversioned_source_is_marked_dirty(toolkit: Path, caplog: pytest.LogCaptureFixture) -> None:
    (toolkit / "versions.yaml").write_text("1.0.0: 4b7e02a91c3f\n")

    assert toolkit_version(toolkit) == "1.0.0+dirty"
    assert "does not match versions.yaml" in caplog.text


def test_toolkit_without_versions_file_has_no_version(toolkit: Path) -> None:
    assert toolkit_version(toolkit) is None
