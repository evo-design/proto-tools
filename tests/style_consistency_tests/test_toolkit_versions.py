"""tests/style_consistency_tests/test_toolkit_versions.py.

Every packaged toolkit pins its current version to its source hash in ``versions.yaml``.

Any change to the hashed source needs a new top entry, so a version always names one
exact source. See ``notes/versioning.md``.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

from proto_tools.tools import ToolRegistry
from proto_tools.utils.tool_versions import HASH_LENGTH, VERSIONS_FILE, hashed_files, load_versions, source_hash

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # requires-python allows 3.10, where tomllib does not exist
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

_HASH = re.compile(rf"^[0-9a-f]{{{HASH_LENGTH}}}$")

# Mock tools that exist only to exercise the framework.
_EXEMPT_CATEGORIES = frozenset({"testing"})

_TOOLKIT_DIRS = sorted(
    {
        spec.toolkit_dir
        for spec in ToolRegistry.list_all()
        if not spec.is_external and spec.toolkit_dir.parent.name not in _EXEMPT_CATEGORIES
    }
)


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _semver_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


@pytest.mark.parametrize("toolkit_dir", _TOOLKIT_DIRS, ids=lambda d: d.name)
def test_versions_file_is_well_formed(toolkit_dir: Path) -> None:
    """Keys are X.Y.Z versions in strictly descending order, values are source hashes."""
    versions = load_versions(toolkit_dir)
    assert versions, f"{_rel(toolkit_dir)} has no {VERSIONS_FILE}; add one with `1.0.0: {source_hash(toolkit_dir)}`"

    bad_versions = [v for v in versions if not _SEMVER.match(v)]
    assert not bad_versions, f"{_rel(toolkit_dir / VERSIONS_FILE)}: not X.Y.Z versions: {bad_versions}"
    bad_hashes = [h for h in versions.values() if not _HASH.match(h)]
    assert not bad_hashes, f"{_rel(toolkit_dir / VERSIONS_FILE)}: not {HASH_LENGTH}-hex hashes: {bad_hashes}"

    keys = [_semver_key(v) for v in versions]
    assert keys == sorted(keys, reverse=True) and len(set(keys)) == len(keys), (
        f"{_rel(toolkit_dir / VERSIONS_FILE)}: versions must be unique and newest first: {list(versions)}"
    )


@pytest.mark.parametrize("toolkit_dir", _TOOLKIT_DIRS, ids=lambda d: d.name)
def test_current_version_matches_source(toolkit_dir: Path) -> None:
    """The newest entry's hash is the toolkit's current source hash."""
    versions = load_versions(toolkit_dir)
    if not versions:
        pytest.skip(f"no {VERSIONS_FILE}; reported by test_versions_file_is_well_formed")
    version, recorded = next(iter(versions.items()))
    live = source_hash(toolkit_dir)
    major, minor, patch = _semver_key(version)
    assert live == recorded, (
        f"{toolkit_dir.name} source changed: {VERSIONS_FILE} has {version} → {recorded}, current hash is {live}\n"
        f"Add a new top entry to {_rel(toolkit_dir / VERSIONS_FILE)}:\n"
        f"  {major}.{minor}.{patch + 1}: {live}   (patch: no output change)\n"
        f"  {major}.{minor + 1}.0: {live}   (minor: outputs may differ)\n"
        f"  {major + 1}.0.0: {live}   (major: schema break)"
    )


def test_hashed_files_ship_in_the_wheel() -> None:
    """A hashed file missing from the wheel would give installed copies a different hash."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        patterns = tomllib.load(fh)["tool"]["setuptools"]["package-data"]["*"]
    unshipped = sorted(
        _rel(path)
        for toolkit_dir in _TOOLKIT_DIRS
        for path in hashed_files(toolkit_dir).values()
        if path.suffix != ".py" and not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
    )
    assert not unshipped, (
        f"Hashed files not matched by [tool.setuptools.package-data] in pyproject.toml: {unshipped}. "
        "Add a pattern for them so wheel installs hash the same source."
    )
