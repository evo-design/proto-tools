"""tests/style_consistency_tests/test_toolkit_versions.py.

Every packaged toolkit pins its current version to its source hash in ``versions.yaml``.

Any change to the hashed source needs a new top entry, so a version always names one
exact source. See ``notes/versioning.md``.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

import pytest

from proto_tools.utils.tool_versions import (
    HASH_LENGTH,
    VERSIONS_FILE,
    hashed_files,
    load_versions,
    next_version,
    packaged_toolkit_dirs,
    source_hash,
)

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # requires-python allows 3.10, where tomllib does not exist
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

_HASH = re.compile(rf"^[0-9a-f]{{{HASH_LENGTH}}}$")

_TOOLKIT_DIRS = packaged_toolkit_dirs()


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
    assert live == recorded, (
        f"{toolkit_dir.name} source changed: {VERSIONS_FILE} has {version} → {recorded}, current hash is {live}\n"
        f"Add a new top entry to {_rel(toolkit_dir / VERSIONS_FILE)}:\n"
        f"  {next_version(version, 'patch')}: {live}   (patch: no output change)\n"
        f"  {next_version(version, 'minor')}: {live}   (minor: outputs may differ)\n"
        f"  {next_version(version, 'major')}: {live}   (major: schema break)\n"
        f"or run: python scripts/update_versions.py --bump patch {toolkit_dir.name}"
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


def _is_reexport(node: ast.stmt) -> bool:
    if isinstance(node, ast.Import | ast.ImportFrom):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return True  # docstring
    return isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["__all__"]


def test_unhashed_init_files_only_reexport() -> None:
    """Toolkit and category ``__init__.py`` files are not hashed, so they must hold no logic."""
    inits = {toolkit_dir / "__init__.py" for toolkit_dir in _TOOLKIT_DIRS}
    inits |= {toolkit_dir.parent / "__init__.py" for toolkit_dir in _TOOLKIT_DIRS}
    offenders = sorted(
        _rel(path)
        for path in inits
        if path.is_file() and not all(_is_reexport(node) for node in ast.parse(path.read_text()).body)
    )
    assert not offenders, (
        f"These __init__.py files are left out of toolkit version hashes but contain logic: {offenders}. "
        "Move the code into a module of the toolkit or category."
    )
