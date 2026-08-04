"""tests/style_consistency_tests/test_packaging_excludes_helper_copies.py.

Guards the packaging rule that keeps runtime helper copies out of the wheel.

Older proto-tools versions copied ``standalone_helpers/`` into each tool's ``standalone/``
directory at runtime. Those copies are gitignored, but ``.gitignore`` does not constrain
setuptools, so building from a working tree that has ever run a tool would package them and
ship copies that then shadow the real package on ``sys.path[0]`` in one-shot dispatch.

``[tool.setuptools.packages.find] exclude`` prevents that. These tests run the same discovery
setuptools does, so the rule cannot be dropped without a failure here.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_namespace_packages

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # requires-python allows 3.10, where tomllib does not exist
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_SOURCE_PACKAGE = "proto_tools.utils.standalone_helpers_source.standalone_helpers"


def _find_config() -> dict:
    """Return the ``[tool.setuptools.packages.find]`` table from pyproject.toml."""
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["tool"]["setuptools"]["packages"]["find"]


def _discovered_packages() -> list[str]:
    """Run setuptools' package discovery exactly as configured."""
    cfg = _find_config()
    return find_namespace_packages(
        where=str(_REPO_ROOT),
        include=cfg.get("include", ("*",)),
        exclude=cfg.get("exclude", ()),
    )


def test_runtime_helper_copies_are_not_packaged() -> None:
    """No ``<tool>/standalone/standalone_helpers`` package may be discovered."""
    leaked = [p for p in _discovered_packages() if ".standalone.standalone_helpers" in f".{p}."]
    assert not leaked, (
        "runtime helper copies would be packaged into the wheel and shadow the real package:\n"
        + "\n".join(f"  {p}" for p in leaked)
    )


def test_authoritative_helper_package_is_still_packaged() -> None:
    """The exclusion must not catch the source package the exclusion exists to protect."""
    assert _SOURCE_PACKAGE in _discovered_packages()


def test_exclusion_is_declared() -> None:
    """A tree with no leftover copies would pass the first test vacuously; pin the rule itself."""
    exclude = _find_config().get("exclude", [])
    assert any("standalone.standalone_helpers" in pattern for pattern in exclude), (
        "packages.find must exclude *.standalone.standalone_helpers; without it a wheel built "
        "from a working tree that has run tools ships shadowing copies"
    )
