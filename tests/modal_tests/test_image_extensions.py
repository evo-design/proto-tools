"""Extra packages and source a deployment adds to every service image."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import pytest

from proto_tools.modal.base_images import (
    EXTRA_PACKAGES_ENV,
    EXTRA_SOURCE_ENV,
    EXTRA_SOURCE_IGNORE,
    _with_extras,
    extra_packages,
    extra_source_dirs,
)
from proto_tools.modal.hooks import PLUGINS_ENV
from tests.modal_tests.helpers import MODAL_ROOT, service_modules


@pytest.fixture(autouse=True)
def _clear_extension_env(monkeypatch: pytest.MonkeyPatch):
    """A developer with any of these exported would otherwise fail tests that assert exact layers."""
    for name in (EXTRA_PACKAGES_ENV, EXTRA_SOURCE_ENV, PLUGINS_ENV):
        monkeypatch.delenv(name, raising=False)


class _RecordingImage:
    """Records the layers ``_with_extras`` adds, standing in for a ``modal.Image``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def pip_install(self, *args: Any, **kwargs: Any) -> _RecordingImage:
        self.calls.append(("pip_install", args, kwargs))
        return self

    def add_local_dir(self, *args: Any, **kwargs: Any) -> _RecordingImage:
        self.calls.append(("add_local_dir", args, kwargs))
        return self


def test_no_extras_leaves_the_image_untouched() -> None:
    """Every deployment that asks for nothing must get the image it would have had."""
    image = _RecordingImage()
    assert _with_extras(image) is image
    assert image.calls == []


def test_packages_split_on_whitespace_not_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version range is comma-separated, so splitting on commas would tear one in half."""
    monkeypatch.setenv(EXTRA_PACKAGES_ENV, "alpha>=1,<2  beta==3.4")
    assert extra_packages() == ["alpha>=1,<2", "beta==3.4"]


def test_a_source_dir_is_mounted_under_its_own_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The mount name is what a container imports it as, so it must follow the directory."""
    package = tmp_path / "extras"
    package.mkdir()
    monkeypatch.setenv(EXTRA_SOURCE_ENV, str(package))

    image = _RecordingImage()
    _with_extras(image)
    assert image.calls == [
        ("add_local_dir", (str(package), "/root/extras"), {"copy": True, "ignore": EXTRA_SOURCE_IGNORE})
    ]


def test_a_mounted_dir_excludes_its_repository_metadata() -> None:
    """A checkout is the obvious thing to point this at, and git rewrites its index as you work.

    Mounting that would change the layer hash between two deploys of identical code, rebuilding
    every image below it.
    """
    assert ".git" in EXTRA_SOURCE_IGNORE
    assert "tests" not in EXTRA_SOURCE_IGNORE, "someone else's tests/ may be a package their code imports"


def test_several_source_dirs_are_path_separated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A path may contain spaces, so the list separator cannot be whitespace."""
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(EXTRA_SOURCE_ENV, os.pathsep.join([str(first), str(second)]))
    assert extra_source_dirs() == [first, second]


def test_a_home_relative_source_dir_is_expanded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Nothing expands ``~`` when the value arrives from a config file rather than a shell."""
    package = tmp_path / "extras"
    package.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(EXTRA_SOURCE_ENV, "~/extras")
    assert extra_source_dirs() == [package]


def test_a_missing_source_dir_fails_the_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Deferring this failure hides it until a container runs, long after the deploy reported success."""
    monkeypatch.setenv(EXTRA_SOURCE_ENV, str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="not a directory"):
        extra_source_dirs()


@pytest.mark.parametrize("name", ["my-lib", "2foo", ""])
def test_a_dir_nothing_could_import_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str) -> None:
    """It mounts by its own name, so a name that is not an identifier promises an import that fails."""
    target = tmp_path / name if name else Path(os.sep)
    if name:
        target.mkdir()
    monkeypatch.setenv(EXTRA_SOURCE_ENV, str(target))
    with pytest.raises(ValueError, match="identifier"):
        extra_source_dirs()


def test_every_service_image_is_built_through_with_proto_tools() -> None:
    """The image extension points live there, so a service bypassing it would ignore them."""
    missing = [str(path.relative_to(MODAL_ROOT)) for path in service_modules() if not _calls(path, "with_proto_tools")]
    assert service_modules(), "found no service modules to check — the scan itself is broken"
    assert not missing, f"service images not built through with_proto_tools: {missing}"


def test_every_service_image_applies_the_runtime_environment() -> None:
    """``RUNTIME_ENV`` carries the plugin list, and two services once omitted it and loaded none."""
    missing = [str(path.relative_to(MODAL_ROOT)) for path in service_modules() if "RUNTIME_ENV" not in path.read_text()]
    assert not missing, f"service images that never apply RUNTIME_ENV: {missing}"


def _calls(path: Path, name: str) -> bool:
    """Report whether ``path`` calls ``name``, rather than merely importing or mentioning it."""
    tree = ast.parse(path.read_text(), str(path))
    return any(
        isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name
        for node in ast.walk(tree)
    )
