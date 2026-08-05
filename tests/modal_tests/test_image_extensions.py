"""Extra packages and source a deployment adds to every service image."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from proto_tools.modal.base_images import (
    EXTRA_PACKAGES_ENV,
    EXTRA_SOURCE_ENV,
    _with_extras,
    extra_packages,
    extra_source_dirs,
)


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

    def env(self, *args: Any, **kwargs: Any) -> _RecordingImage:
        self.calls.append(("env", args, kwargs))
        return self


def test_no_extras_leaves_the_image_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every deployment that asks for nothing must get the image it would have had."""
    monkeypatch.delenv(EXTRA_PACKAGES_ENV, raising=False)
    monkeypatch.delenv(EXTRA_SOURCE_ENV, raising=False)

    image = _RecordingImage()
    assert _with_extras(image) is image
    assert image.calls == []


def test_packages_split_on_whitespace_not_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    """A version range is comma-separated, so splitting on commas would tear one in half."""
    monkeypatch.setenv(EXTRA_PACKAGES_ENV, "alpha>=1,<2  beta==3.4")
    assert extra_packages() == ["alpha>=1,<2", "beta==3.4"]


def test_a_source_dir_is_mounted_under_its_own_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The mount name is what a container imports it as, so it must follow the directory."""
    package = tmp_path / "observability"
    package.mkdir()
    monkeypatch.delenv(EXTRA_PACKAGES_ENV, raising=False)
    monkeypatch.setenv(EXTRA_SOURCE_ENV, str(package))

    image = _RecordingImage()
    _with_extras(image)
    assert image.calls == [("add_local_dir", (str(package), "/root/observability"), {"copy": True})]


def test_several_source_dirs_are_path_separated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A path may contain spaces, so the list separator cannot be whitespace."""
    first, second = tmp_path / "one", tmp_path / "two dirs"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv(EXTRA_SOURCE_ENV, os.pathsep.join([str(first), str(second)]))
    assert extra_source_dirs() == [first, second]


def test_a_missing_source_dir_fails_the_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Deferring this failure hides it until a container runs, long after the deploy reported success."""
    monkeypatch.setenv(EXTRA_SOURCE_ENV, str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="not a directory"):
        extra_source_dirs()


def test_the_plugin_list_is_baked_into_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """A container reads the variable from its own environment, not the one that deployed it."""
    from proto_tools.modal.hooks import PLUGINS_ENV

    monkeypatch.delenv(EXTRA_PACKAGES_ENV, raising=False)
    monkeypatch.delenv(EXTRA_SOURCE_ENV, raising=False)
    monkeypatch.setenv(PLUGINS_ENV, "observability.hooks")

    image = _RecordingImage()
    _with_extras(image)
    assert image.calls == [("env", ({PLUGINS_ENV: "observability.hooks"},), {})]


def test_every_service_image_is_built_through_with_proto_tools() -> None:
    """The extension points live there, so a service bypassing it would silently ignore all three."""
    import pathlib

    modal_root = pathlib.Path(__file__).resolve().parents[2] / "proto_tools" / "modal"
    services = sorted(p for p in modal_root.rglob("*_service.py") if "__pycache__" not in p.parts)
    missing = [str(p.relative_to(modal_root)) for p in services if "with_proto_tools" not in p.read_text()]

    assert services, "found no service modules to check — the scan itself is broken"
    assert not missing, f"service images not built through with_proto_tools: {missing}"
