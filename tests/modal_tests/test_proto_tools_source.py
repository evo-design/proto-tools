"""proto-tools must resolve identically however it was installed.

Deploying from a plain install, an editable install, and a clone have to produce
the same image. The failure that matters here is silent: a wheel missing service
packages looks fine until someone tries to deploy.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile

import pytest

from tests.modal_tests.helpers import REPO


def test_explicit_path_override_rejects_non_checkout(tmp_path, monkeypatch):
    """A bad override must fail loudly rather than silently falling back."""
    from proto_tools.modal.proto_tools_source import PATH_ENV, ProtoToolsUnavailableError, resolve_proto_tools

    monkeypatch.setenv(PATH_ENV, str(tmp_path))
    with pytest.raises(ProtoToolsUnavailableError, match="holds no proto_tools package"):
        resolve_proto_tools()


@pytest.fixture(scope="module")
def wheel_contents(tmp_path_factory):
    """Build the wheel once and return the names it contains.

    Build isolation stays on. CI installs with uv, which leaves no setuptools
    in the environment, so ``--no-build-isolation`` fails there while passing
    anywhere setuptools happens to be installed.
    """
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(out), str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # pip's own diagnosis, rather than a bare non-zero exit code.
        pytest.fail(f"wheel build failed ({result.returncode}):\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return set(zipfile.ZipFile(next(out.glob("*.whl"))).namelist())


def test_wheel_ships_every_service_module(wheel_contents):
    """Namespace packages are invisible to default discovery; without them a wheel deploys nothing."""
    from proto_tools.modal.manifest import SERVICE_TO_MODULE

    missing = sorted(
        module for module in set(SERVICE_TO_MODULE.values()) if f"{module.replace('.', '/')}.py" not in wheel_contents
    )
    assert not missing, f"wheel is missing service modules: {missing}"


def test_wheel_ships_standalone_overrides(wheel_contents):
    """An override missing from the wheel builds an image that fails only at warmup."""
    expected = {str(p.relative_to(REPO)) for p in (REPO / "deployment" / "standalone_overrides").rglob("*.sh")}
    missing = sorted(expected - wheel_contents)
    assert not missing, f"wheel is missing override scripts: {missing}"


def test_override_may_be_a_bare_package(tmp_path, monkeypatch):
    """A container carries the package without a pyproject, so mounting must not demand one."""
    from proto_tools.modal.proto_tools_source import PATH_ENV, checkout_or_none, resolve_proto_tools

    (tmp_path / "proto_tools").mkdir()
    monkeypatch.setenv(PATH_ENV, str(tmp_path))

    assert resolve_proto_tools() == tmp_path
    assert checkout_or_none() is None, "a bare package is not a checkout and cannot supply dependencies"


def test_checkout_requires_a_pyproject(tmp_path, monkeypatch):
    """Only a checkout can supply dependencies, which is what selects the image recipe."""
    from proto_tools.modal.proto_tools_source import PATH_ENV, checkout_or_none

    (tmp_path / "proto_tools").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'proto-tools'\n")
    monkeypatch.setenv(PATH_ENV, str(tmp_path))

    assert checkout_or_none() == tmp_path
