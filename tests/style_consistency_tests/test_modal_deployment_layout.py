"""tests/style_consistency_tests/test_modal_deployment_layout.py.

Layout checks for Modal service modules under proto_tools/modal.

Each service lives in its own ``{toolkit}_deployment/`` directory beside anything
it owns, such as ``standalone_overrides/``. Keeping every service in the same
shape is what makes one findable from its tool name alone.
"""

from pathlib import Path

import pytest

_MODAL_ROOT = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "modal"

_DEPLOYMENT_SUFFIX = "_deployment"
_SERVICE_SUFFIX = "_service.py"


def _service_modules() -> list[Path]:
    """Find every Modal service module."""
    return sorted(p for p in _MODAL_ROOT.rglob(f"*{_SERVICE_SUFFIX}") if "__pycache__" not in p.parts)


def _service_id(path: Path) -> str:
    """Return a path relative to proto_tools/modal for readable test ids."""
    return path.relative_to(_MODAL_ROOT).as_posix()


_SERVICES = _service_modules()


def test_services_are_discovered() -> None:
    """A glob that silently matches nothing would make every check below vacuous."""
    assert _SERVICES, f"no *{_SERVICE_SUFFIX} modules found under {_MODAL_ROOT}"


@pytest.mark.parametrize("service", _SERVICES, ids=_service_id)
def test_deployment_directory_matches_its_service(service: Path) -> None:
    """The directory name is the service's own stem, so the pair is greppable together."""
    expected = f"{service.name[: -len(_SERVICE_SUFFIX)]}{_DEPLOYMENT_SUFFIX}"
    assert service.parent.name == expected, (
        f"{_service_id(service)} is in '{service.parent.name}'; expected '{expected}'"
    )


def test_no_stray_standalone_overrides_directory() -> None:
    """Overrides belong beside the service that applies them, not in a shared directory."""
    strays = [
        d.relative_to(_MODAL_ROOT).as_posix()
        for d in _MODAL_ROOT.rglob("standalone_overrides")
        if d.is_dir() and not d.parent.name.endswith(_DEPLOYMENT_SUFFIX)
    ]
    assert not strays, f"standalone_overrides/ outside a *_deployment/ directory: {strays}"
