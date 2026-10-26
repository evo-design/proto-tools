"""Consistency check for ``cuda-version`` pinning in standalone setup scripts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PROTO_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools"


def _discover_setup_scripts() -> list[Path]:
    """Find every ``setup.sh`` under a toolkit's ``standalone/`` or a shared env."""
    scripts = list((_PROTO_TOOLS_DIR / "tools").rglob("standalone/setup.sh"))
    scripts.extend((_PROTO_TOOLS_DIR / "shared_envs").rglob("setup.sh"))
    return sorted(scripts)


_ALL_SETUP_SCRIPTS = _discover_setup_scripts()

_ID = lambda p: p.parent.parent.name if p.parent.name == "standalone" else p.parent.name  # noqa: E731


@pytest.mark.parametrize("setup_script", _ALL_SETUP_SCRIPTS, ids=[_ID(p) for p in _ALL_SETUP_SCRIPTS])
def test_cuda_toolkit_install_pins_cuda_version(setup_script: Path) -> None:
    """A conda command installing ``cuda-toolkit=`` must also pin ``cuda-version=``."""
    # Join backslash-continued lines so a multi-line command reads as one.
    for line in re.sub(r"\\\n\s*", " ", setup_script.read_text()).splitlines():
        if line.lstrip().startswith("#") or "cuda-toolkit=" not in line:
            continue
        assert "cuda-version=" in line, (
            f"{setup_script.relative_to(_PROTO_TOOLS_DIR)}: installs cuda-toolkit without pinning "
            "cuda-version, so sub-packages can resolve to a newer CUDA generation than the "
            'toolkit. Add "cuda-version=<same constraint>" to the same command.'
        )
