"""Consistency check for binary installs in standalone setup scripts.

``proto_install_binary`` (from ``standalone_helpers.sh``) loads ``binary_config.py`` from the
directory ``setup.sh`` runs in, so the install works from a shared env, an ejected override, and an
out-of-tree toolkit alike. Every env definition that ships a ``binary_config.py`` must install
through it rather than locating ``utils/install_binary.py`` from the script's path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools"

_BINARY_ENV_DIRS = sorted(p.parent for p in _PACKAGE_DIR.rglob("binary_config.py"))


@pytest.mark.parametrize("env_dir", _BINARY_ENV_DIRS, ids=lambda p: p.relative_to(_PACKAGE_DIR).as_posix())
def test_binary_envs_install_through_the_helper(env_dir: Path):
    """``setup.sh`` sources the shell helpers and calls ``proto_install_binary``."""
    setup = (env_dir / "setup.sh").read_text()
    assert "source standalone_helpers.sh" in setup
    assert "proto_install_binary " in setup
    assert "utils/install_binary.py" not in setup
