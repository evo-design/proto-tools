"""tests/tool_infra_tests/test_uv_version_files.py.

Tests for the optional standalone/uv_version.txt override.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proto_tools.utils.tool_instance import UV_VERSION, ToolInstance


def _make_test_instance(setup_dir: Path) -> ToolInstance:
    """Create a ToolInstance with a fake setup script for testing."""
    (setup_dir / "setup.sh").write_text("#!/bin/bash\necho test")
    inst = ToolInstance.__new__(ToolInstance)
    inst.toolkit = "test_tool"
    inst.setup_script = setup_dir / "setup.sh"
    return inst


def test_missing_file_falls_back_to_the_global_pin(tmp_path):
    """Most tools ship no override and get UV_VERSION."""
    assert _make_test_instance(tmp_path)._get_uv_version() == UV_VERSION


def test_override_replaces_the_global_pin(tmp_path):
    """A tool that breaks on UV_VERSION pins its own."""
    (tmp_path / "uv_version.txt").write_text("default: 0.11.33\n")
    assert _make_test_instance(tmp_path)._get_uv_version() == "0.11.33"


@pytest.mark.parametrize("invalid", ["0.12", "latest", "0.12.x", ">=0.12.0"])
def test_invalid_version_is_rejected(tmp_path, invalid):
    """Only an exact major.minor.patch release can be installed by micromamba as a pin."""
    (tmp_path / "uv_version.txt").write_text(f"default: {invalid}\n")
    with pytest.raises(ValueError, match="Invalid uv version format"):
        _make_test_instance(tmp_path)._get_uv_version()


def test_missing_default_names_the_uv_file(tmp_path):
    """Errors from the shared parser name the file that is actually wrong."""
    (tmp_path / "uv_version.txt").write_text("linux: 0.11.33\n")
    with pytest.raises(ValueError, match=r"uv_version\.txt .* missing required 'default' key"):
        _make_test_instance(tmp_path)._get_uv_version()


def test_override_changes_the_setup_hash(tmp_path):
    """Adding or editing the override rebuilds that tool's env."""
    inst = _make_test_instance(tmp_path)
    (tmp_path / "python_version.txt").write_text("default: 3.12\n")
    without = inst._setup_hash()
    (tmp_path / "uv_version.txt").write_text("default: 0.11.33\n")
    pinned = inst._setup_hash()
    (tmp_path / "uv_version.txt").write_text("default: 0.11.32\n")
    assert len({without, pinned, inst._setup_hash()}) == 3
