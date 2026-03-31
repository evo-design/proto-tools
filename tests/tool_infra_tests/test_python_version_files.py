"""tests/tool_infra_tests/test_python_version_files.py

Tests for python_version.txt validation."""

import sys
from pathlib import Path

import pytest

from proto_tools.utils.tool_instance import ToolInstance

_SETUP_SH_CONTENT = "#!/bin/bash\necho test"


def _make_test_instance(setup_dir: Path) -> ToolInstance:
    """Create a ToolInstance with a fake setup script for testing."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.tool_name = "test_tool"
    inst.setup_script = setup_dir / "setup.sh"
    return inst


# ── Format validation ────────────────────────────────────────────────────────


def test_valid_major_minor_format(tmp_path):
    """major.minor format is accepted."""
    (tmp_path / "python_version.txt").write_text("3.11")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11"


def test_valid_major_minor_patch_format(tmp_path):
    """major.minor.patch format is accepted."""
    (tmp_path / "python_version.txt").write_text("3.11.5")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11.5"


def test_missing_file_defaults_to_current_python(tmp_path):
    """Missing python_version.txt defaults to current Python version."""
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert tool._get_python_version() == expected


def test_whitespace_is_stripped(tmp_path):
    """Leading/trailing whitespace is handled gracefully."""
    (tmp_path / "python_version.txt").write_text("  3.11  \n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11"


# ── Rejection cases ──────────────────────────────────────────────────────────


def test_empty_file_raises_error(tmp_path):
    """Empty python_version.txt raises clear error."""
    (tmp_path / "python_version.txt").write_text("")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"python_version\.txt.*is empty"):
        tool._get_python_version()


def test_multiple_lines_raises_error(tmp_path):
    """Multiple lines in python_version.txt raises error."""
    (tmp_path / "python_version.txt").write_text("3.11\n3.12\n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"Invalid.*version"):
        tool._get_python_version()


@pytest.mark.parametrize("invalid", [
    "3",           # Only major
    "python3.11",  # Text prefix
    "v3.11",       # Version prefix
    "3.11.5.2",    # Too many parts
    "3.x",         # Non-numeric
    "latest",      # Non-version string
])
def test_invalid_format_raises_error(tmp_path, invalid):
    """Invalid version formats are rejected."""
    (tmp_path / "python_version.txt").write_text(invalid)
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"Invalid.*version"):
        tool._get_python_version()


@pytest.mark.parametrize("version", ["2.7", "3.6", "3.7"])
def test_unsupported_version_raises_error(tmp_path, version):
    """Python versions before 3.8 are rejected."""
    (tmp_path / "python_version.txt").write_text(version)
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"(Unsupported|version.*3\.8)"):
        tool._get_python_version()


def test_comments_not_allowed(tmp_path):
    """Comments in python_version.txt raise error."""
    (tmp_path / "python_version.txt").write_text("3.11  # Use Python 3.11")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"Invalid.*version"):
        tool._get_python_version()


# ── Codebase-wide validation ─────────────────────────────────────────────────


def test_all_existing_python_version_files_are_valid():
    """Discover and validate all python_version.txt files in tools directory."""
    tools_dir = Path(__file__).parent.parent.parent / "proto_tools" / "tools"

    version_files = list(tools_dir.glob("**/standalone/python_version.txt"))

    if not version_files:
        pytest.skip("No python_version.txt files found yet")

    errors = []
    for version_file in version_files:
        tool_name = version_file.parent.parent.parent.name
        setup_script = version_file.parent / "setup.sh"

        try:
            tool = ToolInstance.__new__(ToolInstance)
            tool.tool_name = tool_name
            tool.setup_script = setup_script

            version = tool._get_python_version()

            parts = version.split(".")
            assert len(parts) in (2, 3), f"{tool_name}: version must be major.minor or major.minor.patch"
            assert all(part.isdigit() for part in parts), f"{tool_name}: version parts must be numeric"

            major, minor = int(parts[0]), int(parts[1])
            assert (major, minor) >= (3, 8), f"{tool_name}: minimum Python 3.8 required"

        except Exception as e:
            errors.append(f"{tool_name} ({version_file}): {e}")

    if errors:
        pytest.fail("Invalid python_version.txt files found:\n" + "\n".join(errors))
