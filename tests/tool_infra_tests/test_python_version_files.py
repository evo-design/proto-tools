"""tests/tool_infra_tests/test_python_version_files.py.

Tests for python_version.txt parsing and validation.
"""

from pathlib import Path

import pytest

from proto_tools.utils.tool_instance import ToolInstance

_SETUP_SH_CONTENT = "#!/bin/bash\necho test"


def _make_test_instance(setup_dir: Path) -> ToolInstance:
    """Create a ToolInstance with a fake setup script for testing."""
    inst = ToolInstance.__new__(ToolInstance)
    inst.toolkit = "test_tool"
    inst.setup_script = setup_dir / "setup.sh"
    return inst


# ── End-to-end via _get_python_version (file I/O + platform resolution) ─────


def test_valid_default_major_minor(tmp_path):
    """Keyed file with `default: X.Y` resolves to that version."""
    (tmp_path / "python_version.txt").write_text("default: 3.11\n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11"


def test_valid_default_major_minor_patch(tmp_path):
    """Keyed file with `default: X.Y.Z` resolves to that version."""
    (tmp_path / "python_version.txt").write_text("default: 3.11.5\n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11.5"


def test_whitespace_around_separator_is_stripped(tmp_path):
    """Whitespace around `:` and surrounding the value is stripped."""
    (tmp_path / "python_version.txt").write_text("default :  3.11  \n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11"


# ── Rejection cases (via _get_python_version) ──────────────────────────────


def test_empty_file_raises_error(tmp_path):
    """Empty python_version.txt raises clear error."""
    (tmp_path / "python_version.txt").write_text("")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"no entries after stripping"):
        tool._get_python_version()


def test_line_without_colon_raises_error(tmp_path):
    """A line that's not `key: value` (e.g. a stray bare version) raises."""
    (tmp_path / "python_version.txt").write_text("default: 3.11\n3.12\n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"Expected 'key: value' format"):
        tool._get_python_version()


@pytest.mark.parametrize(
    "invalid",
    [
        "3",  # Only major
        "python3.11",  # Text prefix
        "v3.11",  # Version prefix
        "3.11.5.2",  # Too many parts
        "3.x",  # Non-numeric
        "latest",  # Non-version string
    ],
)
def test_invalid_default_value_raises_error(tmp_path, invalid):
    """Invalid version values are rejected."""
    (tmp_path / "python_version.txt").write_text(f"default: {invalid}\n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"Invalid.*version"):
        tool._get_python_version()


@pytest.mark.parametrize("version", ["2.7", "3.6", "3.7"])
def test_unsupported_default_version_raises_error(tmp_path, version):
    """Python versions before 3.8 are rejected."""
    (tmp_path / "python_version.txt").write_text(f"default: {version}\n")
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    with pytest.raises(ValueError, match=r"(Unsupported|version.*3\.8)"):
        tool._get_python_version()


def test_comments_are_stripped(tmp_path):
    """`#` comments are ignored; the rest of the file parses normally."""
    content = "# Use Python 3.11 because of upstream constraints\ndefault: 3.11  # inline comment too\n"
    (tmp_path / "python_version.txt").write_text(content)
    (tmp_path / "setup.sh").write_text(_SETUP_SH_CONTENT)

    tool = _make_test_instance(tmp_path)
    assert tool._get_python_version() == "3.11"


# ── _parse_python_version (pure parser, with explicit platform_key) ────────


def test_keyed_form_default_only_resolves_for_any_platform():
    """A file with only `default:` returns that value for every platform key."""
    content = "default: 3.11\n"
    for platform_key in ("linux-x86_64", "linux-aarch64", "darwin-arm64", "darwin-x86_64"):
        assert ToolInstance._parse_python_version(content, platform_key, "<test>") == "3.11"


def test_keyed_form_specific_override_matches_platform():
    """An exact `{system}-{machine}` key wins over `default`."""
    content = "default: 3.11\nlinux-aarch64: 3.10\n"
    assert ToolInstance._parse_python_version(content, "linux-aarch64", "<test>") == "3.10"


def test_keyed_form_specific_override_does_not_cross_platform():
    """A `linux-aarch64` override does not affect `linux-x86_64`."""
    content = "default: 3.11\nlinux-aarch64: 3.10\n"
    assert ToolInstance._parse_python_version(content, "linux-x86_64", "<test>") == "3.11"


def test_keyed_form_os_fallback_matches():
    """An OS-only key (e.g. `linux:`) hits when no exact override exists."""
    content = "default: 3.11\nlinux: 3.10\n"
    assert ToolInstance._parse_python_version(content, "linux-x86_64", "<test>") == "3.10"
    assert ToolInstance._parse_python_version(content, "linux-aarch64", "<test>") == "3.10"


def test_keyed_form_specific_beats_os_fallback():
    """When both OS and specific keys are defined, the more specific one wins."""
    content = "default: 3.11\nlinux: 3.10\nlinux-aarch64: 3.9\n"
    assert ToolInstance._parse_python_version(content, "linux-aarch64", "<test>") == "3.9"
    # And the OS-only fallback still applies for siblings
    assert ToolInstance._parse_python_version(content, "linux-x86_64", "<test>") == "3.10"


def test_keyed_form_os_fallback_does_not_cross_os():
    """A `linux:` key does not match darwin platforms; they fall through to default."""
    content = "default: 3.11\nlinux: 3.10\n"
    assert ToolInstance._parse_python_version(content, "darwin-arm64", "<test>") == "3.11"
    assert ToolInstance._parse_python_version(content, "darwin-x86_64", "<test>") == "3.11"


def test_keyed_form_missing_default_raises():
    """Missing `default` key surfaces a clear error."""
    content = "linux-aarch64: 3.10\n"
    with pytest.raises(ValueError, match=r"missing required 'default' key"):
        ToolInstance._parse_python_version(content, "linux-aarch64", "<test>")


def test_keyed_form_duplicate_key_raises():
    """Duplicate keys (e.g. two `default:` lines) raise."""
    content = "default: 3.11\ndefault: 3.12\n"
    with pytest.raises(ValueError, match=r"Duplicate key 'default'"):
        ToolInstance._parse_python_version(content, "linux-x86_64", "<test>")


def test_keyed_form_comments_and_blank_lines_ignored():
    """Comments and blank lines surrounding the keyed entries are ignored."""
    content = (
        "# This file pins Python per platform.\n"
        "\n"
        "default: 3.11  # most platforms get latest\n"
        "\n"
        "# linux-aarch64 is stuck on the 2023.11 build\n"
        "linux-aarch64: 3.10\n"
        "\n"
    )
    assert ToolInstance._parse_python_version(content, "linux-x86_64", "<test>") == "3.11"
    assert ToolInstance._parse_python_version(content, "linux-aarch64", "<test>") == "3.10"


def test_keyed_form_invalid_override_value_raises_even_when_unmatched():
    """Validation happens up front for every value, regardless of current platform.

    A typo in `linux-aarch64: not-a-version` must fail in CI on x86_64 — not
    silently wait until someone runs on aarch64.
    """
    content = "default: 3.11\nlinux-aarch64: not-a-version\n"
    with pytest.raises(ValueError, match=r"Invalid Python version format"):
        ToolInstance._parse_python_version(content, "linux-x86_64", "<test>")


def test_keyed_form_unknown_key_silently_ignored_at_lookup():
    """Unknown keys (typos, platforms you're not on) parse fine; they just never match."""
    content = "default: 3.11\nlinux-fooarch: 3.10\n"
    # The override is still validated as a version, but it won't match real platforms.
    assert ToolInstance._parse_python_version(content, "linux-x86_64", "<test>") == "3.11"
    assert ToolInstance._parse_python_version(content, "darwin-arm64", "<test>") == "3.11"
