"""Tests for ToolInstance shared-env resolution.

Two or more tools can share a single micromamba environment by placing a
``shared_env.txt`` marker in their ``standalone/`` directory pointing to a
named env definition under ``proto_tools/shared_envs/``. These tests cover the
resolution, validation, and disk-sharing behavior in isolation from any real
tool environment.
"""

from pathlib import Path

import pytest

from proto_tools.utils.tool_instance import ToolInstance


def _make_tool(tools_root: Path, name: str, *, with_setup: bool = False, shared: str | None = None) -> Path:
    """Create a fake tool directory with a standalone/ layout for testing."""
    tool_dir = tools_root / name
    standalone = tool_dir / "standalone"
    standalone.mkdir(parents=True)
    (standalone / "inference.py").write_text("# stub\n")
    if with_setup:
        (standalone / "setup.sh").write_text("#!/bin/bash\necho local\n")
        (standalone / "requirements.txt").write_text("tqdm\n")
    if shared is not None:
        (standalone / "shared_env.txt").write_text(shared)
    return tool_dir


def _make_shared_env(envs_root: Path, name: str, *, requirements: str = "tqdm\n") -> Path:
    """Create a fake shared-env definition."""
    env_def = envs_root / name
    env_def.mkdir(parents=True)
    (env_def / "setup.sh").write_text("#!/bin/bash\necho shared\n")
    (env_def / "requirements.txt").write_text(requirements)
    return env_def


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    """Patch ToolInstance class methods so resolution targets a tmp_path layout."""
    tools_root = tmp_path / "tools"
    envs_root = tmp_path / "shared_envs"
    tools_root.mkdir()
    envs_root.mkdir()

    tool_dirs: dict[str, Path] = {}

    monkeypatch.setattr(ToolInstance, "_get_tool_dirs", classmethod(lambda cls: tool_dirs))
    monkeypatch.setattr(ToolInstance, "_shared_envs_root", classmethod(lambda cls: envs_root))

    return {"tools_root": tools_root, "envs_root": envs_root, "tool_dirs": tool_dirs}


# ── Default (no shared_env.txt) ───────────────────────────────────────────────


def test_resolve_env_def_default_uses_tool_standalone(fake_tools):
    tool = _make_tool(fake_tools["tools_root"], "alpha", with_setup=True)
    fake_tools["tool_dirs"]["alpha"] = tool

    env_def, env_name = ToolInstance._resolve_env_def("alpha")

    assert env_def == tool / "standalone"
    assert env_name == "alpha"


# ── Shared env resolution ─────────────────────────────────────────────────────


def test_resolve_env_def_shared_uses_shared_dir(fake_tools):
    shared = _make_shared_env(fake_tools["envs_root"], "myfamily")
    tool = _make_tool(fake_tools["tools_root"], "consumer", shared="myfamily")
    fake_tools["tool_dirs"]["consumer"] = tool

    env_def, env_name = ToolInstance._resolve_env_def("consumer")

    assert env_def == shared
    assert env_name == "myfamily"


def test_two_tools_resolve_to_same_env_name(fake_tools):
    """Both consumers of the same shared env produce the same env_name → same env_path on disk."""
    _make_shared_env(fake_tools["envs_root"], "myfamily")
    fake_tools["tool_dirs"]["alpha"] = _make_tool(fake_tools["tools_root"], "alpha", shared="myfamily")
    fake_tools["tool_dirs"]["beta"] = _make_tool(fake_tools["tools_root"], "beta", shared="myfamily")

    _, name_a = ToolInstance._resolve_env_def("alpha")
    _, name_b = ToolInstance._resolve_env_def("beta")

    assert name_a == name_b == "myfamily"


def test_shared_env_marker_is_trimmed(fake_tools):
    """Trailing whitespace/newlines in shared_env.txt should not break resolution."""
    _make_shared_env(fake_tools["envs_root"], "myfamily")
    tool = _make_tool(fake_tools["tools_root"], "consumer")
    (tool / "standalone" / "shared_env.txt").write_text("  myfamily  \n")
    fake_tools["tool_dirs"]["consumer"] = tool

    _, env_name = ToolInstance._resolve_env_def("consumer")
    assert env_name == "myfamily"


# ── Validation errors ─────────────────────────────────────────────────────────


def test_unknown_tool_raises(fake_tools):
    with pytest.raises(ValueError, match="Unknown tool"):
        ToolInstance._resolve_env_def("nonexistent")


def test_shared_env_marker_pointing_to_missing_def_raises(fake_tools):
    tool = _make_tool(fake_tools["tools_root"], "consumer", shared="nonexistent")
    fake_tools["tool_dirs"]["consumer"] = tool

    with pytest.raises(ValueError, match="references shared env 'nonexistent'"):
        ToolInstance._resolve_env_def("consumer")


def test_shared_env_marker_with_no_setup_in_def_raises(fake_tools):
    """If the shared env dir exists but has no setup.sh, raise clearly."""
    (fake_tools["envs_root"] / "broken").mkdir()  # exists but empty
    tool = _make_tool(fake_tools["tools_root"], "consumer", shared="broken")
    fake_tools["tool_dirs"]["consumer"] = tool

    with pytest.raises(ValueError, match="does not exist"):
        ToolInstance._resolve_env_def("consumer")


def test_empty_shared_env_marker_raises(fake_tools):
    tool = _make_tool(fake_tools["tools_root"], "consumer")
    (tool / "standalone" / "shared_env.txt").write_text("")
    fake_tools["tool_dirs"]["consumer"] = tool

    with pytest.raises(ValueError, match=r"shared_env\.txt for tool 'consumer' is empty"):
        ToolInstance._resolve_env_def("consumer")


def test_conflict_local_setup_and_shared_env_marker_raises(fake_tools):
    """A tool dir cannot have BOTH shared_env.txt and a local setup.sh."""
    _make_shared_env(fake_tools["envs_root"], "myfamily")
    tool = _make_tool(fake_tools["tools_root"], "consumer", with_setup=True, shared="myfamily")
    fake_tools["tool_dirs"]["consumer"] = tool

    with pytest.raises(ValueError, match=r"shared_env\.txt alongside stray env-def file\(s\).*setup\.sh"):
        ToolInstance._resolve_env_def("consumer")


def test_conflict_local_env_def_file_and_shared_marker_raises(fake_tools):
    """A stray env-def file (not just setup.sh) alongside shared_env.txt also raises.

    Catches the migration slip where someone drops a marker but forgets to
    delete the per-tool ``requirements.txt`` / ``python_version.txt`` / ``env_vars.txt``.
    """
    _make_shared_env(fake_tools["envs_root"], "myfamily")
    tool = _make_tool(fake_tools["tools_root"], "consumer", shared="myfamily")
    # Leave behind a stray python_version.txt — no setup.sh, no requirements.txt.
    (tool / "standalone" / "python_version.txt").write_text("default: 3.11\n")
    fake_tools["tool_dirs"]["consumer"] = tool

    with pytest.raises(ValueError, match=r"shared_env\.txt alongside stray env-def file\(s\).*python_version\.txt"):
        ToolInstance._resolve_env_def("consumer")


def test_no_setup_and_no_shared_marker_raises(fake_tools):
    tool = _make_tool(fake_tools["tools_root"], "consumer")  # no setup, no marker
    fake_tools["tool_dirs"]["consumer"] = tool

    with pytest.raises(ValueError, match=r"No setup\.sh found"):
        ToolInstance._resolve_env_def("consumer")
