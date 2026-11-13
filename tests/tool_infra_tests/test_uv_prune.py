"""tests/tool_infra_tests/test_uv_prune.py.

The background uv cache prune a successful env build starts.
"""

from __future__ import annotations

import fcntl
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from proto_tools.utils import uv_prune
from proto_tools.utils.tool_instance import UV_VERSION, ToolInstance


@pytest.fixture
def fake_uv(tmp_path: Path) -> Path:
    """A uv stand-in that records each call's arguments and UV_CACHE_DIR."""
    uv = tmp_path / "uv"
    uv.write_text(f'#!/bin/sh\necho "$* $UV_CACHE_DIR" >> {tmp_path / "calls"}\n')
    uv.chmod(0o755)
    return uv


def _calls(tmp_path: Path) -> list[str]:
    calls = tmp_path / "calls"
    return calls.read_text().splitlines() if calls.exists() else []


def test_prune_runs_once_a_day_and_keeps_its_state_outside_the_cache(tmp_path: Path, fake_uv: Path) -> None:
    """``uv cache prune`` deletes anything it doesn't recognize in the cache, so the stamp lives beside it."""
    cache = tmp_path / "uv_cache"

    assert uv_prune.prune(fake_uv, cache)
    assert not uv_prune.prune(fake_uv, cache)  # pruned within the last day

    assert _calls(tmp_path) == [f"cache prune {cache}"]
    assert (tmp_path / "uv_cache.prune" / "last_prune").is_file()
    assert not cache.exists()


def test_a_running_prune_makes_a_second_one_return(tmp_path: Path, fake_uv: Path) -> None:
    cache = tmp_path / "uv_cache"
    state = uv_prune.state_dir(cache)
    state.mkdir()
    with open(state / "prune.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert not uv_prune.prune(fake_uv, cache)
    assert _calls(tmp_path) == []


def test_schedule_detaches_and_skips_a_cache_pruned_recently(monkeypatch, tmp_path: Path, fake_uv: Path) -> None:
    popen = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", popen)
    cache = tmp_path / "uv_cache"

    uv_prune.schedule(fake_uv, cache)
    assert popen.call_count == 1
    assert popen.call_args.kwargs["start_new_session"] is True  # the build doesn't wait for it

    (uv_prune.state_dir(cache) / "last_prune").touch()
    uv_prune.schedule(fake_uv, cache)
    assert popen.call_count == 1


@pytest.mark.parametrize(("uv_version", "scheduled"), [(UV_VERSION, True), ("0.4.0", False)])
def test_a_build_prunes_only_with_the_default_uv(
    monkeypatch, tmp_path: Path, fake_uv: Path, uv_version: str, scheduled: bool
) -> None:
    """An env pinned to another uv could misjudge the default uv's cache entries, so it never prunes."""
    (tmp_path / "bin").mkdir()
    fake_uv.rename(tmp_path / "bin" / "uv")
    instance = ToolInstance.__new__(ToolInstance)
    instance.env_path = tmp_path
    monkeypatch.setattr(ToolInstance, "_get_uv_version", lambda self: uv_version)
    schedule = MagicMock()
    monkeypatch.setattr(uv_prune, "schedule", schedule)

    instance._schedule_uv_prune(tmp_path / "uv_cache")

    assert schedule.called is scheduled
