"""Tests for cgroup-aware memory detection in ``proto_tools.utils.system_info``."""

from __future__ import annotations

from pathlib import Path

import pytest

from proto_tools.utils import system_info


def test_read_cgroup_value_handles_max_number_and_garbage(tmp_path: Path) -> None:
    """``memory.max``-style files parse to int, with ``"max"`` / missing / junk → None."""
    f = tmp_path / "memory.max"
    f.write_text("max\n")
    assert system_info._read_cgroup_value(f) is None  # "max" == unbounded
    f.write_text("85899345920\n")
    assert system_info._read_cgroup_value(f) == 85899345920
    f.write_text("not-a-number")
    assert system_info._read_cgroup_value(f) is None
    assert system_info._read_cgroup_value(tmp_path / "does-not-exist") is None


def test_available_memory_prefers_smaller_of_physical_and_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cgroup cap below physical RAM is the budget (Slurm/container case)."""
    monkeypatch.setattr(system_info, "_get_ram_gb", lambda: 1024.0)  # 1 TB physical
    monkeypatch.setattr(system_info, "_cgroup_memory_limit_bytes", lambda: 80 * 1024**3)  # 80 GiB cap
    assert system_info.available_memory_bytes() == 80 * 1024**3


def test_available_memory_falls_back_to_physical_when_uncapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no cgroup limit, the budget is physical RAM."""
    monkeypatch.setattr(system_info, "_get_ram_gb", lambda: 64.0)
    monkeypatch.setattr(system_info, "_cgroup_memory_limit_bytes", lambda: None)
    assert system_info.available_memory_bytes() == int(64.0 * 1024**3)


def test_available_memory_zero_when_nothing_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Degrades to 0 when neither physical RAM nor a cgroup limit is readable."""
    monkeypatch.setattr(system_info, "_get_ram_gb", lambda: 0.0)
    monkeypatch.setattr(system_info, "_cgroup_memory_limit_bytes", lambda: None)
    assert system_info.available_memory_bytes() == 0
