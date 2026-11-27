"""Tests for the mmseqs2 ``setup_databases`` CLI and the MMseqs2 build recipes it runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from proto_tools.databases import DatasetRegistry
from proto_tools.tools.sequence_alignment.mmseqs2 import setup_databases


@pytest.mark.parametrize(
    "dataset_name",
    ["colabfold-envdb-202108", "uniref30-2302", "rfam-15-1", "rfam-14-9-90-80", "rnacentral-active-90-80"],
)
def test_createindex_recipes_use_cgroup_aware_split_limit(dataset_name: str) -> None:
    """Every ``createindex`` step carries ``--split-memory-limit {split_memory_limit}`` (no hardcoded ``--split``)."""
    entry = DatasetRegistry.get(dataset_name)
    createindex_steps = [s for s in entry.build.steps if "createindex" in s.command]
    assert createindex_steps, f"{dataset_name} has no createindex step"
    for step in createindex_steps:
        assert "{split_memory_limit}" in step.command, f"{dataset_name} createindex missing split-memory-limit"
        assert "--split" not in step.command, f"{dataset_name} should not hardcode --split (use the memory limit)"


def test_ensure_mmseqs_on_path_prepends_toolkit_bin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The helper builds the mmseqs2 toolkit env and prepends its bin to PATH so recipe steps resolve ``mmseqs`` even on hosts without it on the caller's PATH."""
    fake_env_path = tmp_path / "fake_mmseqs2_env"
    (fake_env_path / "bin").mkdir(parents=True)
    fake_mmseqs = fake_env_path / "bin" / "mmseqs"
    fake_mmseqs.write_text("#!/bin/sh\necho stub\n")
    fake_mmseqs.chmod(0o755)

    ensure_calls = {"count": 0}

    class FakeTool:
        env_path = fake_env_path

        def ensure_ready(self) -> None:
            ensure_calls["count"] += 1

    monkeypatch.setattr(setup_databases.ToolInstance, "get", staticmethod(lambda _: FakeTool()))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    setup_databases._ensure_mmseqs_on_path()

    assert ensure_calls["count"] == 1, "ensure_ready() should fire to build the toolkit env on first call"
    new_path = setup_databases.os.environ["PATH"]
    assert new_path.startswith(str(fake_env_path / "bin")), f"toolkit bin should be prepended to PATH, got {new_path!r}"
    assert "/usr/bin:/bin" in new_path, "system PATH must be preserved after the prepend"


def test_ensure_mmseqs_on_path_raises_when_binary_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Helper errors clearly when ensure_ready() returns but no ``mmseqs`` binary is present."""

    class FakeTool:
        env_path = tmp_path  # no bin/mmseqs inside

        def ensure_ready(self) -> None:
            pass

    monkeypatch.setattr(setup_databases.ToolInstance, "get", staticmethod(lambda _: FakeTool()))

    with pytest.raises(RuntimeError, match="mmseqs binary not found"):
        setup_databases._ensure_mmseqs_on_path()
