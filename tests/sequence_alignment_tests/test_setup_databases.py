"""Tests for the mmseqs2 ``setup_databases`` provisioning helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from proto_tools.databases import DatasetRegistry, IndexStep
from proto_tools.tools.sequence_alignment.mmseqs2 import setup_databases


def test_split_memory_limit_caps_at_safety_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split limit is a GiB-suffixed safety fraction of the cgroup-aware budget.

    The ``G`` suffix is essential: mmseqs misreads a bare byte count as an
    enormous (effectively unbounded) limit, so the cap silently wouldn't apply.
    """
    monkeypatch.setattr(setup_databases, "available_memory_bytes", lambda: 80 * 1024**3)
    # 80 GiB * 0.7 = 56 GiB.
    assert setup_databases._split_memory_limit() == "56G"


def test_split_memory_limit_zero_when_budget_undetectable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the budget can't be detected, fall back to ``"0"`` (mmseqs default = all available)."""
    monkeypatch.setattr(setup_databases, "available_memory_bytes", lambda: 0)
    assert setup_databases._split_memory_limit() == "0"


def test_run_step_substitutes_name_and_split_memory_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``_run_step`` substitutes both ``{name}`` and ``{split_memory_limit}`` in the argv."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], *_: Any, **kwargs: Any) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")

    monkeypatch.setattr(setup_databases.subprocess, "run", fake_run)
    step = IndexStep(
        command=["mmseqs", "createindex", "{name}_db", "tmp", "--split-memory-limit", "{split_memory_limit}"],
        description="build index",
    )
    setup_databases._run_step(step, tmp_path, "colabfold-envdb-202108", split_memory_limit="56G")

    assert captured["cmd"] == [
        "mmseqs",
        "createindex",
        "colabfold_envdb_202108_db",
        "tmp",
        "--split-memory-limit",
        "56G",
    ]
    assert captured["cwd"] == tmp_path


@pytest.mark.parametrize(
    "dataset_name",
    ["colabfold-envdb-202108", "uniref30-2302", "rfam-15-1", "rfam-14-9-90-80", "rnacentral-active-90-80"],
)
def test_createindex_recipes_use_cgroup_aware_split_limit(dataset_name: str) -> None:
    """Every ``createindex`` step carries ``--split-memory-limit {split_memory_limit}`` (no hardcoded ``--split``)."""
    entry = DatasetRegistry.get(dataset_name)
    createindex_steps = [s for s in entry.index_recipe.steps if "createindex" in s.command]
    assert createindex_steps, f"{dataset_name} has no createindex step"
    for step in createindex_steps:
        assert "{split_memory_limit}" in step.command, f"{dataset_name} createindex missing split-memory-limit"
        assert "--split" not in step.command, f"{dataset_name} should not hardcode --split (use the memory limit)"
