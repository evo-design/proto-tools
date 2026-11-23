"""Tests for ``proto_tools.databases``: provisioning registered datasets, and the MMseqs2 database type."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pytest

from proto_tools.databases import BuildStep, DatasetRegistry, DownloadSpec, MMseqs2DatabaseEntry
from proto_tools.databases import provision as provision_module

CURL = ("curl", ["curl", "-L", "--fail"])


def _serving(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    """Stub the downloader so it writes ``payload`` to whatever path the argv names."""

    def fake_run(cmd: list[str], *_: Any, **__: Any) -> None:
        Path(cmd[cmd.index("-o") + 1]).write_bytes(payload)

    monkeypatch.setattr(provision_module.subprocess, "run", fake_run)


def test_split_memory_limit_caps_at_safety_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split limit is a GiB-suffixed safety fraction of the cgroup-aware budget.

    The ``G`` suffix is essential: mmseqs misreads a bare byte count as an
    enormous (effectively unbounded) limit, so the cap silently wouldn't apply.
    """
    monkeypatch.setattr(provision_module, "available_memory_bytes", lambda: 80 * 1024**3)
    # 80 GiB * 0.7 = 56 GiB.
    assert provision_module._split_memory_limit() == "56G"


def test_split_memory_limit_zero_when_budget_undetectable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the budget can't be detected, fall back to ``"0"`` (mmseqs default = all available)."""
    monkeypatch.setattr(provision_module, "available_memory_bytes", lambda: 0)
    assert provision_module._split_memory_limit() == "0"


def test_run_step_substitutes_name_and_split_memory_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``_run_step`` substitutes both ``{name}`` and ``{split_memory_limit}`` in the argv."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], *_: Any, **kwargs: Any) -> None:
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")

    monkeypatch.setattr(provision_module.subprocess, "run", fake_run)
    step = BuildStep(
        command=["mmseqs", "createindex", "{name}_db", "tmp", "--split-memory-limit", "{split_memory_limit}"],
        description="build index",
    )
    provision_module._run_step(step, tmp_path, "colabfold-envdb-202108", split_memory_limit="56G")

    assert captured["cmd"] == [
        "mmseqs",
        "createindex",
        "colabfold_envdb_202108_db",
        "tmp",
        "--split-memory-limit",
        "56G",
    ]
    assert captured["cwd"] == tmp_path


def test_incomplete_cached_file_is_downloaded_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cached file whose size disagrees with the spec is replaced, not trusted."""
    spec = DownloadSpec(url="https://example.invalid/g.fa.gz", filename="g.fa.gz", expected_bytes=8)
    (tmp_path / "g.fa.gz").write_bytes(b"cut")
    _serving(monkeypatch, b"complete")

    assert provision_module._download_file(spec, tmp_path, CURL) is True
    assert (tmp_path / "g.fa.gz").read_bytes() == b"complete"


def test_complete_cached_file_is_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cached file of the declared size is kept without re-downloading."""
    spec = DownloadSpec(url="https://example.invalid/g.fa.gz", filename="g.fa.gz", expected_bytes=8)
    (tmp_path / "g.fa.gz").write_bytes(b"complete")

    def explode(*_: Any, **__: Any) -> None:
        raise AssertionError("should not download an already-complete file")

    monkeypatch.setattr(provision_module.subprocess, "run", explode)

    assert provision_module._download_file(spec, tmp_path, CURL) is True


def test_short_download_leaves_nothing_at_the_final_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A transfer that ends early raises and leaves neither the target nor the partial file."""
    spec = DownloadSpec(url="https://example.invalid/g.fa.gz", filename="g.fa.gz", expected_bytes=8)
    _serving(monkeypatch, b"cut")

    with pytest.raises(RuntimeError, match="did not download completely"):
        provision_module._download_file(spec, tmp_path, CURL)

    assert not (tmp_path / "g.fa.gz").exists()
    assert list(tmp_path.iterdir()) == []


def test_checksum_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A download of the right size but the wrong SHA-256 is discarded."""
    spec = DownloadSpec(url="https://example.invalid/g.fa.gz", filename="g.fa.gz", sha256="00" * 32)
    _serving(monkeypatch, b"complete")

    with pytest.raises(RuntimeError, match="did not download completely"):
        provision_module._download_file(spec, tmp_path, CURL)

    assert list(tmp_path.iterdir()) == []


def test_unverifiable_spec_keeps_previous_behaviour(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without a declared size or checksum, any existing file is still reused."""
    spec = DownloadSpec(url="https://example.invalid/g.fa.gz", filename="g.fa.gz")
    (tmp_path / "g.fa.gz").write_bytes(b"whatever")

    def explode(*_: Any, **__: Any) -> None:
        raise AssertionError("should not download when the file is present and unverifiable")

    monkeypatch.setattr(provision_module.subprocess, "run", explode)

    assert provision_module._download_file(spec, tmp_path, CURL) is True


def test_provision_reports_its_long_phases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> None:
    """Inside a worker log lines reach no caller, so a multi-gigabyte first run must report status."""
    events: list[str] = []
    monkeypatch.setattr(provision_module, "_which_downloader", lambda: CURL)
    monkeypatch.setattr(provision_module, "_download_file", lambda spec, *_: events.append(f"download {spec.filename}"))
    monkeypatch.setattr(provision_module, "_run_step", lambda step, *_, **__: events.append("step"))

    class Phases(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if getattr(record, "update_status", False):
                events.append(f"status {record.getMessage()}")

    handler = Phases()
    provision_module.logger.addHandler(handler)
    name = "uniref30-2302"
    entry = DatasetRegistry.get(name)
    try:
        with caplog.at_level(logging.INFO, logger=provision_module.logger.name):
            provision_module.provision(name, workdir=tmp_path)
    finally:
        provision_module.logger.removeHandler(handler)

    downloads = [event for event in events if event.startswith("download")]
    steps = [event for event in events if event == "step"]
    assert events == [
        f"status Downloading sources for {entry.display_name}",
        *downloads,
        f"status Building {entry.display_name}",
        *steps,
    ]
    assert downloads and steps


def test_the_dataset_layer_imports_no_tool() -> None:
    """Datasets are defined, resolved and provisioned without reaching into any tool that consumes them."""
    databases_dir = Path(provision_module.__file__).parent
    offenders = []
    for path in sorted(databases_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            offenders += [f"{path.name}: {m}" for m in modules if m.startswith("proto_tools.tools")]
    assert not offenders


def _mmseqs2_entry(db_prefix: str, gpu_db: str | None) -> MMseqs2DatabaseEntry:
    template = DatasetRegistry.get("uniref30-2302", MMseqs2DatabaseEntry)
    return template.model_copy(update={"db_prefix": db_prefix, "gpu_db": gpu_db})


@pytest.mark.parametrize(
    ("gpu_db", "on_disk", "built_cpu", "built_gpu"),
    [
        # AF3-style: the database is itself padded, so one database serves CPU and GPU search
        ("db", ["db", "db.dbtype"], True, True),
        # ColabFold-style: GPU search reads a separate padded sibling, which may not be built yet
        ("db.idx_pad", ["db", "db.dbtype"], True, False),
        ("db.idx_pad", ["db", "db.dbtype", "db.idx_pad", "db.idx_pad.dbtype"], True, True),
        # CPU-only
        (None, ["db", "db.dbtype"], True, False),
        (None, [], False, False),
    ],
)
def test_an_mmseqs2_database_knows_what_it_searches_and_when_it_is_built(
    tmp_path: Path, gpu_db: str | None, on_disk: list[str], built_cpu: bool, built_gpu: bool
) -> None:
    for name in on_disk:
        (tmp_path / name).write_bytes(b"")
    entry = _mmseqs2_entry("db", gpu_db)

    assert entry.is_built(tmp_path, gpu=False) is built_cpu
    assert entry.is_built(tmp_path, gpu=True) is built_gpu
    assert entry.search_db(tmp_path, gpu=False) == tmp_path / "db"
    if gpu_db is None:
        with pytest.raises(ValueError, match="does not support GPU search"):
            entry.search_db(tmp_path, gpu=True)
    else:
        assert entry.search_db(tmp_path, gpu=True) == tmp_path / gpu_db


def test_a_lookup_names_the_dataset_kind_it_needs() -> None:
    """A genome asked for as an MMseqs2 database is refused, rather than read through missing fields."""
    assert "grch38" in DatasetRegistry.list_all()
    assert "grch38" not in DatasetRegistry.list_all(MMseqs2DatabaseEntry)
    with pytest.raises(ValueError, match="not an MMseqs2DatabaseEntry"):
        DatasetRegistry.get("grch38", MMseqs2DatabaseEntry)
