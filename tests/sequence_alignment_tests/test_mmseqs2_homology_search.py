"""Tests for the mmseqs2-homology-search tool (protein; unpaired + taxonomy-paired)."""

import logging
import platform
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.databases import DatasetRegistry, dataset_slug, get_dataset_dir
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    Mmseqs2HomologySearchQuery,
    run_mmseqs2_homology_search,
)
from proto_tools.tools.sequence_alignment.mmseqs2.homology_search import (
    _assemble_paired_result,
    _check_dataset_provisioned,
    _rename_a3m_to_sequence_id,
)
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)


# ============================================================================
# Test Data
# ============================================================================

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
HEMOGLOBIN_ALPHA = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"

# Full-length human hemoglobin alpha/beta: a vertebrate-wide heterodimer, so the chains pair into deep, row-aligned MSAs.
HBA_HUMAN = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
HBB_HUMAN = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"


# ============================================================================
# Mocking helpers (no GPU / no provisioned DB needed)
# ============================================================================


def _write_a3m(path: Path, query_seq: str, n_rows: int) -> None:
    """Write an A3M with a query row plus ``n_rows - 1`` equal-length homolog rows."""
    lines = [f">query\n{query_seq}"]
    lines += [f">hom{i}\n{query_seq}" for i in range(n_rows - 1)]
    path.write_text("\n".join(lines) + "\n")


def _provision_fake_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal on-disk cache so the pre-dispatch provisioning check passes."""
    entry = DatasetRegistry.get("uniref30-2302")
    cache = tmp_path / dataset_slug("uniref30-2302")
    cache.mkdir()
    for out in entry.index_recipe.output_files or []:
        (cache / out.replace("{name}", dataset_slug("uniref30-2302"))).write_bytes(b"")
    monkeypatch.setattr(
        "proto_tools.tools.sequence_alignment.mmseqs2.homology_search.get_dataset_dir",
        lambda _: cache,
    )
    return cache


def _install_fake_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[dict[str, Any]],
    *,
    paired_depth: int = 3,
    drop_paired_for_chain: int | None = None,
) -> None:
    """Patch ``ToolInstance.dispatch`` to drop synthetic per-chain A3M files.

    A paired call (``pairing_strategy`` set) writes ``{i}.a3m`` (unpaired) and
    ``{i}.paired.a3m`` (row-aligned) per chain; an unpaired batch writes
    ``__q{idx}.a3m`` per sequence. ``drop_paired_for_chain`` omits one chain's
    paired file to simulate partial pairing.
    """

    def fake(toolkit: str, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        captured.append(payload)
        out_dir = Path(payload["output_dir"])
        sequences = payload["sequences"]
        if payload.get("pairing_strategy") is not None:
            for i, seq in enumerate(sequences):
                _write_a3m(out_dir / f"{i}.a3m", seq, 4)
                if i != drop_paired_for_chain:
                    _write_a3m(out_dir / f"{i}.paired.a3m", seq, paired_depth)
        else:
            for idx, seq in enumerate(sequences):
                _write_a3m(out_dir / f"__q{idx}.a3m", seq, 4)
        return {"success": True, "output_dir": payload["output_dir"], "db_name": "uniref30_2302_db"}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake))


def _uniref30_provisioned() -> bool:
    """Whether the UniRef30 dataset is fully provisioned on this host."""
    entry = DatasetRegistry.get("uniref30-2302")
    cache = get_dataset_dir("uniref30-2302")
    return (cache / f"{entry.db_prefix}.dbtype").is_file()


# ============================================================================
# Input validation
# ============================================================================


def test_string_query_sugar_creates_singleton_groups() -> None:
    """Plain strings become singleton query groups with auto-generated IDs."""
    inp = Mmseqs2HomologySearchInput(queries=[UBIQUITIN, HEMOGLOBIN_ALPHA])
    assert len(inp) == 2
    flat = inp.all_queries()
    assert len(flat) == 2
    assert all(q.sequence_id is not None and q.sequence_id.startswith("seq_") for q in flat)


def test_tuple_query_sugar_carries_id() -> None:
    """``(sequence, id)`` tuples carry through as the sequence's identifier."""
    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubi"), (HEMOGLOBIN_ALPHA, "hba")])
    ids = [q.sequence_id for q in inp.all_queries()]
    assert ids == ["ubi", "hba"]


def test_empty_queries_rejected() -> None:
    """An empty queries list is a validation error."""
    with pytest.raises(ValidationError, match="At least one query group"):
        Mmseqs2HomologySearchInput(queries=[])


def test_empty_sequence_rejected() -> None:
    """Whitespace-only sequences fail validation."""
    with pytest.raises(ValidationError, match="non-empty"):
        Mmseqs2HomologySearchQuery(sequence="   ")


def test_duplicate_sequence_ids_rejected() -> None:
    """Globally unique sequence_ids are required across all groups."""
    with pytest.raises(ValidationError, match="Duplicate sequence_id"):
        Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "x"), (HEMOGLOBIN_ALPHA, "x")])


def test_paired_group_accepted_as_nested_list() -> None:
    """A nested list of chains is accepted as one paired group."""
    inp = Mmseqs2HomologySearchInput(
        queries=[
            [
                Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
                Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
            ]
        ]
    )
    assert len(inp) == 1
    assert isinstance(inp.queries[0], list)
    assert [q.sequence_id for q in inp.queries[0]] == ["a", "b"]


# ============================================================================
# Config validation
# ============================================================================


def test_unknown_dataset_rejected() -> None:
    """Datasets must be in the registry."""
    with pytest.raises(ValidationError, match="Unknown dataset"):
        Mmseqs2HomologySearchConfig(datasets=["does-not-exist"])


def test_multi_dataset_rejected() -> None:
    """Exactly one dataset is supported per call."""
    with pytest.raises(ValidationError, match=r"[Ee]xactly one dataset"):
        Mmseqs2HomologySearchConfig(datasets=["uniref30-2302", "uniref30-2302"])


@pytest.mark.skipif(platform.system() != "Linux", reason="GPU validation is Linux-only")
def test_use_gpu_default_true() -> None:
    """GPU is on by default."""
    cfg = Mmseqs2HomologySearchConfig()
    assert cfg.use_gpu is True
    assert cfg.gpus_per_instance == 1


def test_gpus_per_instance_zero_when_cpu() -> None:
    """CPU-mode config reports zero GPU usage."""
    cfg = Mmseqs2HomologySearchConfig(use_gpu=False)
    assert cfg.gpus_per_instance == 0


def test_missing_dataset_dir_gives_provisioning_hint(tmp_path: Path) -> None:
    """Dispatch-time check points users at setup_databases.sh when the DB isn't on disk."""
    entry = DatasetRegistry.get("uniref30-2302")
    bogus_cache = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match=r"setup_databases\.sh"):
        _check_dataset_provisioned("uniref30-2302", entry, bogus_cache, require_idx_pad=True)


def test_missing_gpu_padded_marker_gives_use_gpu_false_hint(tmp_path: Path) -> None:
    """Dispatch-time check explains the missing GPU marker and suggests use_gpu=False."""
    entry = DatasetRegistry.get("uniref30-2302")
    # Provision dbtype but NOT the gpu_padded_marker — simulates a CPU-only DB build.
    (tmp_path / f"{entry.db_prefix}.dbtype").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"use_gpu=False"):
        _check_dataset_provisioned("uniref30-2302", entry, tmp_path, require_idx_pad=True)


def test_padded_marker_without_dbtype_is_rejected(tmp_path: Path) -> None:
    """Bare padded data file without sibling ``.dbtype`` is an incomplete build."""
    entry = DatasetRegistry.get("uniref30-2302")
    (tmp_path / f"{entry.db_prefix}.dbtype").write_bytes(b"")
    (tmp_path / entry.gpu_padded_marker).write_bytes(b"")
    # No <marker>.dbtype companion — must reject.
    with pytest.raises(FileNotFoundError, match=r"sibling \.dbtype"):
        _check_dataset_provisioned("uniref30-2302", entry, tmp_path, require_idx_pad=True)


def test_dispatch_payload_carries_operation_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool layer must include ``operation="homology_search"`` in the dispatch payload.

    Regression: the unified ``mmseqs2`` standalone routes by ``operation`` key.
    A previous version of the merged tool layer omitted this key, which would
    have made the dispatcher raise ``unknown operation None`` at runtime.
    """
    from proto_tools.tools.sequence_alignment.mmseqs2 import (
        Mmseqs2HomologySearchConfig,
        Mmseqs2HomologySearchInput,
        run_mmseqs2_homology_search,
    )
    from proto_tools.utils.tool_instance import ToolInstance

    # Provision a minimal cache so the pre-dispatch ``_check_dataset_provisioned`` passes.
    entry = DatasetRegistry.get("uniref30-2302")
    cache = tmp_path / dataset_slug("uniref30-2302")
    cache.mkdir()
    for out in entry.index_recipe.output_files or []:
        (cache / out.replace("{name}", dataset_slug("uniref30-2302"))).write_bytes(b"")
    monkeypatch.setattr(
        "proto_tools.tools.sequence_alignment.mmseqs2.homology_search.get_dataset_dir",
        lambda _: cache,
    )

    captured: dict[str, object] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, object], **_: object) -> dict[str, object]:
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {"success": True, "output_dir": payload["output_dir"], "db_name": entry.db_prefix}

    monkeypatch.setattr(ToolInstance, "dispatch", staticmethod(fake_dispatch))

    # The @tool wrapper catches the downstream A3M parse failure (output_dir is
    # empty) so the call returns rather than raising. We only need the payload.
    run_mmseqs2_homology_search(
        Mmseqs2HomologySearchInput(queries=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLI"]),
        Mmseqs2HomologySearchConfig(use_gpu=False),
    )

    assert captured["toolkit"] == "mmseqs2"
    assert captured["payload"]["operation"] == "homology_search"


# ============================================================================
# Per-dataset provisioning + integration — auto-skip when not on device
# ============================================================================


def _provisioned_datasets() -> list[str]:
    """Return registry keys whose declared output_files are all present on disk.

    Used to parametrize per-dataset tests so they auto-skip when a dataset
    isn't installed; a fresh dev machine sees no parametrize entries skipped
    here unless the user has run ``setup_databases.py`` for that dataset.
    """
    found: list[str] = []
    for name in DatasetRegistry.list_all():
        entry = DatasetRegistry.get(name)
        cache = get_dataset_dir(name)
        if not entry.index_recipe.output_files:
            if (cache / f"{entry.db_prefix}.dbtype").is_file():
                found.append(name)
            continue
        if all((cache / out.replace("{name}", dataset_slug(name))).exists() for out in entry.index_recipe.output_files):
            found.append(name)
    return found


@pytest.mark.parametrize("dataset_name", DatasetRegistry.list_all())
def test_registry_entry_outputs_match_disk_when_provisioned(dataset_name: str) -> None:
    """If a dataset's cache dir has its dbtype file, all declared output_files exist.

    Catches drift between an entry's ``index_recipe.output_files`` and what
    its provisioning steps actually produce. Skips when not provisioned.
    """
    entry = DatasetRegistry.get(dataset_name)
    cache = get_dataset_dir(dataset_name)
    if not (cache / f"{entry.db_prefix}.dbtype").is_file():
        pytest.skip(f"{dataset_name} not provisioned at {cache}")
    missing = [
        out
        for out in entry.index_recipe.output_files
        if not (cache / out.replace("{name}", dataset_slug(dataset_name))).exists()
    ]
    assert not missing, (
        f"{dataset_name} declared output_files {missing} are missing from {cache}; "
        f"either provisioning failed or the entry's output_files list is stale."
    )


@pytest.mark.uses_gpu
@pytest.mark.parametrize("dataset_name", _provisioned_datasets() or ["__none_provisioned__"])
def test_e2e_search_against_provisioned_protein_dataset(dataset_name: str) -> None:
    """End-to-end GPU search against each provisioned protein dataset.

    Auto-parametrizes over whatever's on disk. RNA datasets are skipped
    (current tool surface is protein-only). Asserts the search returns
    a non-empty MSA for ubiquitin (universally conserved).
    """
    if dataset_name == "__none_provisioned__":
        pytest.skip("No datasets provisioned on this device")

    entry = DatasetRegistry.get(dataset_name)
    if entry.molecule_type != "protein":
        pytest.skip(f"{dataset_name} is {entry.molecule_type}; current tool is protein-only")
    if not entry.supports_gpu:
        pytest.skip(f"{dataset_name} declares supports_gpu=False")
    if entry.a3m_adapter != "colabfold":
        pytest.skip(f"{dataset_name} uses a3m_adapter={entry.a3m_adapter!r}; tool only handles colabfold-style DBs")

    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubiquitin")])
    cfg = Mmseqs2HomologySearchConfig(datasets=[dataset_name], use_gpu=True)
    result = run_mmseqs2_homology_search(inp, cfg)

    assert result.success, f"Search against {dataset_name} failed: {result.errors}"
    assert len(result.results) == 1
    grp = result.results[0]
    assert grp.datasets_searched == [dataset_name]
    # Ubiquitin against any reasonable protein DB should return >0 homologs.
    # Use a low threshold (>5) so PDB-seqres-style tiny DBs still pass.
    assert grp.num_homologs_found[0] > 5, (
        f"{dataset_name} returned only {grp.num_homologs_found[0]} homologs for ubiquitin "
        "(expected >5 — universally conserved protein)"
    )


# ============================================================================
# Paired (multi-chain) groups — mocked dispatch, no GPU / DB needed
# ============================================================================


def test_paired_group_produces_row_aligned_paired_msas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2-chain paired group yields per-chain unpaired + row-aligned paired MSAs."""
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured, paired_depth=3)

    inp = Mmseqs2HomologySearchInput(queries=[[(UBIQUITIN, "chainA"), (HEMOGLOBIN_ALPHA, "chainB")]])
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(use_gpu=False))

    assert len(out.results) == 1
    grp = out.results[0]
    assert grp.sequence_ids == ["chainA", "chainB"]
    assert all(m is not None for m in grp.msas)
    assert all(m is not None for m in grp.paired_msas)
    # Paired MSAs are row-aligned: equal depth across chains.
    depths = {m.num_sequences for m in grp.paired_msas if m is not None}
    assert depths == {3}
    # Exactly one paired dispatch (the group), submitted as a complex.
    assert len(captured) == 1
    assert captured[0]["pairing_strategy"] == 0  # greedy default
    assert captured[0]["sequences"] == [UBIQUITIN, HEMOGLOBIN_ALPHA]


@pytest.mark.parametrize(("strategy", "expected_int"), [("greedy", 0), ("complete", 1)])
def test_pairing_strategy_maps_to_mmseqs_int(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str, expected_int: int
) -> None:
    """``pairing_strategy`` reaches the dispatch payload as the mmseqs pairing-mode int."""
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured)

    inp = Mmseqs2HomologySearchInput(queries=[[(UBIQUITIN, "a"), (HEMOGLOBIN_ALPHA, "b")]])
    run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(use_gpu=False, pairing_strategy=strategy))

    assert captured[0]["pairing_strategy"] == expected_int


def test_singletons_batch_while_paired_group_dispatches_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Singletons share one unpaired batch; the paired group dispatches on its own, order preserved."""
    _provision_fake_cache(tmp_path, monkeypatch)
    captured: list[dict[str, Any]] = []
    _install_fake_dispatch(monkeypatch, captured)

    inp = Mmseqs2HomologySearchInput(
        queries=[
            (UBIQUITIN, "solo1"),
            [(UBIQUITIN, "pairA"), (HEMOGLOBIN_ALPHA, "pairB")],
            (HEMOGLOBIN_ALPHA, "solo2"),
        ]
    )
    out = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig(use_gpu=False))

    # Results stay parallel to input groups.
    assert [r.sequence_ids for r in out.results] == [["solo1"], ["pairA", "pairB"], ["solo2"]]
    # Singleton groups carry no paired MSAs; the paired group does.
    assert out.results[0].paired_msas == [None]
    assert out.results[2].paired_msas == [None]
    assert all(m is not None for m in out.results[1].paired_msas)
    # One batched unpaired dispatch (both singletons) + one paired dispatch.
    unpaired = [p for p in captured if p["pairing_strategy"] is None]
    paired = [p for p in captured if p["pairing_strategy"] is not None]
    assert len(unpaired) == 1 and unpaired[0]["sequences"] == [UBIQUITIN, HEMOGLOBIN_ALPHA]
    assert len(paired) == 1 and paired[0]["sequences"] == [UBIQUITIN, HEMOGLOBIN_ALPHA]


def test_assemble_paired_result_partial_pairing_hard_fails(tmp_path: Path) -> None:
    """One chain paired, the other empty → hard-fail (matches colabfold-search)."""
    members = [
        Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
        Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
    ]
    # Chain 0 gets a paired MSA; chain 1 gets none.
    _write_a3m(tmp_path / "0.a3m", UBIQUITIN, 4)
    _write_a3m(tmp_path / "0.paired.a3m", UBIQUITIN, 3)
    _write_a3m(tmp_path / "1.a3m", HEMOGLOBIN_ALPHA, 4)
    with pytest.raises(RuntimeError, match="partial MSAs"):
        _assemble_paired_result(tmp_path, members, "uniref30-2302")


def test_assemble_paired_result_no_pairing_returns_all_none(tmp_path: Path) -> None:
    """No paired files at all (no shared species) → paired_msas all None, no error."""
    members = [
        Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
        Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
    ]
    _write_a3m(tmp_path / "0.a3m", UBIQUITIN, 4)
    _write_a3m(tmp_path / "1.a3m", HEMOGLOBIN_ALPHA, 4)
    result = _assemble_paired_result(tmp_path, members, "uniref30-2302")
    assert result.paired_msas == [None, None]
    assert all(m is not None for m in result.msas)  # unpaired still present


@pytest.mark.integration
@pytest.mark.uses_cpu
@pytest.mark.slow
@pytest.mark.skip_ci
def test_real_paired_search_against_mini_db() -> None:
    """Real taxonomy-paired search against the auto-provisioning mini SwissProt DB.

    Exercises the full ``--unpack 0`` + ``unpackdb`` paired pipeline end-to-end
    (no GPU, no manual DB setup — the ``tiny-test-colabfold`` fixture
    auto-provisions on first run). Hemoglobin alpha+beta pair into deep,
    row-aligned MSAs.
    """
    inp = Mmseqs2HomologySearchInput(queries=[[(HBA_HUMAN, "hba"), (HBB_HUMAN, "hbb")]])
    cfg = Mmseqs2HomologySearchConfig(datasets=["tiny-test-colabfold"], use_gpu=False)
    out = run_mmseqs2_homology_search(inp, cfg)

    assert out.success, f"paired search failed: {out.errors}"
    assert len(out.results) == 1
    grp = out.results[0]
    assert grp.sequence_ids == ["hba", "hbb"]
    # Both chains paired and row-aligned: one depth shared across the group.
    assert all(m is not None for m in grp.paired_msas)
    depths = {m.num_sequences for m in grp.paired_msas if m is not None}
    assert len(depths) == 1 and next(iter(depths)) > 5, (
        f"expected deep equal-depth paired MSAs, got {[m.num_sequences for m in grp.paired_msas if m]}"
    )
    # Unpaired per-chain MSAs are present alongside the paired ones.
    assert all(m is not None for m in grp.msas)


def test_rename_a3m_avoids_collision_with_adversarial_numeric_ids(tmp_path: Path) -> None:
    """Regression: numeric sequence_ids that look like ``__q{idx}`` indices must not collide.

    Pre-fix the FASTA used user sequence_ids as headers, so colabfold_search's
    output naming collided with user IDs (sequence_id="1" for the query at
    idx=0 would clobber the second query's "1.a3m" before our code ran). The
    fix uses internal ``__q{idx}`` source names that can't collide with user
    IDs. This test exercises that path directly with adversarial swapped IDs
    — fast, runs in CI, doesn't need a GPU or provisioned database.
    """
    # Two distinct internal A3Ms — each carries identifiable content so we can
    # verify no swap or clobber.
    (tmp_path / "__q0.a3m").write_text(">__q0\nUBIQUITIN_PLACEHOLDER\n>homolog_a\nA\n>homolog_b\nB\n")
    (tmp_path / "__q1.a3m").write_text(">__q1\nHEMOGLOBIN_PLACEHOLDER\n>homolog_x\nX\n")

    # Adversarial IDs: idx=0 wants "1", idx=1 wants "0" — exact swap.
    p0 = _rename_a3m_to_sequence_id(tmp_path, idx=0, sequence_id="1")
    p1 = _rename_a3m_to_sequence_id(tmp_path, idx=1, sequence_id="0")

    assert p0 == tmp_path / "1.a3m"
    assert p1 == tmp_path / "0.a3m"

    # Critical: each user-facing file holds the right query's content.
    assert "UBIQUITIN_PLACEHOLDER" in p0.read_text()
    assert "HEMOGLOBIN_PLACEHOLDER" in p1.read_text()

    # Query headers were rewritten; homolog headers untouched.
    p0_lines = p0.read_text().split("\n")
    p1_lines = p1.read_text().split("\n")
    assert p0_lines[0] == ">1"
    assert p1_lines[0] == ">0"
    assert ">homolog_a" in p0_lines
    assert ">homolog_x" in p1_lines
