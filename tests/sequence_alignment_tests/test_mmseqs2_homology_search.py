"""Tests for the mmseqs2-homology-search tool (Phase 3: protein, unpaired)."""

import logging
import platform
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_alignment.databases import DatasetRegistry, dataset_slug, get_dataset_dir
from proto_tools.tools.sequence_alignment.mmseqs2_homology_search import (
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    Mmseqs2HomologySearchQuery,
    run_mmseqs2_homology_search,
)
from proto_tools.tools.sequence_alignment.mmseqs2_homology_search.mmseqs2_homology_search import (
    _check_dataset_provisioned,
    _rename_a3m_to_sequence_id,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Test Data
# ============================================================================

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
HEMOGLOBIN_ALPHA = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"


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


def test_paired_group_rejected_in_phase_3() -> None:
    """Phase 3 rejects paired groups with a pointer to issue 543."""
    with pytest.raises(ValidationError, match=r"issues/543"):
        Mmseqs2HomologySearchInput(
            queries=[
                [
                    Mmseqs2HomologySearchQuery(sequence=UBIQUITIN, sequence_id="a"),
                    Mmseqs2HomologySearchQuery(sequence=HEMOGLOBIN_ALPHA, sequence_id="b"),
                ]
            ]
        )


# ============================================================================
# Config validation
# ============================================================================


def test_unknown_dataset_rejected() -> None:
    """Datasets must be in the registry."""
    with pytest.raises(ValidationError, match="Unknown dataset"):
        Mmseqs2HomologySearchConfig(datasets=["does-not-exist"])


def test_multi_dataset_rejected_in_phase_3() -> None:
    """Phase 3 supports exactly one dataset."""
    with pytest.raises(ValidationError, match="exactly one dataset"):
        Mmseqs2HomologySearchConfig(datasets=["uniref30-2302", "uniref30-2302"])


@pytest.mark.skipif(platform.system() != "Linux", reason="GPU validation is Linux-only")
def test_use_gpu_default_true() -> None:
    """GPU is on by default."""
    cfg = Mmseqs2HomologySearchConfig()
    assert cfg.use_gpu is True
    assert cfg.devices_per_instance == 1


def test_devices_per_instance_zero_when_cpu() -> None:
    """CPU-mode config reports zero GPU usage."""
    cfg = Mmseqs2HomologySearchConfig(use_gpu=False)
    assert cfg.devices_per_instance == 0


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
        pytest.skip(
            f"{dataset_name} uses a3m_adapter={entry.a3m_adapter!r}; tool currently only handles colabfold-style DBs (Phase 4)"
        )

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
