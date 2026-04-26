"""Tests for the mmseqs2-homology-search tool (Phase 3: protein, unpaired)."""

import logging
import platform
from pathlib import Path

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_alignment.databases import DatasetRegistry, get_dataset_dir
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


def test_missing_idx_pad_gives_use_gpu_false_hint(tmp_path: Path) -> None:
    """Dispatch-time check explains the .idx_pad omission and suggests use_gpu=False."""
    entry = DatasetRegistry.get("uniref30-2302")
    # Provision dbtype but NOT idx_pad — simulates a CPU-only DB build.
    (tmp_path / f"{entry.db_prefix}.dbtype").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"use_gpu=False"):
        _check_dataset_provisioned("uniref30-2302", entry, tmp_path, require_idx_pad=True)


# ============================================================================
# Integration (GPU) — requires uniref30 + GPU
# ============================================================================


@pytest.mark.uses_gpu
def test_gpu_search_against_uniref30_finds_homologs() -> None:
    """End-to-end GPU search against the provisioned UniRef30 DB.

    Skipped automatically without a GPU. Skipped explicitly when UniRef30 is
    not provisioned. On a working host, ubiquitin should return >>1000 homologs
    (it's one of the most conserved proteins; 9902 observed at PR time).
    """
    if not _uniref30_provisioned():
        pytest.skip("uniref30-2302 not provisioned locally")

    inp = Mmseqs2HomologySearchInput(queries=[(UBIQUITIN, "ubiquitin")])
    cfg = Mmseqs2HomologySearchConfig(use_gpu=True)
    result = run_mmseqs2_homology_search(inp, cfg)

    assert result.success
    assert len(result.results) == 1
    grp = result.results[0]
    assert grp.sequence_ids == ["ubiquitin"]
    assert grp.datasets_searched == ["uniref30-2302"]
    assert grp.paired_msas == [None]
    assert grp.num_homologs_found[0] > 100
    assert grp.msas[0] is not None


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
