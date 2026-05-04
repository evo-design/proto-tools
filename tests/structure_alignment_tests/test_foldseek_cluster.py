"""Tests for foldseek-cluster (local-only structural clustering)."""

import gzip
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldseekClusterConfig,
    FoldseekClusterInput,
    run_foldseek_cluster,
)
from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import _parse_cluster_tsv

_TINY_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
_TINY_CIF = "data_TEST\nloop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── _parse_cluster_tsv ────────────────────────────────────────────────────────


def test_parse_cluster_tsv_groups_members_by_representative():
    """Multiple lines per representative collapse into one cluster with all members."""
    clusters = _parse_cluster_tsv("rep1\tmem1\nrep1\tmem2\nrep1\trep1\nrep2\tmem3\nrep2\trep2\n")

    by_rep = {c.representative_id: c for c in clusters}
    assert sorted(by_rep["rep1"].member_ids) == ["mem1", "mem2", "rep1"]
    assert sorted(by_rep["rep2"].member_ids) == ["mem3", "rep2"]


def test_parse_cluster_tsv_skips_blank_and_short_lines():
    """Empty / single-column lines are silently dropped."""
    clusters = _parse_cluster_tsv("rep1\tmem1\n\nshort_line\nrep1\tmem2\n")

    assert len(clusters) == 1
    assert sorted(clusters[0].member_ids) == ["mem1", "mem2"]


# ── FoldseekClusterInput validators ───────────────────────────────────────────


def test_input_requires_at_least_two_structures():
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekClusterInput(structures=[_TINY_PDB])


def test_input_requires_one_of_structures_or_dir():
    with pytest.raises(ValidationError, match="exactly one"):
        FoldseekClusterInput()


def test_input_rejects_both_structures_and_dir(tmp_path):
    (tmp_path / "a.pdb").write_text(_TINY_PDB)
    (tmp_path / "b.pdb").write_text(_TINY_PDB)
    with pytest.raises(ValidationError, match="exactly one"):
        FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB], structures_dir=str(tmp_path))


def test_input_rejects_ids_with_dir(tmp_path):
    (tmp_path / "a.pdb").write_text(_TINY_PDB)
    (tmp_path / "b.pdb").write_text(_TINY_PDB)
    with pytest.raises(ValidationError, match="may not be combined"):
        FoldseekClusterInput(structures_dir=str(tmp_path), structure_ids=["x", "y"])


def test_input_rejects_id_count_mismatch():
    with pytest.raises(ValidationError, match="structure_ids length"):
        FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=["only-one"])


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "/etc/passwd", "..", ".", "path/with/slash", "path\\with\\backslash", ""],
    ids=["dot-dot-prefix", "absolute-path", "dotdot", "dot", "fwd-slash", "back-slash", "empty"],
)
def test_input_rejects_unsafe_structure_ids(bad_id):
    """IDs are written to disk as ``{id}.{ext}``; reject path-traversal-shaped values."""
    with pytest.raises(ValidationError, match="not a safe filename"):
        FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=[bad_id, "ok"])


def test_input_rejects_nonexistent_dir():
    with pytest.raises(ValidationError, match="not an existing directory"):
        FoldseekClusterInput(structures_dir="/nonexistent/path/abcxyz")


def test_input_rejects_dir_with_too_few_files(tmp_path):
    (tmp_path / "only.pdb").write_text(_TINY_PDB)
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekClusterInput(structures_dir=str(tmp_path))


def test_input_rejects_duplicate_user_supplied_ids():
    """Duplicate IDs would clobber each other in foldseek's TSV output."""
    with pytest.raises(ValidationError, match="duplicated"):
        FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=["a", "a"])


def test_input_rejects_dir_with_colliding_stems(tmp_path):
    """`protein.pdb` + `protein.cif` produce identical stems and would collide silently."""
    (tmp_path / "protein.pdb").write_text(_TINY_PDB)
    (tmp_path / "protein.cif").write_text(_TINY_CIF)
    with pytest.raises(ValidationError, match="duplicated"):
        FoldseekClusterInput(structures_dir=str(tmp_path))


# ── FoldseekClusterInput resolution from directory ────────────────────────────


def test_input_resolves_dir_with_mixed_formats(tmp_path):
    """Mixed .pdb + .cif: both read into structures, stems become structure_ids, structures_dir cleared."""
    (tmp_path / "alpha.pdb").write_text(_TINY_PDB)
    (tmp_path / "beta.cif").write_text(_TINY_CIF)

    inputs = FoldseekClusterInput(structures_dir=str(tmp_path))

    assert inputs.structures_dir is None
    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_PDB in (inputs.structures or [])
    assert _TINY_CIF in (inputs.structures or [])


def test_input_resolves_dir_decompresses_gz_files(tmp_path):
    """`.pdb.gz` / `.cif.gz` are decompressed during read."""
    with gzip.open(tmp_path / "alpha.pdb.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_PDB)
    with gzip.open(tmp_path / "beta.cif.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_CIF)

    inputs = FoldseekClusterInput(structures_dir=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_PDB in (inputs.structures or [])
    assert _TINY_CIF in (inputs.structures or [])


def test_input_dir_skips_unsupported_files(tmp_path):
    """Files without supported extensions (e.g. README.md) are filtered out."""
    (tmp_path / "alpha.pdb").write_text(_TINY_PDB)
    (tmp_path / "beta.pdb").write_text(_TINY_PDB)
    (tmp_path / "README.md").write_text("just notes")
    (tmp_path / ".DS_Store").write_text("junk")

    inputs = FoldseekClusterInput(structures_dir=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]


# ── run_foldseek_cluster (mocked dispatch) ────────────────────────────────────


def test_run_foldseek_cluster_dispatches_easy_cluster():
    """Wrapper sends operation=easy_cluster, default IDs, and per-structure formats."""
    inputs = FoldseekClusterInput(structures=[_TINY_PDB, _TINY_CIF, _TINY_PDB])

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_cluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"clusters_tsv": "structure_0\tstructure_0\nstructure_0\tstructure_1\n"}
        output = run_foldseek_cluster(inputs, FoldseekClusterConfig())

    assert output.success
    assert output.num_structures == 3

    payload = mock_dispatch.call_args.args[1]
    assert mock_dispatch.call_args.args[0] == "foldseek"
    assert payload["operation"] == "easy_cluster"
    assert payload["structure_ids"] == ["structure_0", "structure_1", "structure_2"]
    assert payload["structure_formats"] == ["pdb", "cif", "pdb"]


def test_run_foldseek_cluster_uses_user_supplied_ids():
    inputs = FoldseekClusterInput(
        structures=[_TINY_PDB, _TINY_PDB],
        structure_ids=["my-protein-a", "my-protein-b"],
    )

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_cluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"clusters_tsv": ""}
        run_foldseek_cluster(inputs, FoldseekClusterConfig())

    assert mock_dispatch.call_args.args[1]["structure_ids"] == ["my-protein-a", "my-protein-b"]


def test_run_foldseek_cluster_with_directory(tmp_path):
    """Directory input flows through the wrapper: stems → IDs, per-file format detected."""
    (tmp_path / "alpha.pdb").write_text(_TINY_PDB)
    (tmp_path / "beta.cif").write_text(_TINY_CIF)
    inputs = FoldseekClusterInput(structures_dir=str(tmp_path))

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_cluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"clusters_tsv": ""}
        run_foldseek_cluster(inputs, FoldseekClusterConfig())

    payload = mock_dispatch.call_args.args[1]
    assert dict(zip(payload["structure_ids"], payload["structure_formats"], strict=True)) == {
        "alpha": "pdb",
        "beta": "cif",
    }


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_cluster_end_to_end_with_directory(tmp_path):
    """Real foldseek binary processes a directory of mixed PDB + CIF fixtures."""
    (tmp_path / "renin-pdb.pdb").write_text((_FIXTURES / "renin_af3.pdb").read_text())
    (tmp_path / "renin-cif.cif").write_text((_FIXTURES / "renin.cif").read_text())

    output = run_foldseek_cluster(
        FoldseekClusterInput(structures_dir=str(tmp_path)),
        FoldseekClusterConfig(),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.num_structures == 2
    assert output.num_clusters >= 1
