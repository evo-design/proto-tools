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
from proto_tools.tools.structure_alignment.foldseek.foldseek_cluster import (
    _coerce_structure_items_to_text,
    _parse_cluster_tsv,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_TINY_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
_TINY_CIF = "data_TEST\nloop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n"
_TINY_FASTA = ">protein_a\nMKTLLEVAEK\n"
_TINY_FASTA_2 = ">protein_b\nMQTLLDVAEH\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── _coerce_structure_items_to_text ──────────────────────────────────────────


def test_coerce_passes_str_items_through_unchanged():
    """Plain str items pass through; per-string format detection happens downstream."""
    data = {"structures": [_TINY_PDB, _TINY_FASTA]}
    out = _coerce_structure_items_to_text(data)
    assert out["structures"] == [_TINY_PDB, _TINY_FASTA]


def test_coerce_serializes_structure_objects_to_pdb():
    """Structure inputs are serialized to PDB string via .structure_pdb."""
    from proto_tools.entities import Structure

    s = Structure(structure=_TINY_PDB)
    out = _coerce_structure_items_to_text({"structures": [s, _TINY_PDB]})
    assert out["structures"] == [_TINY_PDB, _TINY_PDB]


def test_coerce_structure_path_loads_and_serializes_to_pdb(tmp_path):
    """Path with a structure extension loads via Structure.from_file and emits PDB."""
    p = tmp_path / "x.pdb"
    p.write_text(_TINY_PDB)
    out = _coerce_structure_items_to_text({"structures": [p]})
    assert out["structures"] == [_TINY_PDB]


def test_coerce_fasta_path_reads_text_verbatim(tmp_path):
    """Path with a FASTA extension is read as text; no Structure parsing (which would reject FASTA)."""
    p = tmp_path / "seqs.fasta"
    p.write_text(_TINY_FASTA)
    out = _coerce_structure_items_to_text({"structures": [p]})
    assert out["structures"] == [_TINY_FASTA]


def test_coerce_gzipped_fasta_path_reads_decompressed_text(tmp_path):
    """Gzipped FASTA path is decompressed and read as text."""
    p = tmp_path / "seqs.fasta.gz"
    with gzip.open(p, "wt") as f:
        f.write(_TINY_FASTA)
    out = _coerce_structure_items_to_text({"structures": [p]})
    assert out["structures"] == [_TINY_FASTA]


def test_coerce_noop_when_structures_absent_or_not_list():
    """Helper short-circuits when `structures` is missing or a non-list (e.g. a directory path)."""
    assert _coerce_structure_items_to_text({"structures": "some/dir"}) == {"structures": "some/dir"}
    assert _coerce_structure_items_to_text({"structures": None}) == {"structures": None}
    # Non-dict input passes through unchanged.
    assert _coerce_structure_items_to_text("not a dict") == "not a dict"


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


def test_input_requires_structures():
    with pytest.raises(ValidationError, match="required"):
        FoldseekClusterInput()


def test_input_accepts_structure_objects():
    """`Structure` objects in the list are accepted (typed-entity input) and coerced to text."""
    from proto_tools.entities import Structure

    s = Structure(structure=_TINY_PDB)
    inputs = FoldseekClusterInput(structures=[s, s])
    assert inputs.structures == [_TINY_PDB, _TINY_PDB]
    # Schema exposes the Structure entity (str carries FASTA).
    assert "Structure" in FoldseekClusterInput.model_json_schema().get("$defs", {})


def test_input_rejects_ids_with_dir(tmp_path):
    (tmp_path / "a.pdb").write_text(_TINY_PDB)
    (tmp_path / "b.pdb").write_text(_TINY_PDB)
    with pytest.raises(ValidationError, match="may not be combined"):
        FoldseekClusterInput(structures=str(tmp_path), structure_ids=["x", "y"])


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
        FoldseekClusterInput(structures="/nonexistent/path/abcxyz")


def test_input_rejects_dir_with_too_few_files(tmp_path):
    (tmp_path / "only.pdb").write_text(_TINY_PDB)
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekClusterInput(structures=str(tmp_path))


def test_input_rejects_duplicate_user_supplied_ids():
    """Duplicate IDs would clobber each other in foldseek's TSV output."""
    with pytest.raises(ValidationError, match="duplicated"):
        FoldseekClusterInput(structures=[_TINY_PDB, _TINY_PDB], structure_ids=["a", "a"])


def test_input_rejects_dir_with_colliding_stems(tmp_path):
    """`protein.pdb` + `protein.cif` produce identical stems and would collide silently."""
    (tmp_path / "protein.pdb").write_text(_TINY_PDB)
    (tmp_path / "protein.cif").write_text(_TINY_CIF)
    with pytest.raises(ValidationError, match="duplicated"):
        FoldseekClusterInput(structures=str(tmp_path))


def test_input_rejects_mixed_pdb_and_fasta():
    """A single call must be all-FASTA or all-structure."""
    with pytest.raises(ValidationError, match="Cannot mix FASTA"):
        FoldseekClusterInput(structures=[_TINY_FASTA, _TINY_PDB])


# ── FoldseekClusterInput resolution from directory ────────────────────────────


def test_input_resolves_dir_with_mixed_formats(tmp_path):
    """Mixed .pdb + .cif: both read into structures, stems become structure_ids."""
    (tmp_path / "alpha.pdb").write_text(_TINY_PDB)
    (tmp_path / "beta.cif").write_text(_TINY_CIF)

    inputs = FoldseekClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_PDB in (inputs.structures or [])
    assert _TINY_CIF in (inputs.structures or [])


def test_input_resolves_dir_decompresses_gz_files(tmp_path):
    """`.pdb.gz` / `.cif.gz` are decompressed during read."""
    with gzip.open(tmp_path / "alpha.pdb.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_PDB)
    with gzip.open(tmp_path / "beta.cif.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_CIF)

    inputs = FoldseekClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_PDB in (inputs.structures or [])
    assert _TINY_CIF in (inputs.structures or [])


def test_input_dir_skips_unsupported_files(tmp_path):
    """Files without supported extensions (e.g. README.md) are filtered out."""
    (tmp_path / "alpha.pdb").write_text(_TINY_PDB)
    (tmp_path / "beta.pdb").write_text(_TINY_PDB)
    (tmp_path / "README.md").write_text("just notes")
    (tmp_path / ".DS_Store").write_text("junk")

    inputs = FoldseekClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]


def test_input_resolves_dir_with_fasta_files(tmp_path):
    """`.fasta` / `.fa` files are enumerated by the cluster dir resolver."""
    (tmp_path / "alpha.fasta").write_text(_TINY_FASTA)
    (tmp_path / "beta.fa").write_text(_TINY_FASTA_2)

    inputs = FoldseekClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_FASTA in (inputs.structures or [])
    assert _TINY_FASTA_2 in (inputs.structures or [])


def test_input_resolves_dir_from_path_object(tmp_path):
    """A `Path` directory (not just a str) is accepted by the merged `structures` field."""
    (tmp_path / "alpha.pdb").write_text(_TINY_PDB)
    (tmp_path / "beta.cif").write_text(_TINY_CIF)

    inputs = FoldseekClusterInput(structures=tmp_path)

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_PDB in (inputs.structures or [])


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
    inputs = FoldseekClusterInput(structures=str(tmp_path))

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


def test_run_foldseek_cluster_with_fasta_inputs():
    """FASTA inputs propagate as `structure_formats=['fasta', ...]` and carry prostt5_weights_dir."""
    inputs = FoldseekClusterInput(structures=[_TINY_FASTA, _TINY_FASTA_2])

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_cluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"clusters_tsv": ""}
        run_foldseek_cluster(
            inputs,
            FoldseekClusterConfig(prostt5_weights_dir="/path/to/prostt5"),
        )

    payload = mock_dispatch.call_args.args[1]
    assert payload["structure_formats"] == ["fasta", "fasta"]
    assert payload["prostt5_weights_dir"] == "/path/to/prostt5"


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_cluster_end_to_end_with_directory(tmp_path):
    """Real foldseek binary processes a directory of mixed PDB + CIF fixtures."""
    (tmp_path / "renin-pdb.pdb").write_text((_FIXTURES / "renin_af3.pdb").read_text())
    (tmp_path / "renin-cif.cif").write_text((_FIXTURES / "renin.cif").read_text())

    output = run_foldseek_cluster(
        FoldseekClusterInput(structures=str(tmp_path)),
        FoldseekClusterConfig(),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.num_structures == 2
    assert output.num_clusters >= 1


# ── Benchmark ───────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("foldseek-cluster")
@pytest.mark.slow
def test_foldseek_cluster_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark foldseek-cluster: 3 easy-cluster runs over 90 structures (3 distinct folds cycled) (cold + warm)."""
    pdbs = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
        (_FIXTURES / "pdl1.pdb").read_text(),
    ]
    n = 90
    structures = [pdbs[i % len(pdbs)] for i in range(n)]
    structure_ids = [f"struct_{i}" for i in range(n)]
    inputs = FoldseekClusterInput(structures=structures, structure_ids=structure_ids)
    config = FoldseekClusterConfig(num_threads=8)

    def run_batch():
        last = None
        for _ in range(3):
            last = run_foldseek_cluster(inputs, config)
        return last

    result = benchmark_twice(request, "foldseek", run_batch)
    validate_output(result)

    assert result.tool_id == "foldseek-cluster"
    assert result.num_structures == n
    assert result.num_clusters >= 2
    assert len(result.clusters) == result.num_clusters
