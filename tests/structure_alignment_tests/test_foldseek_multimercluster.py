"""Tests for foldseek-multimercluster (local-only multimer structural clustering)."""

import gzip
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldseekMultimerClusterConfig,
    FoldseekMultimerClusterInput,
    run_foldseek_multimercluster,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_TINY_MULTIMER_PDB = (
    "ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00  0.00\n"
    "ATOM      2  CA  MET B   1      10.000  10.000  10.000  1.00  0.00\n"
)
_TINY_MULTIMER_CIF = "data_TEST\nloop_\n_atom_site.id\n_atom_site.label_asym_id\n1 A\n2 B\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── FoldseekMultimerClusterInput validators ───────────────────────────────────


def test_input_requires_at_least_two_structures():
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekMultimerClusterInput(structures=[_TINY_MULTIMER_PDB])


def test_input_requires_structures():
    with pytest.raises(ValidationError, match="required"):
        FoldseekMultimerClusterInput()


def test_input_accepts_structure_objects():
    """`Structure` objects in the list are accepted (typed-entity input) and coerced to text."""
    from proto_tools.entities import Structure

    s = Structure(structure=_TINY_MULTIMER_PDB)
    inputs = FoldseekMultimerClusterInput(structures=[s, s])
    assert inputs.structures == [_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB]
    # Schema exposes the Structure entity.
    assert "Structure" in FoldseekMultimerClusterInput.model_json_schema().get("$defs", {})


def test_input_rejects_ids_with_dir(tmp_path):
    (tmp_path / "a.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "b.pdb").write_text(_TINY_MULTIMER_PDB)
    with pytest.raises(ValidationError, match="may not be combined"):
        FoldseekMultimerClusterInput(structures=str(tmp_path), structure_ids=["x", "y"])


def test_input_rejects_id_count_mismatch():
    with pytest.raises(ValidationError, match="structure_ids length"):
        FoldseekMultimerClusterInput(
            structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
            structure_ids=["only-one"],
        )


def test_input_rejects_user_supplied_ids_with_underscore():
    """`_` collides with Foldseek's `{multimer_id}_{chain}` schema."""
    with pytest.raises(ValidationError, match="collides"):
        FoldseekMultimerClusterInput(
            structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
            structure_ids=["my_protein", "ok"],
        )


def test_input_rejects_filename_stems_with_underscore(tmp_path):
    """Filename-derived IDs containing `_` are rejected to prevent silent ID mangling."""
    (tmp_path / "complex_a.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "complex_b.pdb").write_text(_TINY_MULTIMER_PDB)
    with pytest.raises(ValidationError, match="collides"):
        FoldseekMultimerClusterInput(structures=str(tmp_path))


def test_input_rejects_nonexistent_dir():
    with pytest.raises(ValidationError, match="not an existing directory"):
        FoldseekMultimerClusterInput(structures="/nonexistent/path/abcxyz")


def test_input_rejects_dir_with_too_few_files(tmp_path):
    (tmp_path / "only.pdb").write_text(_TINY_MULTIMER_PDB)
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekMultimerClusterInput(structures=str(tmp_path))


# ── FoldseekMultimerClusterInput resolution from directory ────────────────────


def test_input_resolves_dir_with_mixed_formats(tmp_path):
    """Mixed .pdb + .cif: both read into structures, stems become structure_ids."""
    (tmp_path / "alpha.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "beta.cif").write_text(_TINY_MULTIMER_CIF)

    inputs = FoldseekMultimerClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_MULTIMER_PDB in (inputs.structures or [])
    assert _TINY_MULTIMER_CIF in (inputs.structures or [])


def test_input_dir_skips_fasta_files(tmp_path):
    """Multimer dir enumerator ignores FASTA — multimer doesn't support FASTA inputs at all."""
    (tmp_path / "alpha.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "beta.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "ignored.fasta").write_text(">seq\nMKTL\n")

    inputs = FoldseekMultimerClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]


def test_input_rejects_fasta_text():
    """Direct FASTA text is rejected at validation, not at runtime in the wrapper."""
    with pytest.raises(ValidationError, match="FASTA input is not supported"):
        FoldseekMultimerClusterInput(structures=[">s1\nMKTL\n", ">s2\nMQTL\n"])


def test_input_resolves_dir_decompresses_gz_files(tmp_path):
    """`.pdb.gz` files are decompressed during read."""
    with gzip.open(tmp_path / "alpha.pdb.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_MULTIMER_PDB)
    with gzip.open(tmp_path / "beta.pdb.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_MULTIMER_PDB)

    inputs = FoldseekMultimerClusterInput(structures=str(tmp_path))

    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]


# ── run_foldseek_multimercluster (mocked dispatch) ────────────────────────────


def test_run_foldseek_multimercluster_dispatches_easy_multimercluster():
    """Wrapper sends operation=easy_multimercluster, default IDs (no `_`), per-structure formats, and thresholds."""
    inputs = FoldseekMultimerClusterInput(structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_CIF, _TINY_MULTIMER_PDB])

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_multimercluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "clusters_tsv": "multimer-0\tmultimer-0\nmultimer-0\tmultimer-1\nmultimer-2\tmultimer-2\n",
            "rep_seq_fasta": ">multimer-0_A\nMSEQ\n",
        }
        output = run_foldseek_multimercluster(inputs, FoldseekMultimerClusterConfig())

    assert output.success
    assert output.num_multimers == 3

    payload = mock_dispatch.call_args.args[1]
    assert payload["operation"] == "easy_multimercluster"
    assert payload["structure_ids"] == ["multimer-0", "multimer-1", "multimer-2"]
    assert payload["structure_formats"] == ["pdb", "cif", "pdb"]
    # Defaults from MultimerCluster::setMultimerClusterDefaults — verified against foldseek source.
    assert payload["multimer_tm_threshold"] == 0.65
    assert payload["chain_tm_threshold"] == 0.001
    assert payload["interface_lddt_threshold"] == 0.5


def test_run_foldseek_multimercluster_uses_user_supplied_ids():
    inputs = FoldseekMultimerClusterInput(
        structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
        structure_ids=["complex-alpha", "complex-beta"],
    )

    with patch(
        "proto_tools.tools.structure_alignment.foldseek.foldseek_multimercluster.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {"clusters_tsv": "", "rep_seq_fasta": ""}
        run_foldseek_multimercluster(inputs, FoldseekMultimerClusterConfig())

    assert mock_dispatch.call_args.args[1]["structure_ids"] == ["complex-alpha", "complex-beta"]


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_foldseek_multimercluster_end_to_end_with_directory(tmp_path):
    """Real foldseek binary processes a directory of mixed multi-chain PDB + CIF fixtures."""
    (tmp_path / "pdl1.pdb").write_text((_FIXTURES / "pdl1.pdb").read_text())
    (tmp_path / "renin.cif").write_text((_FIXTURES / "renin.cif").read_text())

    output = run_foldseek_multimercluster(
        FoldseekMultimerClusterInput(structures=str(tmp_path)),
        FoldseekMultimerClusterConfig(num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.num_multimers == 2


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("foldseek-multimercluster")
@pytest.mark.slow
def test_foldseek_multimercluster_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark foldseek-multimercluster: cluster 90 multi-chain complexes (45 pdl1 + 45 renin) in one easy-multimercluster run (cold + warm)."""
    pdl1_pdb = (_FIXTURES / "pdl1.pdb").read_text()
    renin_cif = (_FIXTURES / "renin.cif").read_text()

    structures = []
    structure_ids = []
    for i in range(45):
        structures.append(pdl1_pdb)
        structure_ids.append(f"pdl1-{i}")
        structures.append(renin_cif)
        structure_ids.append(f"renin-{i}")

    inputs = FoldseekMultimerClusterInput(structures=structures, structure_ids=structure_ids)
    config = FoldseekMultimerClusterConfig(num_threads=4)

    result = benchmark_twice(
        request,
        "foldseek",
        lambda: run_foldseek_multimercluster(inputs, config),
    )

    validate_output(result)
    assert result.tool_id == "foldseek-multimercluster"
    assert result.num_multimers == 90
    assert result.num_clusters == len(result.clusters)
    assert result.num_clusters >= 2
