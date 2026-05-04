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


def test_input_requires_one_of_structures_or_dir():
    with pytest.raises(ValidationError, match="exactly one"):
        FoldseekMultimerClusterInput()


def test_input_rejects_both_structures_and_dir(tmp_path):
    (tmp_path / "a.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "b.pdb").write_text(_TINY_MULTIMER_PDB)
    with pytest.raises(ValidationError, match="exactly one"):
        FoldseekMultimerClusterInput(
            structures=[_TINY_MULTIMER_PDB, _TINY_MULTIMER_PDB],
            structures_dir=str(tmp_path),
        )


def test_input_rejects_ids_with_dir(tmp_path):
    (tmp_path / "a.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "b.pdb").write_text(_TINY_MULTIMER_PDB)
    with pytest.raises(ValidationError, match="may not be combined"):
        FoldseekMultimerClusterInput(structures_dir=str(tmp_path), structure_ids=["x", "y"])


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
        FoldseekMultimerClusterInput(structures_dir=str(tmp_path))


def test_input_rejects_nonexistent_dir():
    with pytest.raises(ValidationError, match="not an existing directory"):
        FoldseekMultimerClusterInput(structures_dir="/nonexistent/path/abcxyz")


def test_input_rejects_dir_with_too_few_files(tmp_path):
    (tmp_path / "only.pdb").write_text(_TINY_MULTIMER_PDB)
    with pytest.raises(ValidationError, match="at least 2"):
        FoldseekMultimerClusterInput(structures_dir=str(tmp_path))


# ── FoldseekMultimerClusterInput resolution from directory ────────────────────


def test_input_resolves_dir_with_mixed_formats(tmp_path):
    """Mixed .pdb + .cif: both read into structures, stems become structure_ids, structures_dir cleared."""
    (tmp_path / "alpha.pdb").write_text(_TINY_MULTIMER_PDB)
    (tmp_path / "beta.cif").write_text(_TINY_MULTIMER_CIF)

    inputs = FoldseekMultimerClusterInput(structures_dir=str(tmp_path))

    assert inputs.structures_dir is None
    assert sorted(inputs.structure_ids or []) == ["alpha", "beta"]
    assert _TINY_MULTIMER_PDB in (inputs.structures or [])
    assert _TINY_MULTIMER_CIF in (inputs.structures or [])


def test_input_resolves_dir_decompresses_gz_files(tmp_path):
    """`.pdb.gz` files are decompressed during read."""
    with gzip.open(tmp_path / "alpha.pdb.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_MULTIMER_PDB)
    with gzip.open(tmp_path / "beta.pdb.gz", "wt", encoding="utf-8") as f:
        f.write(_TINY_MULTIMER_PDB)

    inputs = FoldseekMultimerClusterInput(structures_dir=str(tmp_path))

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
        FoldseekMultimerClusterInput(structures_dir=str(tmp_path)),
        FoldseekMultimerClusterConfig(num_threads=2),
    )

    assert output.success, f"errors: {output.errors}"
    assert output.num_multimers == 2
