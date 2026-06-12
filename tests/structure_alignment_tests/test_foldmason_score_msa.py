"""tests/structure_alignment_tests/test_foldmason_score_msa.py.

Tests for foldmason-score-msa (local-only MSA-LDDT scoring).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.entities.msa import MSA
from proto_tools.tools.structure_alignment import (
    FoldmasonMSAConfig,
    FoldmasonMSAInput,
    FoldmasonScoreMSAConfig,
    FoldmasonScoreMSAInput,
    run_foldmason_msa,
    run_foldmason_score_msa,
)
from proto_tools.tools.structure_alignment.foldmason.standalone.run import _parse_msa2lddt_stdout
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

_TINY_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
_TINY_MSA = ">a\nM\n>b\nM\n"
_FIXTURES = Path(__file__).parent.parent / "dummy_data"


# ── _parse_msa2lddt_stdout ────────────────────────────────────────────────────


def test_parse_stdout_picks_up_all_three_fields():
    """The three foldmason msa2lddt output lines parse into typed fields."""
    stdout = (
        "MMseqs Version: ...\n"
        "Threads: 12\n"
        "\n"
        "Average MSA LDDT: 0.779001\n"
        "Columns considered: 340/340\n"
        "Column scores: 0.800000,0.628148,0.840313\n"
        "Time for processing: 0h 0m 0s 3ms\n"
    )

    result = _parse_msa2lddt_stdout(stdout)

    assert result["average_lddt"] == pytest.approx(0.779001)
    assert result["columns_considered"] == 340
    assert result["alignment_length"] == 340
    assert result["column_scores"] == [0.8, pytest.approx(0.628148), pytest.approx(0.840313)]


def test_parse_stdout_handles_scientific_notation():
    """Scientific-notation scores parse correctly."""
    stdout = "Average MSA LDDT: 1.5e-2\nColumns considered: 5/10\nColumn scores: 1.0e-3,2.0e-3,3.0e-3,4.0e-3,5.0e-3\n"
    result = _parse_msa2lddt_stdout(stdout)
    assert result["average_lddt"] == pytest.approx(0.015)
    assert result["columns_considered"] == 5
    assert result["alignment_length"] == 10
    assert result["column_scores"] == [
        pytest.approx(0.001),
        pytest.approx(0.002),
        pytest.approx(0.003),
        pytest.approx(0.004),
        pytest.approx(0.005),
    ]


def test_parse_stdout_raises_when_lines_missing():
    """A malformed stdout (missing one of the three required lines) is a real upstream regression."""
    with pytest.raises(ValueError, match="missing expected lines"):
        _parse_msa2lddt_stdout("foldmason: nothing useful here\n")


# ── run_foldmason_score_msa (mocked dispatch) ────────────────────────────────


def test_run_foldmason_score_msa_dispatches_msa2lddt():
    """The wrapper sends operation=msa2lddt + structures + msa text; output is parsed scores."""
    inputs = FoldmasonScoreMSAInput(structures=[_TINY_PDB, _TINY_PDB], msa=_TINY_MSA)

    with patch(
        "proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "average_lddt": 0.85,
            "columns_considered": 100,
            "alignment_length": 100,
            "column_scores": [0.9] * 100,
        }
        output = run_foldmason_score_msa(inputs, FoldmasonScoreMSAConfig())

    assert output.success
    assert output.average_lddt == pytest.approx(0.85)
    assert output.columns_considered == 100
    assert output.alignment_length == 100
    assert len(output.column_scores) == 100

    payload = mock_dispatch.call_args.args[1]
    assert payload["operation"] == "msa2lddt"
    assert payload["structures"] == [_TINY_PDB, _TINY_PDB]
    assert payload["structure_ids"] == ["structure_0", "structure_1"]
    # Wire payload carries the round-tripped FASTA string (MSA.to_fasta_string output).
    assert payload["aa_msa_fasta"] == MSA.from_fasta_string(_TINY_MSA).to_fasta_string()
    assert payload["pair_threshold"] == 0.0
    assert payload["only_scoring_cols"] is False
    assert payload["guide_tree_newick"] is None
    assert payload["num_threads"] == 4


def test_run_foldmason_score_msa_accepts_msa_object():
    """`msa` accepts an MSA object directly; wire payload carries the FASTA round-trip."""
    msa_obj = MSA(aligned_sequences=["M", "M"], sequence_ids=["a", "b"])
    inputs = FoldmasonScoreMSAInput(structures=[_TINY_PDB, _TINY_PDB], msa=msa_obj)

    with patch(
        "proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "average_lddt": 0.5,
            "columns_considered": 1,
            "alignment_length": 1,
            "column_scores": [0.5],
        }
        run_foldmason_score_msa(inputs, FoldmasonScoreMSAConfig())

    # Validator kept the MSA as-is; standalone dispatch receives its FASTA serialization.
    assert isinstance(inputs.msa, MSA)
    assert mock_dispatch.call_args.args[1]["aa_msa_fasta"] == msa_obj.to_fasta_string()


def test_run_foldmason_score_msa_accepts_structure_objects_and_paths(tmp_path):
    """`structures` dual-accept: Structure objects and file paths normalize to PDB on the wire."""
    from proto_tools.entities import Structure

    pdb_file = tmp_path / "s.pdb"
    pdb_file.write_text(_TINY_PDB)
    structure_obj = Structure(structure=_TINY_PDB)
    inputs = FoldmasonScoreMSAInput(structures=[structure_obj, pdb_file], msa=_TINY_MSA)
    assert all(type(s).__name__ == "Structure" for s in inputs.structures)

    with patch(
        "proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "average_lddt": 1.0,
            "columns_considered": 1,
            "alignment_length": 1,
            "column_scores": [1.0],
        }
        run_foldmason_score_msa(inputs, FoldmasonScoreMSAConfig())

    assert mock_dispatch.call_args.args[1]["structures"] == [_TINY_PDB, _TINY_PDB]


def test_run_foldmason_score_msa_uses_user_supplied_ids():
    """User-supplied structure_ids are forwarded verbatim to the standalone."""
    inputs = FoldmasonScoreMSAInput(
        structures=[_TINY_PDB, _TINY_PDB],
        structure_ids=["alpha", "beta"],
        msa=">alpha\nM\n>beta\nM\n",
    )
    with patch(
        "proto_tools.tools.structure_alignment.foldmason.foldmason_score_msa.ToolInstance.dispatch"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "average_lddt": 0.9,
            "columns_considered": 1,
            "alignment_length": 1,
            "column_scores": [0.9],
        }
        run_foldmason_score_msa(inputs, FoldmasonScoreMSAConfig())

    assert mock_dispatch.call_args.args[1]["structure_ids"] == ["alpha", "beta"]


# ── Validation tests ─────────────────────────────────────────────────────────


def test_input_rejects_id_count_mismatch():
    """structure_ids must match structures length."""
    with pytest.raises(ValidationError, match="structure_ids length"):
        FoldmasonScoreMSAInput(
            structures=[_TINY_PDB, _TINY_PDB],
            structure_ids=["one"],
            msa=_TINY_MSA,
        )


def test_input_requires_min_two_structures():
    """A single structure can't be scored against an MSA."""
    with pytest.raises(ValidationError, match="at least 2"):
        FoldmasonScoreMSAInput(structures=[_TINY_PDB], msa=_TINY_MSA)


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "/etc/passwd", "..", ".", "path/with/slash", "path\\with\\backslash", ""],
    ids=["dot-dot-prefix", "absolute-path", "dotdot", "dot", "fwd-slash", "back-slash", "empty"],
)
def test_input_rejects_unsafe_structure_ids(bad_id):
    """structure_ids are written as `{id}.pdb` files; reject path-traversal-shaped values."""
    with pytest.raises(ValidationError, match="not a safe filename"):
        FoldmasonScoreMSAInput(
            structures=[_TINY_PDB, _TINY_PDB],
            structure_ids=[bad_id, "ok"],
            msa=_TINY_MSA,
        )


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.skip_ci  # downloads foldmason binary from mmseqs.com; upstream is flaky in GH Actions
def test_foldmason_score_msa_chained_after_local_msa():
    """End-to-end: build an MSA with foldmason-msa local, then score it with foldmason-score-msa."""
    structures = [
        (_FIXTURES / "renin_af3.pdb").read_text(),
        (_FIXTURES / "test_structure_similarity.pdb").read_text(),
        (_FIXTURES / "renin_af3.pdb").read_text(),
    ]
    ids = ["a", "b", "c"]

    msa_out = run_foldmason_msa(
        FoldmasonMSAInput(structures=structures, structure_ids=ids),
        FoldmasonMSAConfig(search_mode="local", num_threads=2),
    )
    assert msa_out.success, f"msa errors: {msa_out.errors}"

    score_out = run_foldmason_score_msa(
        FoldmasonScoreMSAInput(structures=structures, structure_ids=ids, msa=msa_out.aa_msa_fasta),
        FoldmasonScoreMSAConfig(num_threads=2),
    )
    assert score_out.success, f"score errors: {score_out.errors}"
    # Real foldmason score for renin/renin self-pair + a divergent third should land in the 0.5-1.0 range.
    assert 0.0 < score_out.average_lddt <= 1.0
    assert score_out.alignment_length == msa_out.alignment_length
    assert len(score_out.column_scores) == score_out.alignment_length
    for s in score_out.column_scores:
        assert 0.0 <= s <= 1.0


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("foldmason-score-msa")
@pytest.mark.slow
def test_foldmason_score_msa_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark foldmason-score-msa: msa2lddt on a real foldmason-msa alignment of 30 structures (2 folds) (cold + warm)."""
    pdbs = [(_FIXTURES / "renin_af3.pdb").read_text(), (_FIXTURES / "test_structure_similarity.pdb").read_text()]
    n = 30
    structures = [pdbs[i % len(pdbs)] for i in range(n)]
    structure_ids = [f"structure_{i}" for i in range(n)]
    # Score a real alignment of two distinct folds (genuine gaps, sub-1.0 LDDT),
    # built once with foldmason-msa as setup — not a degenerate identical MSA.
    msa_out = run_foldmason_msa(
        FoldmasonMSAInput(structures=structures, structure_ids=structure_ids),
        FoldmasonMSAConfig(search_mode="local", num_threads=4),
    )
    assert msa_out.success, f"msa setup errors: {msa_out.errors}"
    inputs = FoldmasonScoreMSAInput(structures=structures, structure_ids=structure_ids, msa=msa_out.aa_msa_fasta)
    config = FoldmasonScoreMSAConfig(num_threads=4)

    def run_batch():
        last = None
        for _ in range(5):
            last = run_foldmason_score_msa(inputs, config)
        return last

    result = benchmark_twice(request, "foldmason", run_batch)
    validate_output(result)

    assert result.tool_id == "foldmason-score-msa"
    assert result.success, f"errors: {result.errors}"
    assert result.alignment_length == msa_out.alignment_length
    assert len(result.column_scores) == result.alignment_length
    assert 0.0 <= result.average_lddt <= 1.0
