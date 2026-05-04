"""tests/structure_alignment_tests/test_foldmason_score_msa.py.

Tests for foldmason-score-msa (local-only MSA-LDDT scoring).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from proto_tools.tools.structure_alignment import (
    FoldmasonMSAConfig,
    FoldmasonMSAInput,
    FoldmasonScoreMSAConfig,
    FoldmasonScoreMSAInput,
    run_foldmason_msa,
    run_foldmason_score_msa,
)
from proto_tools.tools.structure_alignment.foldmason.standalone.run import _parse_msa2lddt_stdout

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
    inputs = FoldmasonScoreMSAInput(structures=[_TINY_PDB, _TINY_PDB], aa_msa_fasta=_TINY_MSA)

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
    assert payload["aa_msa_fasta"] == _TINY_MSA
    assert payload["pair_threshold"] == 0.0
    assert payload["only_scoring_cols"] is False
    assert payload["guide_tree_newick"] is None
    assert payload["num_threads"] == 4


def test_run_foldmason_score_msa_uses_user_supplied_ids():
    """User-supplied structure_ids are forwarded verbatim to the standalone."""
    inputs = FoldmasonScoreMSAInput(
        structures=[_TINY_PDB, _TINY_PDB],
        structure_ids=["alpha", "beta"],
        aa_msa_fasta=">alpha\nM\n>beta\nM\n",
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
            aa_msa_fasta=_TINY_MSA,
        )


def test_input_requires_min_two_structures():
    """A single structure can't be scored against an MSA."""
    with pytest.raises(ValidationError, match="at least 2"):
        FoldmasonScoreMSAInput(structures=[_TINY_PDB], aa_msa_fasta=_TINY_MSA)


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
            aa_msa_fasta=_TINY_MSA,
        )


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
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
        FoldmasonScoreMSAInput(structures=structures, structure_ids=ids, aa_msa_fasta=msa_out.aa_msa_fasta),
        FoldmasonScoreMSAConfig(num_threads=2),
    )
    assert score_out.success, f"score errors: {score_out.errors}"
    # Real foldmason score for renin/renin self-pair + a divergent third should land in the 0.5-1.0 range.
    assert 0.0 < score_out.average_lddt <= 1.0
    assert score_out.alignment_length == msa_out.alignment_length
    assert len(score_out.column_scores) == score_out.alignment_length
    for s in score_out.column_scores:
        assert 0.0 <= s <= 1.0
