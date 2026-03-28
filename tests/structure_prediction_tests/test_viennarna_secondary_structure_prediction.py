"""tests/structure_prediction_tests/test_viennarna_secondary_structure_prediction.py

Tests for ViennaRNA secondary structure prediction."""

import pytest

from bio_programming_tools.tools.structure_prediction import (
    ViennaRNAConfig,
    ViennaRNAInput,
    ViennaRNAOutput,
    run_viennarna,
)
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("viennarna", gpu=False)

# ── Constants ────────────────────────────────────────────────────────────────
_HAIRPIN = "GCGCUUUUGCGC"
_POLY_A = "AAAAAAAAAA"
_HAIRPIN_2 = "GGGGAAAACCCC"
_HAIRPIN_DNA = "GCGCTTTTGCGC"

# ── Validation tests (no dispatch) ───────────────────────────────────────────


def test_viennarna_input_rejects_empty_list():
    """sequences=[] must be rejected at validation time."""
    with pytest.raises(ValueError, match="At least one sequence is required"):
        ViennaRNAInput(sequences=[])


def test_viennarna_input_rejects_invalid_nucleotides():
    """Sequences containing non-nucleotide characters must be rejected."""
    with pytest.raises(ValueError, match="Invalid nucleotide characters"):
        ViennaRNAInput(sequences=["GCGCXYZGCGC"])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="structure_prediction")
def test_viennarna_basic_folding():
    """Basic RNA folding: a classic hairpin should produce a stem-loop structure."""
    inputs = ViennaRNAInput(sequences=[_HAIRPIN])
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    validate_output(output)

    assert isinstance(output, ViennaRNAOutput)
    assert len(output.results) == 1

    result = output.results[0]
    assert result.sequence == _HAIRPIN
    assert len(result.structure) == len(result.sequence)
    assert result.mfe < 0  # stable structure has negative MFE
    assert "(" in result.structure and ")" in result.structure


@pytest.mark.integration
def test_viennarna_multiple_sequences():
    """Batch folding: poly-A should be unstructured; hairpins should be structured."""
    sequences = [_HAIRPIN, _POLY_A, _HAIRPIN_2]
    inputs = ViennaRNAInput(sequences=sequences)
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    validate_output(output)

    assert len(output.results) == 3

    poly_a_result = output.results[1]
    assert poly_a_result.structure == "." * 10

    assert "(" in output.results[0].structure
    assert "(" in output.results[2].structure


@pytest.mark.integration
def test_viennarna_dna_to_rna_conversion():
    """T in input sequences must be converted to U before folding."""
    inputs_dna = ViennaRNAInput(sequences=[_HAIRPIN_DNA])
    inputs_rna = ViennaRNAInput(sequences=[_HAIRPIN])
    config = ViennaRNAConfig()

    output_dna = run_viennarna(inputs_dna, config)
    validate_output(output_dna)

    output_rna = run_viennarna(inputs_rna, config)
    validate_output(output_rna)

    assert output_dna.results[0].sequence == _HAIRPIN
    assert output_dna.results[0].structure == output_rna.results[0].structure
    assert output_dna.results[0].mfe == pytest.approx(output_rna.results[0].mfe)


@pytest.mark.integration
def test_viennarna_empty_sequence():
    """An empty string in the sequence list should yield a null result entry."""
    inputs = ViennaRNAInput(sequences=[""])
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    validate_output(output, check_export=False)

    assert len(output.results) == 1
    result = output.results[0]
    assert result.sequence == ""
    assert result.structure is None
    assert result.mfe is None


@pytest.mark.integration
def test_viennarna_mixed_empty_and_valid_sequences():
    """Empty sequences interleaved with valid ones must each produce the correct result."""
    inputs = ViennaRNAInput(sequences=[_HAIRPIN, "", _POLY_A])
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    validate_output(output, check_export=False)

    assert len(output.results) == 3

    assert output.results[0].structure is not None
    assert output.results[0].mfe is not None

    assert output.results[1].structure is None
    assert output.results[1].mfe is None

    assert output.results[2].structure == "." * 10
    assert output.results[2].mfe is not None
