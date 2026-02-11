"""
test_viennarna.py

Unit tests for ViennaRNA secondary structure prediction.
"""

import pytest

from bio_programming_tools.tools.structure_prediction import (
    ViennaRNAConfig,
    ViennaRNAInput,
    ViennaRNAOutput,
    run_viennarna,
)
from tests.tool_infra_tests.test_export_functionality import validate_output


def test_basic_folding():
    """Test basic RNA folding with a simple hairpin sequence."""
    # Classic hairpin: GCGC...GCGC should form a stem-loop
    inputs = ViennaRNAInput(sequences=["GCGCUUUUGCGC"])
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    # Validate output and export functionality
    validate_output(output)

    assert isinstance(output, ViennaRNAOutput)
    assert len(output.results) == 1

    result = output.results[0]
    assert result.sequence == "GCGCUUUUGCGC"
    assert len(result.structure) == len(result.sequence)
    assert result.mfe < 0  # Should have negative (stable) MFE
    # Structure should contain paired regions
    assert "(" in result.structure and ")" in result.structure


def test_multiple_sequences():
    """Test folding multiple sequences in one call."""
    sequences = [
        "GCGCUUUUGCGC",  # Hairpin
        "AAAAAAAAAA",    # Poly-A (should be unstructured)
        "GGGGAAAACCCC",  # Another hairpin
    ]
    inputs = ViennaRNAInput(sequences=sequences)
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    # Validate output and export functionality
    validate_output(output)

    assert len(output.results) == 3

    # Poly-A should have no structure (all dots)
    poly_a_result = output.results[1]
    assert poly_a_result.structure == "." * 10

    # Hairpins should have structure
    assert "(" in output.results[0].structure
    assert "(" in output.results[2].structure


def test_invalid_sequence():
    """Test that invalid sequences raise ValueError."""
    with pytest.raises(ValueError, match="Invalid nucleotide"):
        ViennaRNAInput(sequences=["GCGCXYZGCGC"])


def test_dna_to_rna_conversion():
    """Test that T is converted to U for RNA folding."""
    # Same sequence with T vs U should give same result
    inputs_dna = ViennaRNAInput(sequences=["GCGCTTTTGCGC"])
    inputs_rna = ViennaRNAInput(sequences=["GCGCUUUUGCGC"])
    config = ViennaRNAConfig()

    output_dna = run_viennarna(inputs_dna, config)

    # Validate output and export functionality
    validate_output(output_dna)

    output_rna = run_viennarna(inputs_rna, config)

    # Validate output and export functionality
    validate_output(output_rna)

    # After conversion, sequences should match
    assert output_dna.results[0].sequence == "GCGCUUUUGCGC"
    assert output_dna.results[0].structure == output_rna.results[0].structure
    assert output_dna.results[0].mfe == pytest.approx(output_rna.results[0].mfe)


def test_empty_sequence():
    """Test that empty sequences are handled gracefully."""
    inputs = ViennaRNAInput(sequences=[""])
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    # Validate output and export functionality
    validate_output(output)

    assert len(output.results) == 1
    result = output.results[0]
    assert result.sequence == ""
    assert result.structure is None
    assert result.mfe is None


def test_mixed_empty_and_valid_sequences():
    """Test handling of empty sequences mixed with valid ones."""
    inputs = ViennaRNAInput(sequences=["GCGCUUUUGCGC", "", "AAAAAAAAAA"])
    config = ViennaRNAConfig()

    output = run_viennarna(inputs, config)

    # Validate output and export functionality
    validate_output(output)

    assert len(output.results) == 3

    # First sequence should fold normally
    assert output.results[0].structure is not None
    assert output.results[0].mfe is not None

    # Empty sequence
    assert output.results[1].structure is None
    assert output.results[1].mfe is None

    # Third sequence should fold normally
    assert output.results[2].structure == "." * 10
    assert output.results[2].mfe is not None
