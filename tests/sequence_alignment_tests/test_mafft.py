"""tests/sequence_alignment_tests/test_mafft.py.

Tests for MAFFT tool in proto_tools.tools.sequence_alignment.mafft
"""

import pytest

from proto_tools.tools.sequence_alignment.mafft import (
    MafftConfig,
    MafftInput,
    run_mafft_align,
)
from tests.conftest import benchmark_twice, random_protein_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Input validation tests ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "sequences,error_match",
    [
        (["MVLSPADKTN"], "At least 2 sequences"),
        ([], "At least 2 sequences"),
        ("MVLSPADKTN", "must be a list"),
        (["MVLSPADKTN", 123], "must be strings"),
        (["MVLSPADKTN", ""], "non-empty"),
    ],
)
def test_mafft_input_invalid(sequences, error_match):
    with pytest.raises(ValueError, match=error_match):
        MafftInput(sequences=sequences)


# ── Test data constants ───────────────────────────────────────────────────────

# Protein sequences with internal deletion (3 AA gap: PAD missing)
PROTEIN_WITH_GAP_LONG = "MVLSPADKTNVKAAW"  # 15 AA
PROTEIN_WITH_GAP_SHORT = "MVLSKTNVKAAW"  # 12 AA, missing PAD after MVLS
# Expected alignment:
#   MVLSPADKTNVKAAW
#   MVLS---KTNVKAAW

# Protein sequences with terminal extension
PROTEIN_BASE = "MVLSPADKTNVKAAW"  # 15 AA
PROTEIN_EXTENDED = "MVLSPADKTNVKAAWGGG"  # 18 AA, 3 extra at C-terminus
# Expected alignment (with gap sequence):
#   MVLSPADKTNVKAAW---
#   MVLS---KTNVKAAW---
#   MVLSPADKTNVKAAWGGG

# Protein with flanking gaps (short embedded in long)
PROTEIN_FLANKED_LONG = "AAAAMKLVGAAAABBBBB"  # 18 AA
PROTEIN_FLANKED_SHORT = "MKLVG"  # 5 AA, embedded in the middle
# Expected alignment:
#   AAAAMKLVGAAAABBBBB
#   ----MKLVG---------

# Conservation test sequences
PROTEIN_CONSERVED_A = "MKLVGAARLSSG"
PROTEIN_CONSERVED_B = "AKLVGAARLSSG"  # M->A at position 0
PROTEIN_CONSERVED_C = "MKLVGAARLSSG"  # Same as A
# Column 0: ['M', 'A', 'M'] -> conservation 2/3
# Column 1: ['K', 'K', 'K'] -> conservation 1.0

# DNA sequences with 4bp internal gap
DNA_WITH_GAP_LONG = "ATGCGATCGATCGTGAAA"  # 18 bp
DNA_WITH_GAP_SHORT = "ATGCGATCGTGAAA"  # 14 bp
# Expected alignment (MAFFT lowercases DNA):
#   atgcgatcgatcgtgaaa
#   atg----cgatcgtgaaa


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_mafft_protein_alignment_with_internal_gap():
    """Test alignment producing internal gaps (PAD deletion)."""
    inputs = MafftInput(sequences=[PROTEIN_WITH_GAP_LONG, PROTEIN_WITH_GAP_SHORT])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.num_sequences == 2
    assert result.msa.alignment_length == 15
    # Verify original sequences preserved
    assert result.msa.original_sequences[0] == PROTEIN_WITH_GAP_LONG
    assert result.msa.original_sequences[1] == PROTEIN_WITH_GAP_SHORT
    # Verify exact aligned sequences
    assert result.msa.aligned_sequences[0] == "MVLSPADKTNVKAAW"
    assert result.msa.aligned_sequences[1] == "MVLS---KTNVKAAW"
    # Verify gap statistics
    assert result.msa.total_gaps == 3
    assert result.msa.aligned_sequences[0].count("-") == 0
    assert result.msa.aligned_sequences[1].count("-") == 3


@pytest.mark.integration
def test_mafft_protein_alignment_with_terminal_gaps():
    """Test alignment with terminal extension gaps."""
    inputs = MafftInput(sequences=[PROTEIN_BASE, PROTEIN_WITH_GAP_SHORT, PROTEIN_EXTENDED])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.num_sequences == 3
    assert result.msa.alignment_length == 18
    # Verify exact aligned sequences
    assert result.msa.aligned_sequences[0] == "MVLSPADKTNVKAAW---"
    assert result.msa.aligned_sequences[1] == "MVLS---KTNVKAAW---"
    assert result.msa.aligned_sequences[2] == "MVLSPADKTNVKAAWGGG"
    # Verify gap counts
    assert result.msa.aligned_sequences[0].count("-") == 3
    assert result.msa.aligned_sequences[1].count("-") == 6
    assert result.msa.aligned_sequences[2].count("-") == 0
    assert result.msa.total_gaps == 9


@pytest.mark.integration
def test_mafft_protein_alignment_flanked_short_sequence():
    """Test alignment of short sequence embedded within longer sequence."""
    inputs = MafftInput(sequences=[PROTEIN_FLANKED_LONG, PROTEIN_FLANKED_SHORT])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.num_sequences == 2
    assert result.msa.alignment_length == 18
    # Verify exact aligned sequences (13 gaps total)
    assert result.msa.aligned_sequences[0] == "AAAAMKLVGAAAABBBBB"
    assert result.msa.aligned_sequences[1] == "----MKLVG---------"
    assert result.msa.aligned_sequences[0].count("-") == 0
    assert result.msa.aligned_sequences[1].count("-") == 13
    assert result.msa.total_gaps == 13


@pytest.mark.integration
def test_mafft_dna_alignment_with_internal_gap():
    """Test DNA alignment producing gaps (MAFFT lowercases DNA)."""
    inputs = MafftInput(sequences=[DNA_WITH_GAP_LONG, DNA_WITH_GAP_SHORT])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.num_sequences == 2
    assert result.msa.alignment_length == 18
    # Verify original sequences (computed from aligned sequences)
    assert result.msa.original_sequences[0] == str.lower(DNA_WITH_GAP_LONG)
    assert result.msa.original_sequences[1] == str.lower(DNA_WITH_GAP_SHORT)
    # MAFFT lowercases DNA in output
    assert result.msa.aligned_sequences[0] == "atgcgatcgatcgtgaaa"
    assert result.msa.aligned_sequences[1] == "atg----cgatcgtgaaa"
    # Verify gap statistics
    assert result.msa.total_gaps == 4
    assert result.msa.aligned_sequences[0].count("-") == 0
    assert result.msa.aligned_sequences[1].count("-") == 4


@pytest.mark.integration
@pytest.mark.parametrize("method", ["auto", "localpair", "globalpair", "genafpair"])
def test_mafft_all_alignment_methods_produce_correct_gaps(method):
    """Test all alignment methods produce correct alignment with gaps."""
    inputs = MafftInput(sequences=[PROTEIN_WITH_GAP_LONG, PROTEIN_WITH_GAP_SHORT])
    config = MafftConfig(align_method=method)
    result = run_mafft_align(inputs, config)

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.num_sequences == 2
    assert result.msa.alignment_length == 15
    assert result.metadata["align_method"] == method
    # All methods should produce the same optimal alignment
    assert result.msa.aligned_sequences[0] == "MVLSPADKTNVKAAW"
    assert result.msa.aligned_sequences[1] == "MVLS---KTNVKAAW"
    assert result.msa.total_gaps == 3


@pytest.mark.integration
def test_mafft_config_options_passed_to_mafft():
    """Test configuration options are correctly passed and recorded."""
    inputs = MafftInput(sequences=[PROTEIN_WITH_GAP_LONG, PROTEIN_WITH_GAP_SHORT])
    config = MafftConfig(align_method="localpair", threads=2, max_iterations=100)
    result = run_mafft_align(inputs, config)

    # Validate output and export functionality
    validate_output(result)
    assert result.metadata["align_method"] == "localpair"
    assert result.metadata["threads"] == 2
    assert result.metadata["max_iterations"] == 100
    assert result.metadata["num_sequences"] == 2


@pytest.mark.integration
def test_mafft_conservation_scores():
    """Test conservation score calculation with known values."""
    inputs = MafftInput(sequences=[PROTEIN_CONSERVED_A, PROTEIN_CONSERVED_B, PROTEIN_CONSERVED_C])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.alignment_length == 12
    # Column 0: M, A, M -> conservation = 2/3
    assert result.msa.get_column(0) == ["M", "A", "M"]
    assert result.msa.get_conservation(0) == pytest.approx(2 / 3)
    # Column 1: K, K, K -> conservation = 1.0
    assert result.msa.get_column(1) == ["K", "K", "K"]
    assert result.msa.get_conservation(1) == 1.0
    # All other columns are identical -> conservation = 1.0
    for i in range(2, 12):
        assert result.msa.get_conservation(i) == 1.0


@pytest.mark.integration
def test_mafft_to_fasta_output_format():
    """Test FASTA format output is correct."""
    inputs = MafftInput(sequences=[PROTEIN_CONSERVED_A, PROTEIN_CONSERVED_B])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)

    fasta = result.msa.to_fasta_string()
    expected = ">seq_0\nMKLVGAARLSSG\n>seq_1\nAKLVGAARLSSG\n"
    assert fasta == expected


@pytest.mark.integration
def test_mafft_custom_sequence_ids_preserved():
    """Test that custom sequence IDs are preserved in output."""
    inputs = MafftInput(
        sequences=[PROTEIN_CONSERVED_A, PROTEIN_CONSERVED_B],
        sequence_ids=["alpha", "beta"],
    )
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.sequence_ids == ["alpha", "beta"]
    # Verify FASTA output uses custom IDs
    fasta = result.msa.to_fasta_string()
    assert ">alpha\n" in fasta
    assert ">beta\n" in fasta


@pytest.mark.integration
def test_mafft_default_sequence_ids_when_not_provided():
    """Test that default IDs are generated when not provided."""
    inputs = MafftInput(sequences=[PROTEIN_CONSERVED_A, PROTEIN_CONSERVED_B])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)
    assert result.msa.sequence_ids == ["seq_0", "seq_1"]


@pytest.mark.integration
def test_mafft_sequence_ids_length_mismatch_fails():
    """Test that mismatched ID count raises an error."""
    inputs = MafftInput(
        sequences=["MVLS", "AVLS"],
        sequence_ids=["only_one"],
    )
    with pytest.raises(Exception, match="ids length"):
        run_mafft_align(inputs, MafftConfig())


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("mafft-align")
@pytest.mark.slow
def test_mafft_align_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark mafft-align: 50 random 150-aa protein sequences, default 'auto' method (cold + warm)."""
    sequences = random_protein_sequences(n=50, length=150, seed=0)
    inputs = MafftInput(sequences=sequences)
    config = MafftConfig(threads=4)

    result = benchmark_twice(request, "mafft", lambda: run_mafft_align(inputs, config))
    validate_output(result)

    assert result.tool_id == "mafft-align"
    assert result.msa.num_sequences == 50
