"""Tests for MAFFT tool in bio_programming_tools.tools.sequence_alignment.mafft"""

import pytest

from bio_programming_tools.tools.sequence_alignment.mafft import (
    MafftConfig,
    MafftInput,
    MafftOutput,
    run_mafft_align,
)
from bio_programming_tools.tools.sequence_alignment.msas import MSA
from tests.tool_infra_tests.test_export_functionality import validate_output


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def two_seq_output():
    return MafftOutput(
        msa=MSA(
            aligned_sequences_or_filepath=["MVLS", "AVLS"],
            sequence_ids=["seq_0", "seq_1"],
        ),
        metadata={},
    )


@pytest.fixture
def gapped_output():
    return MafftOutput(
        msa=MSA(
            aligned_sequences_or_filepath=["MV--LS", "MVLLS-"],
            sequence_ids=["seq_0", "seq_1"],
        ),
        metadata={},
    )


# ── MafftOutput tests ────────────────────────────────────────────────────────

def test_mafft_output_basic_properties(two_seq_output):
    assert two_seq_output.msa.num_sequences == 2
    assert two_seq_output.msa.alignment_length == 4
    assert two_seq_output.msa.total_gaps == 0
    assert two_seq_output.msa.average_gap_fraction == 0.0


def test_mafft_output_sequence_ids_and_originals(two_seq_output):
    assert two_seq_output.msa.sequence_ids == ["seq_0", "seq_1"]
    assert two_seq_output.msa.original_sequences == ["MVLS", "AVLS"]


def test_mafft_output_gap_statistics(gapped_output):
    assert gapped_output.msa.total_gaps == 3
    assert gapped_output.msa.average_gap_fraction == pytest.approx((2 / 6 + 1 / 6) / 2)


def test_mafft_output_get_column(two_seq_output):
    assert two_seq_output.msa.get_column(0) == ["M", "A"]
    assert two_seq_output.msa.get_column(1) == ["V", "V"]


def test_mafft_output_get_column_out_of_range(two_seq_output):
    with pytest.raises(IndexError):
        two_seq_output.msa.get_column(4)
    with pytest.raises(IndexError):
        two_seq_output.msa.get_column(-1)


def test_mafft_output_get_conservation():
    output = MafftOutput(
        msa=MSA(
            aligned_sequences_or_filepath=["MVLS", "MVLS", "AVLS"],
            sequence_ids=["seq_0", "seq_1", "seq_2"],
        ),
        metadata={},
    )
    assert output.msa.get_conservation(0) == pytest.approx(2 / 3)  # 2 M, 1 A
    assert output.msa.get_conservation(1) == 1.0  # all V


def test_mafft_output_get_conservation_all_gaps():
    output = MafftOutput(
        msa=MSA(
            aligned_sequences_or_filepath=["MVL-", "MVL-"],
            sequence_ids=["seq_0", "seq_1"],
        ),
        metadata={},
    )
    assert output.msa.get_conservation(3) == 0.0


def test_mafft_output_to_fasta(two_seq_output):
    fasta = two_seq_output.msa.to_fasta_string()
    assert fasta == ">seq_0\nMVLS\n>seq_1\nAVLS"


def test_mafft_output_msa_directly_accessible(two_seq_output):
    """Test that MSA is directly accessible without conversion."""
    assert isinstance(two_seq_output.msa, MSA)
    assert two_seq_output.msa.num_sequences == 2
    assert two_seq_output.msa.alignment_length == 4
    assert two_seq_output.msa.aligned_sequences == ["MVLS", "AVLS"]
    assert two_seq_output.msa.get_column(0) == ["M", "A"]
    assert two_seq_output.msa.get_conservation(0) == pytest.approx(0.5)


# ── Input validation tests ───────────────────────────────────────────────────

def test_mafft_input_valid():
    inputs = MafftInput(sequences=["MVLSPADKTN", "MKLLVVAAAA"])
    assert len(inputs.sequences) == 2


def test_mafft_input_valid_with_custom_ids():
    inputs = MafftInput(
        sequences=["MVLSPADKTN", "MKLLVVAAAA"],
        sequence_ids=["protein_a", "protein_b"],
    )
    assert inputs.sequence_ids == ["protein_a", "protein_b"]


def test_mafft_input_valid_without_ids():
    inputs = MafftInput(sequences=["MVLSPADKTN", "MKLLVVAAAA"])
    assert inputs.sequence_ids is None


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


# ── Config validation tests ──────────────────────────────────────────────────

def test_mafft_config_defaults():
    config = MafftConfig()
    assert config.align_method == "auto"
    assert config.max_iterations == 0
    assert config.threads == 1


@pytest.mark.parametrize("method", ["auto", "localpair", "globalpair", "genafpair"])
def test_mafft_config_valid_align_methods(method):
    config = MafftConfig(align_method=method)
    assert config.align_method == method


def test_mafft_config_invalid_align_method():
    with pytest.raises(ValueError, match="Input should be"):
        MafftConfig(align_method="invalid")


def test_mafft_config_invalid_iterations():
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MafftConfig(max_iterations=-1)


def test_mafft_config_invalid_threads():
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        MafftConfig(threads=0)


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
@pytest.mark.include_in_env_report
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
    inputs = MafftInput(
        sequences=[PROTEIN_CONSERVED_A, PROTEIN_CONSERVED_B, PROTEIN_CONSERVED_C]
    )
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
    expected = ">seq_0\nMKLVGAARLSSG\n>seq_1\nAKLVGAARLSSG"
    assert fasta == expected


@pytest.mark.integration
def test_mafft_gap_statistics_accuracy():
    """Test gap fraction and average gap statistics are accurate."""
    inputs = MafftInput(sequences=[PROTEIN_BASE, PROTEIN_WITH_GAP_SHORT, PROTEIN_EXTENDED])
    result = run_mafft_align(inputs, MafftConfig())

    # Validate output and export functionality
    validate_output(result)

    # Alignment length is 18
    # seq_0: 3 gaps, seq_1: 6 gaps, seq_2: 0 gaps
    assert result.msa.total_gaps == 9
    # Average gap fraction = (3/18 + 6/18 + 0/18) / 3 = 9/54 = 1/6
    assert result.msa.average_gap_fraction == pytest.approx((3 / 18 + 6 / 18 + 0 / 18) / 3)


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
    """Test that mismatched ID count returns error output."""
    # Validation happens at resolve_sequence_ids call time in run_mafft_align
    # ToolRegistry decorator catches exceptions and returns error output
    inputs = MafftInput(
        sequences=["MVLS", "AVLS"],
        sequence_ids=["only_one"],
    )
    result = run_mafft_align(inputs, MafftConfig())
    assert result.success is False
    assert any("must match" in err for err in result.errors)
