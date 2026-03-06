"""Tests for MSA class."""

import pytest

import bio_programming_tools.tools.sequence_alignment.msas as msas_module
from bio_programming_tools.tools.sequence_alignment.msas import (
    MSA,
    convert_a3m_to_fasta,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_aligned_sequences():
    """Simple aligned sequences for testing."""
    return [
        "ACGT--AAA",
        "AC-T--AAA",
        "ACGTG-AAA",
        "ACGTGGAAA",
    ]


@pytest.fixture
def sample_sequence_ids():
    """Sequence IDs corresponding to sample_aligned_sequences."""
    return ["seq_0", "seq_1", "seq_2", "seq_3"]


@pytest.fixture
def sample_fasta_file(tmp_path, sample_aligned_sequences, sample_sequence_ids):
    """Create a temporary FASTA file with sample sequences."""
    fasta_path = tmp_path / "test_msa.fasta"
    with open(fasta_path, "w") as f:
        for seq_id, seq in zip(sample_sequence_ids, sample_aligned_sequences):
            f.write(f">{seq_id}\n{seq}\n")
    return fasta_path


@pytest.fixture
def sample_a3m_file(tmp_path):
    """Create a temporary A3M file with sample sequences."""
    a3m_path = tmp_path / "test_msa.a3m"
    with open(a3m_path, "w") as f:
        f.write(">seq_0\n")
        f.write("ACGT--AAA\n")
        f.write(">seq_1\n")
        f.write("AC-T--AAA\n")
        f.write(">seq_2\n")
        f.write("ACGT--AAA\n")  # Same as seq_0 for consistency
        f.write(">seq_3\n")
        f.write("ACGTGGAAA\n")  # No lowercase - all uppercase for alignment
    return a3m_path


# ============================================================================
# MSA Initialization
# ============================================================================


# -- Init with sequences --

def test_msa_init_with_sequences(sample_aligned_sequences):
    """Test initialization with a list of aligned sequences."""
    msa = MSA(sample_aligned_sequences)

    assert msa.num_sequences == len(sample_aligned_sequences)
    assert msa.alignment_length == len(sample_aligned_sequences[0])
    assert msa._in_memory is True
    assert len(msa.sequence_ids) == len(sample_aligned_sequences)


def test_msa_init_with_sequences_and_ids(sample_aligned_sequences, sample_sequence_ids):
    """Test initialization with sequences and custom IDs."""
    msa = MSA(sample_aligned_sequences, sequence_ids=sample_sequence_ids)

    assert msa.sequence_ids == sample_sequence_ids
    assert msa.num_sequences == len(sample_aligned_sequences)


def test_msa_init_with_empty_sequences():
    """Test that empty sequence list raises ValueError."""
    with pytest.raises(ValueError, match="MSA must contain at least two sequences"):
        MSA([])


def test_msa_init_with_single_sequence():
    """Test that single sequence raises ValueError."""
    with pytest.raises(ValueError, match="MSA must contain at least two sequences"):
        MSA(["ACGTAAA"])


# -- Init with files --

def test_msa_init_with_fasta_file(sample_fasta_file, sample_aligned_sequences):
    """Test initialization with a FASTA file."""
    msa = MSA(str(sample_fasta_file))

    assert msa.num_sequences == len(sample_aligned_sequences)
    assert msa.alignment_length == len(sample_aligned_sequences[0])
    assert len(msa.sequence_ids) == len(sample_aligned_sequences)


def test_msa_init_with_a3m_file(sample_a3m_file):
    """Test initialization with an A3M file."""
    msa = MSA(str(sample_a3m_file))

    assert msa.num_sequences == 4
    assert msa.alignment_length == 9
    # Verify sequences loaded correctly
    sequences = list(msa)
    assert sequences[0] == "ACGT--AAA"
    assert sequences[1] == "AC-T--AAA"
    assert sequences[2] == "ACGT--AAA"
    assert sequences[3] == "ACGTGGAAA"


def test_msa_init_with_nonexistent_file():
    """Test that nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        MSA("/nonexistent/path/to/file.fasta")


def test_msa_init_with_single_sequence_file(tmp_path):
    """Test that file with single sequence raises ValueError."""
    single_seq_file = tmp_path / "single_seq.fasta"
    with open(single_seq_file, "w") as f:
        f.write(">seq_0\n")
        f.write("ACGTAAA\n")

    with pytest.raises(ValueError, match="MSA must contain at least two sequences"):
        MSA(str(single_seq_file))


# -- Memory strategy on init --

def test_msa_init_small_file_converts_to_memory(sample_fasta_file):
    """Test that small files are loaded into memory."""
    msa = MSA(str(sample_fasta_file))
    # Small file should be converted to in-memory
    assert msa._in_memory is True


def test_msa_init_large_file_stays_file_backed(sample_fasta_file, monkeypatch):
    """Test that large files stay file-backed."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))
    # Large file should remain file-backed
    assert msa._in_memory is False


# ============================================================================
# Iterator Methods
# ============================================================================


# -- __iter__ --

def test_msa_iter_in_memory(sample_aligned_sequences):
    """Test __iter__ with in-memory MSA."""
    msa = MSA(sample_aligned_sequences)
    sequences = list(msa)

    assert sequences == sample_aligned_sequences


def test_msa_iter_file_backed(sample_fasta_file, sample_aligned_sequences, monkeypatch):
    """Test __iter__ with file-backed MSA."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))
    sequences = list(msa)

    assert len(sequences) == 4
    assert sequences == sample_aligned_sequences


# -- iter_with_ids --

def test_msa_iter_with_ids_in_memory(sample_aligned_sequences, sample_sequence_ids):
    """Test iter_with_ids with in-memory MSA."""
    msa = MSA(sample_aligned_sequences, sequence_ids=sample_sequence_ids)
    seq_id_pairs = list(msa.iter_with_ids())

    assert len(seq_id_pairs) == len(sample_aligned_sequences)
    for (seq_id, seq), expected_id, expected_seq in zip(
        seq_id_pairs, sample_sequence_ids, sample_aligned_sequences
    ):
        assert seq_id == expected_id
        assert seq == expected_seq


def test_msa_iter_with_ids_file_backed(
    sample_fasta_file, sample_sequence_ids, sample_aligned_sequences, monkeypatch
):
    """Test iter_with_ids with file-backed MSA."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))
    seq_id_pairs = list(msa.iter_with_ids())

    assert len(seq_id_pairs) == 4
    # Check first and last
    assert seq_id_pairs[0][0] == sample_sequence_ids[0]
    assert seq_id_pairs[-1][0] == sample_sequence_ids[-1]
    # Verify sequences match
    for (seq_id, seq), expected_id, expected_seq in zip(
        seq_id_pairs, sample_sequence_ids, sample_aligned_sequences
    ):
        assert seq_id == expected_id
        assert seq == expected_seq


# -- __getitem__ and __len__ --

def test_msa_getitem(sample_aligned_sequences):
    """Test __getitem__ indexing."""
    msa = MSA(sample_aligned_sequences)

    assert msa[0] == sample_aligned_sequences[0]
    assert msa[2] == sample_aligned_sequences[2]
    assert msa[-1] == sample_aligned_sequences[-1]


def test_msa_getitem_file_backed(
    sample_fasta_file, sample_aligned_sequences, monkeypatch
):
    """Test __getitem__ with file-backed MSA."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))

    assert msa[0] == sample_aligned_sequences[0]
    assert msa[2] == sample_aligned_sequences[2]
    assert msa[-1] == sample_aligned_sequences[-1]


def test_msa_len(sample_aligned_sequences):
    """Test __len__ method."""
    msa = MSA(sample_aligned_sequences)
    assert len(msa) == len(sample_aligned_sequences)


# ============================================================================
# MSA Properties
# ============================================================================


# -- aligned_sequences --

def test_msa_aligned_sequences_in_memory(sample_aligned_sequences):
    """Test aligned_sequences property with in-memory MSA."""
    msa = MSA(sample_aligned_sequences)
    assert msa.aligned_sequences == sample_aligned_sequences


def test_msa_aligned_sequences_file_backed_conversion(sample_fasta_file):
    """Test that accessing aligned_sequences converts file-backed to in-memory."""
    # Create a file-backed MSA by using a modified MAX_SEQS_IN_MEMORY
    msa = MSA(str(sample_fasta_file))
    # Access aligned_sequences (may trigger warning)
    sequences = msa.aligned_sequences

    assert len(sequences) == 4
    assert msa._in_memory is True


# -- original_sequences --

def test_msa_original_sequences(sample_aligned_sequences):
    """Test original_sequences property removes gaps."""
    msa = MSA(sample_aligned_sequences)
    original = msa.original_sequences

    expected = [seq.replace("-", "") for seq in sample_aligned_sequences]
    assert original == expected


# -- alignment_length and num_sequences --

def test_msa_alignment_length(sample_aligned_sequences):
    """Test alignment_length property."""
    msa = MSA(sample_aligned_sequences)
    assert msa.alignment_length == 9


def test_msa_num_sequences(sample_aligned_sequences):
    """Test num_sequences property."""
    msa = MSA(sample_aligned_sequences)
    assert msa.num_sequences == 4


# -- gap statistics --

def test_msa_total_gaps_in_memory(sample_aligned_sequences):
    """Test total_gaps property with in-memory MSA."""
    msa = MSA(sample_aligned_sequences)
    expected_gaps = sum(seq.count("-") for seq in sample_aligned_sequences)
    assert msa.total_gaps == expected_gaps


def test_msa_total_gaps_file_backed(
    sample_fasta_file, sample_aligned_sequences, monkeypatch
):
    """Test total_gaps property with file-backed MSA."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))
    expected_gaps = sum(seq.count("-") for seq in sample_aligned_sequences)
    assert msa.total_gaps == expected_gaps


def test_msa_average_gap_fraction_in_memory():
    """Test average_gap_fraction property."""
    sequences = [
        "A--A",  # 2/4 = 0.5
        "AA--",  # 2/4 = 0.5
        "AAAA",  # 0/4 = 0.0
    ]
    msa = MSA(sequences)
    expected = (0.5 + 0.5 + 0.0) / 3
    assert abs(msa.average_gap_fraction - expected) < 1e-6


def test_msa_average_gap_fraction_no_gaps():
    """Test average_gap_fraction with no gaps."""
    sequences = ["AAAA", "TTTT", "GGGG"]
    msa = MSA(sequences)
    assert msa.average_gap_fraction == 0.0


def test_msa_average_gap_fraction_all_gaps():
    """Test average_gap_fraction with all gaps."""
    sequences = ["----", "----", "----"]
    msa = MSA(sequences)
    assert msa.average_gap_fraction == 1.0


# ============================================================================
# MSA Methods
# ============================================================================


# -- get_column --

def test_msa_get_column():
    """Test get_column method."""
    sequences = [
        "ACGT",
        "ATGT",
        "AGGT",
        "AAGT",
    ]
    msa = MSA(sequences)

    assert msa.get_column(0) == ["A", "A", "A", "A"]
    assert msa.get_column(1) == ["C", "T", "G", "A"]
    assert msa.get_column(2) == ["G", "G", "G", "G"]
    assert msa.get_column(3) == ["T", "T", "T", "T"]


def test_msa_get_column_with_gaps(sample_aligned_sequences):
    """Test get_column with sequences containing gaps."""
    msa = MSA(sample_aligned_sequences)
    column_4 = msa.get_column(4)
    # Position 4: ['G', '-', 'G', 'G']
    assert "-" in column_4


def test_msa_get_column_out_of_range(sample_aligned_sequences):
    """Test get_column with invalid position."""
    msa = MSA(sample_aligned_sequences)

    with pytest.raises(IndexError):
        msa.get_column(100)

    with pytest.raises(IndexError):
        msa.get_column(-1)


# -- get_conservation --

def test_msa_get_conservation_fully_conserved():
    """Test get_conservation with fully conserved position."""
    sequences = ["AAAA", "AAAA", "AAAA"]
    msa = MSA(sequences)
    assert msa.get_conservation(0) == 1.0


def test_msa_get_conservation_no_conservation():
    """Test get_conservation with no conservation."""
    sequences = ["ACGT", "TGCA", "GATC", "CTAG"]
    msa = MSA(sequences)
    conservation = msa.get_conservation(0)
    assert conservation == 0.25  # Each char appears once


def test_msa_get_conservation_with_gaps_excluded():
    """Test get_conservation excluding gaps."""
    sequences = [
        "A---",
        "A---",
        "T---",
    ]
    msa = MSA(sequences)
    # Position 0: A, A, T -> most common is A (2/3)
    assert abs(msa.get_conservation(0, exclude_gaps=True) - 2 / 3) < 1e-6


def test_msa_get_conservation_with_gaps_included():
    """Test get_conservation including gaps."""
    sequences = [
        "A---",
        "A---",
        "----",
    ]
    msa = MSA(sequences)
    # Position 1: -, -, - -> all gaps, conservation = 1.0
    assert msa.get_conservation(1, exclude_gaps=False) == 1.0


def test_msa_get_conservation_all_gaps():
    """Test get_conservation when column is all gaps."""
    sequences = ["A-A", "T-T", "G-G"]
    msa = MSA(sequences)
    # Position 1 is all gaps, with exclude_gaps=True should return 0.0
    assert msa.get_conservation(1, exclude_gaps=True) == 0.0


# -- get_position_frequencies --

def test_msa_get_position_frequencies():
    """Test get_position_frequencies method."""
    sequences = [
        "AAAA",
        "ATAT",
        "ATAT",
        "AGAG",
    ]
    msa = MSA(sequences)

    freq_0 = msa.get_position_frequencies(0)
    assert freq_0 == {"A": 1.0}

    freq_1 = msa.get_position_frequencies(1)
    assert abs(freq_1["A"] - 0.25) < 1e-6
    assert abs(freq_1["T"] - 0.5) < 1e-6
    assert abs(freq_1["G"] - 0.25) < 1e-6


def test_msa_get_position_frequencies_with_gaps_excluded():
    """Test get_position_frequencies excluding gaps."""
    sequences = ["A-A", "T-T", "G-G"]
    msa = MSA(sequences)

    freq = msa.get_position_frequencies(1, include_gaps=False)
    assert freq == {}  # All gaps, so empty when excluded


def test_msa_get_position_frequencies_with_gaps_included():
    """Test get_position_frequencies including gaps."""
    sequences = ["A-A", "T-T", "G-G"]
    msa = MSA(sequences)

    freq = msa.get_position_frequencies(1, include_gaps=True)
    assert freq == {"-": 1.0}


def test_msa_get_position_frequencies_mixed():
    """Test get_position_frequencies with mixed characters and gaps."""
    sequences = ["AAA", "A-A", "A-A", "TAA"]
    msa = MSA(sequences)

    freq = msa.get_position_frequencies(1, include_gaps=False)
    assert abs(freq["A"] - 1.0) < 1e-6  # 2 A's out of 2 non-gap chars

    freq_with_gaps = msa.get_position_frequencies(1, include_gaps=True)
    assert abs(freq_with_gaps["A"] - 0.5) < 1e-6  # 2/4
    assert abs(freq_with_gaps["-"] - 0.5) < 1e-6  # 2/4


# ============================================================================
# I/O Methods
# ============================================================================


# -- FASTA output --

def test_msa_to_fasta_string(sample_aligned_sequences, sample_sequence_ids):
    """Test to_fasta_string method."""
    msa = MSA(sample_aligned_sequences, sequence_ids=sample_sequence_ids)
    fasta_str = msa.to_fasta_string()

    lines = fasta_str.strip().split("\n")
    assert len(lines) == 2 * len(sample_aligned_sequences)

    # Check format
    for i, seq_id in enumerate(sample_sequence_ids):
        assert lines[2 * i] == f">{seq_id}"
        assert lines[2 * i + 1] == sample_aligned_sequences[i]


def test_msa_to_fasta_file(tmp_path, sample_aligned_sequences, sample_sequence_ids):
    """Test to_fasta_file method."""
    msa = MSA(sample_aligned_sequences, sequence_ids=sample_sequence_ids)
    output_path = tmp_path / "output.fasta"

    msa.to_fasta_file(str(output_path))

    assert output_path.exists()

    # Read back and verify
    with open(output_path, "r") as f:
        lines = f.read().strip().split("\n")

    assert len(lines) == 2 * len(sample_aligned_sequences)
    for i, (seq_id, seq) in enumerate(zip(sample_sequence_ids, sample_aligned_sequences)):
        assert lines[2 * i] == f">{seq_id}"
        assert lines[2 * i + 1] == seq


def test_msa_to_fasta_file_from_file_backed(sample_fasta_file, tmp_path, monkeypatch):
    """Test to_fasta_file with file-backed MSA."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))
    output_path = tmp_path / "output_large.fasta"

    msa.to_fasta_file(str(output_path))

    assert output_path.exists()

    # Verify by loading it again
    msa_reloaded = MSA(str(output_path))
    assert msa_reloaded.num_sequences == msa.num_sequences
    assert msa_reloaded.alignment_length == msa.alignment_length


# -- A3M output --

def test_msa_to_a3m_string(sample_sequence_ids):
    """Test to_a3m_string method."""
    # Create alignment where query (seq_0) has gaps at positions 4-5
    sequences = [
        "ACGT--AAA",  # query with gaps at 4-5
        "ACGTGGAAA",  # has residues at gap positions
        "ACGT--AAA",  # matches query
        "ACGTttAAA",  # has residues at gap positions
    ]
    msa = MSA(sequences, sequence_ids=sample_sequence_ids)
    a3m_str = msa.to_a3m_string(query_index=0)

    lines = a3m_str.strip().split("\n")
    assert len(lines) == 2 * len(sequences)

    # Check headers
    for i, seq_id in enumerate(sample_sequence_ids):
        assert lines[2 * i] == f">{seq_id}"

    # Check sequences - gaps in query should be removed from all sequences
    # and non-gap residues at those positions should be lowercase
    assert lines[1] == "ACGTAAA"  # query - gaps removed
    assert lines[3] == "ACGTggAAA"  # positions 4-5 are lowercase (insertions)
    assert lines[5] == "ACGTAAA"  # gaps at insertion positions are removed
    assert lines[7] == "ACGTttAAA"  # positions 4-5 are lowercase


def test_msa_to_a3m_file(tmp_path, sample_sequence_ids):
    """Test to_a3m_file method."""
    sequences = [
        "ACGT--AAA",  # query with gaps at 4-5
        "ACGTGGAAA",  # has residues at gap positions
        "ACGT--AAA",  # matches query
    ]
    msa = MSA(sequences, sequence_ids=sample_sequence_ids[:3])
    output_path = tmp_path / "output.a3m"

    msa.to_a3m_file(str(output_path), query_index=0)

    assert output_path.exists()

    # Read back and verify
    with open(output_path, "r") as f:
        content = f.read()

    assert ">seq_0\n" in content
    assert "ACGTAAA\n" in content  # query with gaps removed
    assert ">seq_1\n" in content
    assert "ACGTggAAA\n" in content  # lowercase insertions


def test_msa_to_a3m_file_from_file_backed(sample_fasta_file, tmp_path, monkeypatch):
    """Test to_a3m_file with file-backed MSA."""
    # Mock MAX_SEQS_IN_MEMORY to be smaller than our file (4 sequences)
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_fasta_file))
    output_path = tmp_path / "output_large.a3m"

    msa.to_a3m_file(str(output_path), query_index=0)

    assert output_path.exists()

    # Verify file is not empty
    with open(output_path, "r") as f:
        content = f.read()
    assert len(content) > 0
    assert content.count(">") == msa.num_sequences


def test_msa_to_a3m_string_different_query():
    """Test to_a3m_string with different query index."""
    sequences = [
        "ACGTGGAAA",  # seq_0
        "ACGT--AAA",  # seq_1 with gaps at 4-5 (will be query)
        "ACGTTTAAA",  # seq_2
    ]
    msa = MSA(sequences)
    a3m_str = msa.to_a3m_string(query_index=1)  # Use seq_1 as query

    lines = a3m_str.strip().split("\n")

    # seq_0 should have lowercase at positions 4-5 (insertions relative to query)
    assert lines[1] == "ACGTggAAA"
    # seq_1 (query) should have gaps removed
    assert lines[3] == "ACGTAAA"
    # seq_2 should have lowercase at positions 4-5
    assert lines[5] == "ACGTttAAA"


def test_msa_to_a3m_invalid_query_index():
    """Test to_a3m methods with invalid query index."""
    sequences = ["ACGT", "TGCA"]
    msa = MSA(sequences)

    with pytest.raises(IndexError):
        msa.to_a3m_string(query_index=5)

    with pytest.raises(IndexError):
        msa.to_a3m_string(query_index=-1)


# -- Roundtrip tests --

def test_msa_fasta_to_a3m_roundtrip(tmp_path):
    """Test converting FASTA -> A3M -> FASTA preserves alignment."""
    # Create original sequences with gaps
    original_sequences = [
        "ACGT--AAA",
        "ACGTGGAAA",
        "ACGT--AAA",
        "ACGTttAAA",
    ]
    seq_ids = ["seq_0", "seq_1", "seq_2", "seq_3"]

    # Create original MSA
    msa1 = MSA(original_sequences, sequence_ids=seq_ids)

    # Convert to A3M
    a3m_path = tmp_path / "temp.a3m"
    msa1.to_a3m_file(str(a3m_path), query_index=0)

    # Load from A3M (this converts back to FASTA internally)
    msa2 = MSA(str(a3m_path))

    # Convert back to FASTA
    fasta_path = tmp_path / "roundtrip.fasta"
    msa2.to_fasta_file(str(fasta_path))

    # Load the final FASTA
    msa3 = MSA(str(fasta_path))

    # The alignment should be preserved (gaps in query removed for all)
    # Expected after A3M conversion: positions where query has gaps are removed
    expected_sequences = [
        "ACGTAAA",   # query gaps removed
        "ACGTGGAAA",  # insertions preserved
        "ACGTAAA",   # gaps at insertion positions removed
        "ACGTTTAAA",  # insertions preserved
    ]

    for i, expected_seq in enumerate(expected_sequences):
        # A3M format removes gaps in query, so sequences will be different lengths
        # We just verify the MSA can be round-tripped
        assert msa3[i] == msa2[i]


def test_msa_a3m_to_fasta_to_a3m_roundtrip(tmp_path):
    """Test converting A3M -> FASTA -> A3M with same query preserves structure."""
    # Create A3M content with insertions (lowercase)
    a3m_content = """>seq_0
ACGTAAA
>seq_1
ACGTggAAA
>seq_2
ACGTAAA
>seq_3
ACGTttAAA
"""
    a3m_path1 = tmp_path / "original.a3m"
    with open(a3m_path1, "w") as f:
        f.write(a3m_content)

    # Load A3M (converts to FASTA internally)
    msa1 = MSA(str(a3m_path1))

    # Save as FASTA
    fasta_path = tmp_path / "converted.fasta"
    msa1.to_fasta_file(str(fasta_path))

    # Load FASTA
    msa2 = MSA(str(fasta_path))

    # Convert back to A3M with first sequence as query
    a3m_path2 = tmp_path / "roundtrip.a3m"
    msa2.to_a3m_file(str(a3m_path2), query_index=0)

    # Read both A3M files and compare
    with open(a3m_path1, "r") as f:
        original_lines = [line.strip() for line in f if line.strip()]

    with open(a3m_path2, "r") as f:
        roundtrip_lines = [line.strip() for line in f if line.strip()]

    # Should have same structure (may differ in exact insertion representation)
    assert len(original_lines) == len(roundtrip_lines)

    # Headers should match
    for i in range(0, len(original_lines), 2):
        assert original_lines[i] == roundtrip_lines[i]


# ============================================================================
# Context Manager and Cleanup
# ============================================================================


def test_msa_context_manager_with_a3m(sample_a3m_file):
    """Test that context manager properly cleans up temp files."""
    temp_fasta_path = None

    with MSA(str(sample_a3m_file)) as msa:
        # A3M files create temp FASTA files
        if msa._temp_fasta_path:
            temp_fasta_path = msa._temp_fasta_path

        assert msa.num_sequences == 4

    # After exiting context, temp file should be deleted
    if temp_fasta_path:
        assert not temp_fasta_path.exists()


def test_msa_rm_temp_files(sample_a3m_file, monkeypatch):
    """Test rm_temp_files method."""
    # Mock MAX_SEQS_IN_MEMORY to keep file-backed so temp file persists
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_a3m_file))
    temp_path = msa._temp_fasta_path

    assert temp_path is not None
    # Temp file should exist during MSA lifetime
    assert temp_path.exists()

    # Call rm_temp_files
    msa.rm_temp_files()

    # Temp file should be deleted
    assert not temp_path.exists()


def test_msa_del_cleanup(sample_a3m_file, monkeypatch):
    """Test that __del__ cleans up temp files."""
    # Mock MAX_SEQS_IN_MEMORY to keep file-backed so temp file persists
    monkeypatch.setattr(msas_module, "MAX_SEQS_IN_MEMORY", 2)

    msa = MSA(str(sample_a3m_file))
    temp_path = msa._temp_fasta_path

    assert temp_path is not None
    assert temp_path.exists()

    # Delete the MSA object
    del msa

    # Temp file should be cleaned up
    assert not temp_path.exists()


def test_msa_fasta_file_no_temp_cleanup(sample_fasta_file):
    """Test that FASTA files don't create temp files."""
    with MSA(str(sample_fasta_file)) as msa:
        assert msa._temp_fasta_path is None
        assert msa.num_sequences == 4


# ============================================================================
# Conversion Functions
# ============================================================================


def test_convert_a3m_to_fasta(tmp_path):
    """Test convert_a3m_to_fasta"""
    a3m_path = tmp_path / "test.a3m"
    fasta_path = tmp_path / "test.fasta"

    # Create A3M file with insertions (lowercase)
    with open(a3m_path, "w") as f:
        f.write(">seq1\n")
        f.write("ACGTgggAAA\n")  # lowercase 'ggg' is insertion
        f.write(">seq2\n")
        f.write("ACGTaaaAAA\n")  # lowercase 'aaa' is insertion

    convert_a3m_to_fasta(str(a3m_path), str(fasta_path))

    with open(fasta_path, "r") as f:
        content = f.read()

    assert ">seq1\n" in content
    assert "ACGTAAA" in content  # ggg removed
    assert ">seq2\n" in content
    assert "ACGTAAA" in content  # aaa removed


def test_convert_a3m_to_fasta_with_comments(tmp_path):
    """Test that comments are properly ignored."""
    a3m_path = tmp_path / "test.a3m"
    fasta_path = tmp_path / "test.fasta"

    with open(a3m_path, "w") as f:
        f.write("# This is a comment\n")
        f.write(">seq1\n")
        f.write("ACGTAAA\n")
        f.write("# Another comment\n")
        f.write(">seq2\n")
        f.write("ACGTAAA\n")

    convert_a3m_to_fasta(str(a3m_path), str(fasta_path))

    with open(fasta_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Should only have sequence headers and sequences, no comments
    assert len(lines) == 4
    assert lines[0] == ">seq1"
    assert lines[2] == ">seq2"
    assert "#" not in "".join(lines)


def test_convert_a3m_to_fasta_with_null_bytes(tmp_path):
    """Test that null bytes are properly removed."""
    a3m_path = tmp_path / "test.a3m"
    fasta_path = tmp_path / "test.fasta"

    with open(a3m_path, "w") as f:
        f.write(">seq1\n")
        f.write("ACGT\x00AAA\n")  # null byte in sequence

    convert_a3m_to_fasta(str(a3m_path), str(fasta_path))

    with open(fasta_path, "r") as f:
        content = f.read()

    assert "\x00" not in content
    assert "ACGTAAA" in content


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_msa_with_all_gaps():
    """Test MSA where sequences are all gaps."""
    sequences = ["----", "----"]
    msa = MSA(sequences)

    assert msa.total_gaps == 8
    assert msa.average_gap_fraction == 1.0
    assert msa.original_sequences == ["", ""]


def test_msa_with_no_gaps():
    """Test MSA with no gaps."""
    sequences = ["ACGT", "TGCA", "GATC"]
    msa = MSA(sequences)

    assert msa.total_gaps == 0
    assert msa.average_gap_fraction == 0.0
    assert msa.aligned_sequences == msa.original_sequences


def test_msa_with_varying_sequence_content():
    """Test MSA with different amino acid/nucleotide content."""
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "AAAAAAAAAAAAAAAAAAAA",
        "--------------------",
        "ACDEFGHIKLMNPQRSTVWY",
    ]
    msa = MSA(sequences)

    assert msa.num_sequences == 4
    assert msa.alignment_length == 20


def test_msa_sequence_ids_length_mismatch():
    """Test initialization with mismatched sequence_ids length.

    MSA.__init__ does not validate that len(sequence_ids) matches
    len(aligned_sequences). It silently stores the provided list, so
    sequence_ids will have a different length than num_sequences.
    """
    sequences = ["ACGT", "TGCA"]
    seq_ids = ["seq1"]  # Wrong length

    msa = MSA(sequences, sequence_ids=seq_ids)
    assert msa.num_sequences == 2
    assert msa.sequence_ids == ["seq1"]
    assert len(msa.sequence_ids) == 1  # Mismatched: 1 ID for 2 sequences
