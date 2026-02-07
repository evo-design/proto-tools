"""
test_pyhmmer.py

Comprehensive tests for PyHMMER tools in bio_programming.tools.gene_annotation.pyhmmer
"""

from pathlib import Path

import pandas as pd
import pytest
from Bio import SeqIO

# Public API imports
from bio_programming.tools.gene_annotation.pyhmmer import (
    PyJackhmmerConfig,
    PyJackhmmerInput,
    PyHmmerConfig,
    PyHmmerOutput,
    PyHmmscanInput,
    PyHmmsearchInput,
    PyNhmmerInput,
    PyPhmmerInput,
    pyhmmer_jackhmmer,
    pyhmmer_hmmscan,
    pyhmmer_hmmsearch,
    pyhmmer_nhmmer,
    pyhmmer_phmmer,
)

# Private helper functions (imported directly from module for testing)
from bio_programming.tools.gene_annotation.pyhmmer.hmmsearch import (
    _build_dataframes,
    _convert_dtypes,
)
from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output

# ============================================================================
# Sample Data for Testing
# ============================================================================

# Path to test HMM file
TEST_HMM_FILE = Path(__file__).parent.parent / "dummy_data" / "test_multiple_hmm.hmm"

# Load test sequences and properly close the file handle
TEST_FASTA_PATH = Path(__file__).parent.parent / "dummy_data" / "test_sequences_for_pyhmmer.fasta"
with open(TEST_FASTA_PATH, "r") as fasta_file:
    sequence_iterator = SeqIO.parse(fasta_file, "fasta")
    SAMPLE_SEQUENCES = [str(seq.seq) for seq in sequence_iterator]

TEST_DNA_FASTA_PATH = Path(__file__).parent.parent / "dummy_data" / "test_dna_sequences.fna"
with open(TEST_DNA_FASTA_PATH, "r") as dna_fasta_file:
    dna_sequence_iterator = SeqIO.parse(dna_fasta_file, "fasta")
    SAMPLE_DNA_SEQUENCES = [str(seq.seq) for seq in dna_sequence_iterator]


# ============================================================================
# Config Validation Tests
# ============================================================================


class TestPyHmmsearchInput:
    """Tests for PyHmmsearchInput validation."""

    def test_valid_input_with_hmm_file(self):
        """
        Test valid PyHmmsearchInput with HMM file input.
        """
        inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES)
        assert inputs.hmm == str(TEST_HMM_FILE)  # Should be file path
        assert inputs.sequences == SAMPLE_SEQUENCES

    def test_single_sequence_input(self):
        """Test with single sequence string."""
        inputs = PyHmmsearchInput(
            hmm=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES[0]
        )
        assert inputs.sequences == [SAMPLE_SEQUENCES[0]]

    def test_list_sequence_input(self):
        """Test with list of sequence strings."""
        sequence_list = SAMPLE_SEQUENCES[:2]

        inputs = PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=sequence_list)
        assert inputs.sequences == sequence_list

    def test_missing_hmm_file(self):
        """Test that missing HMM file raises ValueError."""
        with pytest.raises(ValueError, match="HMM file not found"):
            PyHmmsearchInput(hmm="/nonexistent/test.hmm", sequences=SAMPLE_SEQUENCES)

    def test_empty_hmm_path(self):
        """Test that empty HMM path raises ValueError."""
        with pytest.raises(ValueError, match="HMM path is not a file"):
            PyHmmsearchInput(hmm="", sequences=SAMPLE_SEQUENCES)

    def test_empty_sequences(self):
        """Test that empty sequences raise ValueError."""
        with pytest.raises(ValueError, match="At least one sequence is required"):
            PyHmmsearchInput(hmm=str(TEST_HMM_FILE), sequences=[])

class TestPyHmmscanInput:
    """Tests for PyHmmscanInput validation."""

    def test_valid_input_with_hmm_file(self):
        """Test valid PyHmmscanInput with HMM database file."""
        input_obj = PyHmmscanInput(hmm_db=str(TEST_HMM_FILE), sequences=SAMPLE_SEQUENCES)
        assert input_obj.hmm_db == str(TEST_HMM_FILE)  # Should be file path
        assert input_obj.sequences == SAMPLE_SEQUENCES

class TestPyPhmmerInput:
    """Tests for PyPhmmerInput validation."""

    def test_valid_input(self):
        """Test valid PyPhmmerInput."""
        inputs = PyPhmmerInput(
            sequences=SAMPLE_SEQUENCES[:2],  # queries
            target_sequences=SAMPLE_SEQUENCES[1:],  # targets
        )
        assert inputs.sequences == SAMPLE_SEQUENCES[:2]
        assert inputs.target_sequences == SAMPLE_SEQUENCES[1:]

    def test_single_strings(self):
        """Test with single string inputs."""
        inputs = PyPhmmerInput(
            sequences="MVLSPADKTNVKAAW", target_sequences="ATCGATCGATCGAT"
        )
        assert inputs.sequences == ["MVLSPADKTNVKAAW"]
        assert inputs.target_sequences == ["ATCGATCGATCGAT"]


class TestPyNhmmerInput:
    """Tests for PyNhmmerInput validation."""

    def test_valid_input(self):
        """Test valid PyNhmmerInput."""
        inputs = PyNhmmerInput(
            sequences=SAMPLE_DNA_SEQUENCES[:1],
            target_sequences=SAMPLE_DNA_SEQUENCES[1:],
        )
        assert inputs.sequences == SAMPLE_DNA_SEQUENCES[:1]
        assert inputs.target_sequences == SAMPLE_DNA_SEQUENCES[1:]


class TestPyJackhmmerInput:
    """Tests for PyJackhmmerInput validation."""

    def test_valid_input(self):
        """Test valid PyJackhmmerInput."""
        inputs = PyJackhmmerInput(
            sequences=SAMPLE_SEQUENCES[:1],
            target_sequences=SAMPLE_SEQUENCES[1:],
        )
        assert inputs.sequences == SAMPLE_SEQUENCES[:1]
        assert inputs.target_sequences == SAMPLE_SEQUENCES[1:]


class TestPyJackhmmerConfig:
    """Tests for PyJackhmmerConfig validation."""

    def test_invalid_max_iterations(self):
        """Test max_iterations must be positive."""
        with pytest.raises(ValueError):
            PyJackhmmerConfig(max_iterations=0)


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_convert_dtypes_sequence_format(self):
        """Test data type conversion for sequence format."""
        df = pd.DataFrame(
            {
                "target_name": ["test"],
                "evalue": ["1.2e-10"],
                "score": ["45.6"],
                "bias": ["0.1"],
            }
        )

        result = _convert_dtypes(df, is_domain=False)

        assert result["evalue"].dtype == float
        assert result["score"].dtype == float
        assert result["bias"].dtype == float

    def test_convert_dtypes_domain_format(self):
        """Test data type conversion for domain format."""
        df = pd.DataFrame(
            {
                "target_name": ["test"],
                "domain_idx": ["1"],
                "c_evalue": ["2.5e-11"],
                "domain_score": ["44.2"],
                "target_length": ["150"],
                "hmm_from": ["5"],
                "hmm_to": ["78"],
            }
        )

        result = _convert_dtypes(df, is_domain=True)

        assert result["domain_idx"].iloc[0] == 1.0
        assert result["c_evalue"].dtype == float
        assert result["domain_score"].dtype == float
        assert result["target_length"].iloc[0] == 150.0
        assert result["hmm_from"].iloc[0] == 5.0
        assert result["hmm_to"].iloc[0] == 78.0

    def test_build_dataframes_empty(self):
        """Test conversion with empty hits list."""
        seq_df, dom_df = _build_dataframes([], [])
        assert seq_df is None
        assert dom_df is None


# ============================================================================
# Tool Execution Tests (Mocked)
# ============================================================================


class TestPyHmmsearchExecution:
    """Tests for pyhmmer_hmmsearch function."""

    def test_pyhmmer_hmmsearch_success(self):
        """Test successful pyhmmer_hmmsearch execution with real PyHMMER."""
        inputs = PyHmmsearchInput(
            hmm=str(TEST_HMM_FILE),
            sequences=SAMPLE_SEQUENCES
        )
        config = PyHmmerConfig(
            evalue_threshold=1000.0,  # Very permissive to allow hits
            domain_evalue_threshold=1000.0,  # Very permissive to allow hits
        )
        result = pyhmmer_hmmsearch(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, PyHmmerOutput)
        assert result.tool_id == "pyhmmer-hmmsearch"
        assert result.num_sequence_hits == 2
        assert result.num_domain_hits == 2

    def test_pyhmmer_hmmsearch_no_hits(self):
        """Test pyhmmer_hmmsearch with very restrictive thresholds to get no hits."""
        inputs = PyHmmsearchInput(
            hmm=str(TEST_HMM_FILE),
            sequences=SAMPLE_SEQUENCES
        )
        config = PyHmmerConfig(
            evalue_threshold=1e-100,  # Extremely restrictive threshold
            domain_evalue_threshold=1e-100,  # Extremely restrictive domain threshold
        )
        result = pyhmmer_hmmsearch(inputs, config)

        assert result.success is True
        # With such restrictive thresholds, we should get no hits
        assert result.num_sequence_hits == 0
        assert result.num_domain_hits == 0


class TestPyHmmscanExecution:
    """Tests for pyhmmer_hmmscan function."""
    def test_pyhmmer_hmmscan_success(self):
        inputs = PyHmmscanInput(
            hmm_db=str(TEST_HMM_FILE),
            sequences=SAMPLE_SEQUENCES,
        )
        config = PyHmmerConfig()
        result = pyhmmer_hmmscan(inputs, config)

        # Validate output and export functionality
        validate_output(result)


class TestPyPhmmerExecution:
    """Tests for pyhmmer_phmmer function."""

    def test_pyhmmer_phmmer_success(self):
        inputs = PyPhmmerInput(
            sequences=SAMPLE_SEQUENCES[:2],
            target_sequences=SAMPLE_SEQUENCES[2:],
        )
        config = PyHmmerConfig()
        result = pyhmmer_phmmer(inputs, config)

        # Validate output (skip export check since test may have no hits)
        validate_output(result, check_export=False)


class TestPyNhmmerExecution:
    """Tests for pyhmmer_nhmmer function."""

    def test_pyhmmer_nhmmer_success(self):
        inputs = PyNhmmerInput(
            sequences=SAMPLE_DNA_SEQUENCES[:1],
            target_sequences=SAMPLE_DNA_SEQUENCES,
        )
        config = PyHmmerConfig(
            evalue_threshold=1000.0,
            domain_evalue_threshold=1000.0,
        )
        result = pyhmmer_nhmmer(inputs, config)

        validate_output(result, check_export=False)

        assert isinstance(result, PyHmmerOutput)
        assert result.tool_id == "pyhmmer-nhmmer"
        assert result.num_sequence_hits >= 1


class TestPyJackhmmerExecution:
    """Tests for pyhmmer_jackhmmer function."""

    def test_pyhmmer_jackhmmer_success(self):
        inputs = PyJackhmmerInput(
            sequences=SAMPLE_SEQUENCES[:1],
            target_sequences=SAMPLE_SEQUENCES,
        )
        config = PyJackhmmerConfig(
            max_iterations=2,
            evalue_threshold=1000.0,
            domain_evalue_threshold=1000.0,
        )
        result = pyhmmer_jackhmmer(inputs, config)

        validate_output(result, check_export=False)

        assert isinstance(result, PyHmmerOutput)
        assert result.tool_id == "pyhmmer-jackhmmer"
        assert result.num_sequence_hits >= 1
        assert result.metadata["max_iterations"] == 2
        assert len(result.metadata["iterations_per_query"]) == 1
