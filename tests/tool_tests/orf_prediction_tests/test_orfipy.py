"""
Tests for Orfipy ORF prediction tool.

Tests the registry-based interface, configuration validation, and parsing utilities.
"""

from pathlib import Path

import pandas as pd
import pytest

from bio_programming.bio_tools.tools.orf_prediction import (
    OrfipyConfig,
    OrfipyInput,
    OrfipyOutput,
    run_orfipy_prediction,
)
from bio_programming.bio_tools.tools.orf_prediction.orf import ORF
from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output

# Test data file paths
TEST_DATA_DIR = Path("tests/dummy_data")
ORFIPY_AA_FILE = TEST_DATA_DIR / "test_orfipy_aa.faa"
ORFIPY_NT_FILE = TEST_DATA_DIR / "test_orfipy_nt.fna"


class TestOrfipyParsing:
    """Test Orfipy result parsing (via output)."""

    def test_parsing_with_test_data(self):
        """Test that test data files can be parsed correctly."""
        if not ORFIPY_AA_FILE.exists() or not ORFIPY_NT_FILE.exists():
            pytest.skip("Test data files not available")

        # Access the private parsing function for testing (import directly from standalone)
        from bio_programming.bio_tools.tools.orf_prediction.orfipy.standalone.run import (
            _parse_orfipy_results,
        )

        results = _parse_orfipy_results(ORFIPY_AA_FILE, ORFIPY_NT_FILE, "dna_seq_1")

        assert len(results) == 4  # Based on test data

        # Check that all expected keys are present
        first_row = results[0]
        assert "parent_id" in first_row
        assert "orf_id" in first_row
        assert "amino_acid_sequence" in first_row
        assert "nucleotide_sequence" in first_row
        assert "nucleotide_start" in first_row
        assert "nucleotide_end" in first_row
        assert "strand" in first_row
        assert "frame" in first_row

        # Check first row data
        assert first_row["parent_id"] == "dna_seq_1"
        assert first_row["orf_id"] == "ORF.1"
        assert first_row["amino_acid_sequence"].startswith("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGK")
        assert first_row["nucleotide_sequence"].startswith("ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAG")

    @pytest.mark.parametrize(
        "header, expected",
        [
            (
                "dna_seq_1_ORF.1 [0-180](+) frame:1",
                {"parent_id": "dna_seq_1", "orf_id": "ORF.1", "start": 0, "end": 180, "strand": "+", "frame": 1}
            ),
            (
                "complex-name_ORF.15 [100-250](-) frame:2",
                {"parent_id": "complex-name", "orf_id": "ORF.15", "start": 100, "end": 250, "strand": "-", "frame": 2}
            ),
            ("invalid header", None)
        ],
    )
    def test_header_parsing(self, header, expected):
        """Test parsing of individual Orfipy headers."""
        # Access the private parsing function for testing (import directly from standalone)
        from bio_programming.bio_tools.tools.orf_prediction.orfipy.standalone.run import (
            _parse_orfipy_header as parse_header,
        )

        result = parse_header(header)

        if expected:
            assert result is not None
            assert result["parent_id"] == expected["parent_id"]
            assert result["orf_id"] == expected["orf_id"]
            assert result["start"] == expected["start"]
            assert result["end"] == expected["end"]
            assert result["strand"] == expected["strand"]
        else:
            assert result is None

    def test_test_data_integrity(self):
        """Test that test data files are consistent."""
        if not ORFIPY_AA_FILE.exists() or not ORFIPY_NT_FILE.exists():
            pytest.skip("Test data files not available")

        # Read both files
        with open(ORFIPY_AA_FILE, 'r') as f:
            aa_lines = f.readlines()

        with open(ORFIPY_NT_FILE, 'r') as f:
            nt_lines = f.readlines()

        # Count headers (should be same in both files)
        aa_headers = [line for line in aa_lines if line.startswith('>')]
        nt_headers = [line for line in nt_lines if line.startswith('>')]

        assert len(aa_headers) == len(nt_headers), \
            "AA and NT files should have same number of sequences"

        # Check that headers match
        for aa_header, nt_header in zip(aa_headers, nt_headers):
            assert aa_header.strip() == nt_header.strip(), \
                f"Headers don't match: {aa_header.strip()} vs {nt_header.strip()}"

@pytest.mark.skip_ci
class TestOrfipyIntegration:
    """Integration tests for the complete Orfipy tool."""

    def test_full_workflow(self):
        """Test complete workflow from input to output."""
        # Create input and config and run
        inputs = OrfipyInput(
            sequences="ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGTGA"
        )
        config = OrfipyConfig(min_len=30)

        result = run_orfipy_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        assert result.tool_id == "orfipy-prediction"
        assert result.num_orfs >= 0

        # Verify granular sequence results
        assert result.predicted_orfs is not None
        assert len(result.predicted_orfs) == 1

        # Verify aggregation consistency
        total_orfs_in_list = sum(len(sr) for sr in result.predicted_orfs)
        assert result.num_orfs == total_orfs_in_list

        # Check that results_df is consistent (always a DataFrame, possibly empty)
        assert isinstance(result.results_df, pd.DataFrame)
        assert len(result.results_df) == result.num_orfs

        # Verify ORFs in sequence results match dataframe content
        if result.num_orfs > 0:
            first_orf_model = result.predicted_orfs[0][0]
            first_row = result.results_df.iloc[0]
            assert first_orf_model.amino_acid_sequence == first_row["amino_acid_sequence"]

    def test_custom_sequence_ids_preserved(self):
        """Test that custom sequence IDs are preserved in output ORFs."""
        # Use a unique sequence to avoid cache interference from other tests
        unique_seq = "ATGAAACCCGGGAAATTTCCCGGGAAATTTCCCGGGAAATTTCCCGGGTAG"
        inputs = OrfipyInput(
            sequences=[unique_seq],
            sequence_ids=["my_custom_gene"],
        )
        config = OrfipyConfig(min_len=30)

        result = run_orfipy_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        # Check that ORFs have the custom parent_id
        if result.num_orfs > 0:
            for orf in result.predicted_orfs[0]:
                assert orf.parent_id == "my_custom_gene"
            # Check in DataFrame too
            assert all(result.results_df["parent_id"] == "my_custom_gene")

    def test_default_sequence_ids_when_not_provided(self):
        """Test that default IDs are generated when not provided."""
        inputs = OrfipyInput(
            sequences=["ATGGTGCTGAGCCCGGCGGACAAGACCAACGTGAAGGCGGCGTGGGGCAAGTGA"]
        )
        config = OrfipyConfig(min_len=30)

        result = run_orfipy_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        # Check that ORFs have the default parent_id
        if result.num_orfs > 0:
            for orf in result.predicted_orfs[0]:
                assert orf.parent_id == "seq_0"

    def test_multiple_sequences_with_custom_ids(self):
        """Test multiple sequences with custom IDs."""
        # Use unique sequences to avoid cache interference
        inputs = OrfipyInput(
            sequences=[
                "ATGCCCAAATTTGGGCCCAAATTTGGGCCCAAATTTGGGTAG",
                "ATGTTTCCCGGGAAATTTCCCGGGTAA",
            ],
            sequence_ids=["gene_a", "gene_b"],
        )
        config = OrfipyConfig(min_len=12)

        result = run_orfipy_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        assert len(result.predicted_orfs) == 2
        # Check parent_ids
        if result.predicted_orfs[0]:
            assert result.predicted_orfs[0][0].parent_id == "gene_a"
        if result.predicted_orfs[1]:
            assert result.predicted_orfs[1][0].parent_id == "gene_b"

    def test_sequence_ids_length_mismatch_raises(self):
        """Test that mismatched ID count raises ValueError."""
        from bio_programming.bio_tools.tools.utils import resolve_sequence_ids
        with pytest.raises(ValueError, match="must match"):
            resolve_sequence_ids(["ATGAAA", "ATGBBB"], ["only_one"])


def _create_sample_orf(
    parent_id: str = "seq_0",
    orf_id: str = "ORF.1",
) -> ORF:
    """Helper to create a sample ORF for testing."""
    return ORF(
        parent_id=parent_id,
        orf_id=orf_id,
        strand="+",
        frame=1,
        amino_acid_sequence="MVLS",
        nucleotide_sequence="ATGGTGCTGAGC",
        amino_acid_length=4,
        nucleotide_length=12,
        nucleotide_start=1,  # 1-indexed (biology convention)
        nucleotide_end=12,   # 1-indexed, inclusive
    )


class TestOrfipyOutputComputedFields:
    """Test OrfipyOutput computed fields (num_orfs, results_df)."""

    @pytest.mark.parametrize(
        "orfs_per_sequence,expected_total",
        [
            ([], 0),  # No sequences
            ([0], 0),  # One sequence, no ORFs
            ([1], 1),  # One sequence, one ORF
            ([3], 3),  # One sequence, multiple ORFs
            ([0, 0, 0], 0),  # Multiple sequences, no ORFs
            ([2, 1, 3], 6),  # Multiple sequences, varying ORF counts
            ([0, 2, 0], 2),  # Mixed empty and non-empty
        ],
    )
    def test_computed_fields_count(self, orfs_per_sequence, expected_total):
        """Test num_orfs and results_df length for various sequence configurations."""
        predicted_orfs = [
            [_create_sample_orf(f"seq_{i}", f"ORF.{j}") for j in range(count)]
            for i, count in enumerate(orfs_per_sequence)
        ]

        output = OrfipyOutput(
            predicted_orfs=predicted_orfs,
            tool_id="orfipy-prediction",
            execution_time=0.1,
            success=True,
        )

        assert output.num_orfs == expected_total
        assert isinstance(output.results_df, pd.DataFrame)
        assert len(output.results_df) == expected_total

    def test_results_df_columns_and_content(self):
        """Test that results_df has correct columns and content."""
        output = OrfipyOutput(
            predicted_orfs=[[_create_sample_orf("seq_0", "ORF.1")]],
            tool_id="orfipy-prediction",
            execution_time=0.1,
            success=True,
        )

        expected_columns = {
            "parent_id",
            "orf_id",
            "id",
            "strand",
            "frame",
            "amino_acid_sequence",
            "nucleotide_sequence",
            "amino_acid_length",
            "nucleotide_length",
            "nucleotide_start",
            "nucleotide_end",
            "metrics",
            "gc_content",
        }
        assert set(output.results_df.columns) == expected_columns
        assert output.results_df.iloc[0]["parent_id"] == "seq_0"
        assert output.results_df.iloc[0]["amino_acid_sequence"] == "MVLS"

    def test_cache_reconstruction(self):
        """Test output works when reconstructed from cache (only predicted_orfs passed)."""
        orfs = [_create_sample_orf("seq_0", "ORF.1"), _create_sample_orf("seq_0", "ORF.2")]

        # Simulate cache reconstruction - minimal fields like tool_cache_iterable does
        output = OrfipyOutput(
            tool_id="orfipy-prediction",
            execution_time=0.0,
            success=True,
            warnings=[],
            metadata={},
            predicted_orfs=[orfs],
        )

        assert output.num_orfs == 2
        assert len(output.results_df) == 2
