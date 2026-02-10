"""
test_mmseqs.py

Comprehensive tests for MMseqs2 tools in bio_programming.bio_tools.tools.gene_annotation.mmseqs
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Public API imports
from bio_programming.bio_tools.tools.gene_annotation.mmseqs import (
    MmseqsClusteringConfig,
    MmseqsClusteringInput,
    MmseqsClusteringOutput,
    MmseqsClusterResult,
    MmseqsHit,
    MmseqsSearchGenomesConfig,
    MmseqsSearchGenomesInput,
    MmseqsSearchGenomesOutput,
    MmseqsSearchProteinsConfig,
    MmseqsSearchProteinsInput,
    MmseqsSearchProteinsOutput,
    MmseqsSequenceSearchResult,
    run_mmseqs_clustering,
    run_mmseqs_search_genomes,
    run_mmseqs_search_proteins,
)

# Private helper functions (imported directly from submodules for testing)
from bio_programming.bio_tools.tools.gene_annotation.mmseqs.search_proteins import (
    _build_sequence_search_results,
    _filter_top_hits,
    _parse_m8_output,
)
from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output

# Test data file paths
TEST_DATA_DIR = Path("tests/dummy_data")
PROTEIN_FASTA = TEST_DATA_DIR / "test_protein_sequences.faa"
DNA_FASTA = TEST_DATA_DIR / "test_dna_sequences.fna"
M8_FILE = TEST_DATA_DIR / "test_mmseqs_results.m8"


# Fixtures for managing temporary files
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


# ============================================================================
# Schema Tests
# ============================================================================


class TestMmseqsSchemas:
    """Tests for MMseqs schema classes."""

    def test_mmseqs_hit_creation(self):
        """Test MmseqsHit creation and properties."""
        hit = MmseqsHit(target_id="db_seq_1", pident=95.5, evalue=1e-50)
        assert hit.target_id == "db_seq_1"
        assert hit.pident == 95.5
        assert hit.evalue == 1e-50

    def test_mmseqs_sequence_search_result(self):
        """Test MmseqsSequenceSearchResult creation and properties."""
        hits = [
            MmseqsHit(target_id="db_1", pident=95.0, evalue=1e-50),
            MmseqsHit(target_id="db_2", pident=85.0, evalue=1e-30),
        ]
        result = MmseqsSequenceSearchResult(
            query_id="seq_0",
            query_sequence="MVLSPADKTN",
            hits=hits,
        )
        assert result.query_id == "seq_0"
        assert result.query_sequence == "MVLSPADKTN"
        assert result.num_hits == 2
        assert result.has_hits is True
        assert result.top_hit.pident == 95.0

    def test_mmseqs_sequence_search_result_no_hits(self):
        """Test MmseqsSequenceSearchResult with no hits."""
        result = MmseqsSequenceSearchResult(
            query_id="seq_0",
            query_sequence="MVLSPADKTN",
            hits=[],
        )
        assert result.num_hits == 0
        assert result.has_hits is False
        assert result.top_hit is None

    def test_mmseqs_cluster_result(self):
        """Test MmseqsClusterResult creation."""
        result = MmseqsClusterResult(
            sequence_id="seq_0",
            input_sequence="MVLSPADKTN",
            cluster_id="seq_0",
            is_representative=True,
        )
        assert result.sequence_id == "seq_0"
        assert result.input_sequence == "MVLSPADKTN"
        assert result.cluster_id == "seq_0"
        assert result.is_representative is True

    def test_mmseqs_search_proteins_output_iteration(self):
        """Test MmseqsSearchProteinsOutput iteration and indexing."""
        results = [
            MmseqsSequenceSearchResult(query_id="seq_0", query_sequence="SEQ1", hits=[]),
            MmseqsSequenceSearchResult(query_id="seq_1", query_sequence="SEQ2", hits=[]),
        ]
        output = MmseqsSearchProteinsOutput(results=results, metadata={})

        assert len(output) == 2
        assert output[0].query_sequence == "SEQ1"
        assert output[1].query_sequence == "SEQ2"
        assert list(output) == results
        assert output.total_hits == 0

    def test_mmseqs_clustering_output_representative_indices(self):
        """Test MmseqsClusteringOutput representative_indices property."""
        results = [
            MmseqsClusterResult(sequence_id="seq_0", input_sequence="S1", cluster_id="seq_0", is_representative=True),
            MmseqsClusterResult(sequence_id="seq_1", input_sequence="S2", cluster_id="seq_0", is_representative=False),
            MmseqsClusterResult(sequence_id="seq_2", input_sequence="S3", cluster_id="seq_2", is_representative=True),
        ]
        output = MmseqsClusteringOutput(results=results, metadata={})

        assert output.representative_indices == [0, 2]
        assert output.num_clusters == 2


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestParseM8Output:
    """Tests for _parse_m8_output function."""

    def test_parse_valid_m8_string(self):
        """Test parsing of a valid M8 string to DataFrame."""
        if not M8_FILE.exists():
            pytest.skip("Test data file not found")

        raw_output = M8_FILE.read_text()
        df = _parse_m8_output(raw_output)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # Based on test data
        assert list(df.columns) == ["query", "target", "pident", "evalue"]
        assert "protein_seq_1" in df["query"].values
        assert "protein_seq_2" in df["query"].values

    def test_parse_empty_string(self):
        """Test parsing of empty string."""
        df = _parse_m8_output("")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["query", "target", "pident", "evalue"]

    def test_parse_whitespace_only(self):
        """Test parsing of whitespace-only string."""
        df = _parse_m8_output("   \n\n  ")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["query", "target", "pident", "evalue"]

class TestFilterTopHits:
    """Tests for _filter_top_hits function."""

    def test_filter_by_highest_pident(self):
        """Test that filtering keeps highest percent identity per query."""
        data = {
            "query": ["q1", "q1", "q2", "q2", "q3"],
            "target": ["t1", "t2", "t3", "t4", "t5"],
            "evalue": [1e-10, 1e-20, 1e-5, 1e-5, 1e-100],
            "pident": [90.0, 80.0, 95.0, 98.0, 99.0],
        }
        df = pd.DataFrame(data)

        # q1: 90.0% identity is better than 80.0%
        # q2: 98.0% identity is better than 95.0%
        # q3: only one hit (99.0%)
        filtered_df = _filter_top_hits(df)

        assert len(filtered_df) == 3
        assert filtered_df[filtered_df["query"] == "q1"].pident.iloc[0] == 90.0
        assert filtered_df[filtered_df["query"] == "q2"].pident.iloc[0] == 98.0
        assert filtered_df[filtered_df["query"] == "q3"].pident.iloc[0] == 99.0

    def test_filter_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame(columns=["query", "target", "pident", "evalue"])
        filtered_df = _filter_top_hits(df)

        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == 0

    def test_filter_single_hit_per_query(self):
        """Test filtering when each query has only one hit."""
        data = {"query": ["q1", "q2", "q3"], "target": ["t1", "t2", "t3"], "pident": [90.0, 85.0, 95.0], "evalue": [1e-10, 1e-15, 1e-20]}
        df = pd.DataFrame(data)

        filtered_df = _filter_top_hits(df)
        assert len(filtered_df) == 3

    def test_filter_ties_in_pident(self):
        """Test filtering when there are ties in percent identity."""
        data = {"query": ["q1", "q1", "q1"], "target": ["t1", "t2", "t3"], "pident": [90.0, 90.0, 85.0], "evalue": [1e-10, 1e-15, 1e-20]}
        df = pd.DataFrame(data)

        filtered_df = _filter_top_hits(df)
        # Should keep one of the 90.0% hits (first by pandas default)
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["pident"] == 90.0


class TestBuildSequenceSearchResults:
    """Tests for _build_sequence_search_results function."""

    def test_build_results_with_hits(self):
        """Test building results from DataFrame with hits."""
        sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
        sequence_ids = ["seq_0", "seq_1"]
        df = pd.DataFrame({
            "query": ["seq_0", "seq_0", "seq_1"],
            "target": ["db_1", "db_2", "db_3"],
            "pident": [95.0, 85.0, 90.0],
            "evalue": [1e-50, 1e-30, 1e-40],
        })

        results = _build_sequence_search_results(sequences, sequence_ids, df)

        assert len(results) == 2
        assert results[0].query_id == "seq_0"
        assert results[0].query_sequence == "MVLSPADKTN"
        assert results[0].num_hits == 2
        assert results[0].top_hit.pident == 95.0
        assert results[1].query_id == "seq_1"
        assert results[1].query_sequence == "MKLLVVAAAA"
        assert results[1].num_hits == 1

    def test_build_results_no_hits(self):
        """Test building results with empty DataFrame."""
        sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
        sequence_ids = ["seq_0", "seq_1"]
        df = pd.DataFrame(columns=["query", "target", "pident", "evalue"])

        results = _build_sequence_search_results(sequences, sequence_ids, df)

        assert len(results) == 2
        assert all(r.num_hits == 0 for r in results)
        assert all(r.top_hit is None for r in results)

    def test_build_results_partial_hits(self):
        """Test building results where only some sequences have hits."""
        sequences = ["SEQ1", "SEQ2", "SEQ3"]
        sequence_ids = ["seq_0", "seq_1", "seq_2"]
        df = pd.DataFrame({
            "query": ["seq_0", "seq_2"],
            "target": ["db_1", "db_2"],
            "pident": [95.0, 90.0],
            "evalue": [1e-50, 1e-40],
        })

        results = _build_sequence_search_results(sequences, sequence_ids, df)

        assert len(results) == 3
        assert results[0].num_hits == 1
        assert results[1].num_hits == 0  # No hits for seq_1
        assert results[2].num_hits == 1

    def test_build_results_with_custom_ids(self):
        """Test building results with custom sequence IDs."""
        sequences = ["MVLSPADKTN", "MKLLVVAAAA"]
        sequence_ids = ["protein_a", "protein_b"]
        df = pd.DataFrame({
            "query": ["protein_a", "protein_b"],
            "target": ["db_1", "db_2"],
            "pident": [95.0, 90.0],
            "evalue": [1e-50, 1e-40],
        })

        results = _build_sequence_search_results(sequences, sequence_ids, df)

        assert len(results) == 2
        assert results[0].query_id == "protein_a"
        assert results[1].query_id == "protein_b"



# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_search_proteins_input_validation(self, temp_dir):
        """Test MmseqsSearchProteinsInput validation."""
        db_file = temp_dir / "database.faa"
        db_file.write_text(">db1\nMVLSPADKTN\n")

        # Valid input
        inputs = MmseqsSearchProteinsInput(
            query_sequences=["MVLSPADKTN", "MKLLVVAAAA"],
            mmseqs_db=str(db_file),
        )
        assert len(inputs.query_sequences) == 2

        # Empty sequences should fail
        with pytest.raises(ValueError):
            MmseqsSearchProteinsInput(
                query_sequences=[],
                mmseqs_db=str(db_file),
            )

        # Non-list should fail
        with pytest.raises(ValueError):
            MmseqsSearchProteinsInput(
                query_sequences="single_string",
                mmseqs_db=str(db_file),
            )

    def test_search_genomes_input_validation(self):
        """Test MmseqsSearchGenomesInput validation."""
        # Valid input
        inputs = MmseqsSearchGenomesInput(
            query_genomes=["ATGC", "GCTA"],
            target_genomes=["ATGC", "GCTA"],
        )
        assert len(inputs.query_genomes) == 2
        assert len(inputs.target_genomes) == 2

        # Empty query should fail
        with pytest.raises(ValueError):
            MmseqsSearchGenomesInput(
                query_genomes=[],
                target_genomes=["ATGC"],
            )

        # Empty target should fail
        with pytest.raises(ValueError):
            MmseqsSearchGenomesInput(
                query_genomes=["ATGC"],
                target_genomes=[],
            )

    def test_clustering_input_validation(self):
        """Test MmseqsClusteringInput validation."""
        # Valid input
        inputs = MmseqsClusteringInput(
            input_sequences=["MVLSPADKTN", "MKLLVVAAAA"],
        )
        assert len(inputs.input_sequences) == 2

        # Empty sequences should fail
        with pytest.raises(ValueError):
            MmseqsClusteringInput(
                input_sequences=[],
            )


# ============================================================================
# Integration Tests (require MMseqs2 to be installed)
# ============================================================================

@pytest.mark.skip_ci
class TestMmseqs:
    """Integration tests requiring MMseqs2 installation."""

    def test_mmseqs_search_proteins_execution(self, temp_dir):
        """Test mmseqs_search_proteins execution with sequences."""
        db_file = temp_dir / "database.faa"
        db_file.write_text(">db1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n")

        inputs = MmseqsSearchProteinsInput(
            query_sequences=[
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
                "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            ],
            mmseqs_db=str(db_file),
        )
        config = MmseqsSearchProteinsConfig(threads=2)
        result = run_mmseqs_search_proteins(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, MmseqsSearchProteinsOutput)
        assert len(result) == 2
        # Each sequence should have at least one hit (itself in db)
        assert result[0].num_hits >= 1
        assert result[1].num_hits >= 1

    def test_mmseqs_search_proteins_no_hits(self, temp_dir):
        """Test mmseqs_search_proteins with no expected hits."""
        db_file = temp_dir / "database.faa"
        db_file.write_text(">db1\nAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n")

        inputs = MmseqsSearchProteinsInput(
            query_sequences=["WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW"],
            mmseqs_db=str(db_file),
        )
        config = MmseqsSearchProteinsConfig(threads=2)
        result = run_mmseqs_search_proteins(inputs, config)

        assert isinstance(result, MmseqsSearchProteinsOutput)
        assert result.success is True
        assert len(result) == 1
        # Very different sequences should have no hits
        assert result[0].num_hits == 0

    def test_mmseqs_clustering_execution(self, temp_dir):
        """Test mmseqs_clustering execution with sequences."""
        sequences = [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",  # Identical to first
            "MKLLVVAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # Different
        ]

        inputs = MmseqsClusteringInput(input_sequences=sequences)
        config = MmseqsClusteringConfig(min_seq_id=0.95)

        result = run_mmseqs_clustering(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, MmseqsClusteringOutput)
        assert len(result) == 3
        assert result.num_clusters == 2  # Two clusters: identical seqs + different one

        # First two sequences should be in same cluster
        assert result[0].cluster_id == result[1].cluster_id
        # Third should be different
        assert result[2].cluster_id != result[0].cluster_id

    def test_mmseqs_clustering_all_identical(self, temp_dir):
        """Test mmseqs_clustering with all identical sequences."""
        sequences = [
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT",
        ]

        inputs = MmseqsClusteringInput(input_sequences=sequences)
        config = MmseqsClusteringConfig(min_seq_id=0.95)

        result = run_mmseqs_clustering(inputs, config)

        assert result.num_clusters == 1
        assert all(r.cluster_id == result[0].cluster_id for r in result)
        # Only one should be representative
        assert len(result.representative_indices) == 1

    def test_mmseqs_search_genomes_execution(self, temp_dir):
        """Test mmseqs_search_genomes execution with sequence lists."""
        query_seqs = [
            "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTC",
            "ATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCC",
        ]
        target_seqs = [
            "ATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTCATGGTGCTGTCTCCTGCCGACAAGACCAACGTCAAGGCCGCCTGGGGTAAGGTC",
            "ATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCATGAAGCTGCTGGTGGTGGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCCGCC",
        ]

        inputs = MmseqsSearchGenomesInput(query_genomes=query_seqs, target_genomes=target_seqs)
        config = MmseqsSearchGenomesConfig()

        result = run_mmseqs_search_genomes(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, MmseqsSearchGenomesOutput)
        assert len(result) == 2


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.skip_ci
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_workflow_with_real_test_data(self, temp_dir):
        """Test complete workflow using test data files."""
        if not M8_FILE.exists() or not PROTEIN_FASTA.exists():
            pytest.skip("Test data files not found")

        # Test full workflow
        raw_output = M8_FILE.read_text()
        df = _parse_m8_output(raw_output)
        filtered_df = _filter_top_hits(df)

        # Verify results
        assert len(filtered_df) <= len(df)
        query_counts = filtered_df.groupby("query").size()
        assert all(count == 1 for count in query_counts)

    def test_column_names_consistency(self):
        """Test that column names match MMseqs2 format."""
        if not M8_FILE.exists():
            pytest.skip("Test data file not found")

        raw_output = M8_FILE.read_text()
        df = _parse_m8_output(raw_output)

        # Verify column names match MMseqs2 convention
        expected_columns = ["query", "target", "pident", "evalue"]
        assert list(df.columns) == expected_columns

    def test_single_sequence_protein_search(self, temp_dir):
        """Test protein search with a single sequence."""
        db_file = temp_dir / "database.faa"
        db_file.write_text(">db1\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT\n")

        inputs = MmseqsSearchProteinsInput(
            query_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"],
            mmseqs_db=str(db_file),
        )
        config = MmseqsSearchProteinsConfig(threads=2)
        result = run_mmseqs_search_proteins(inputs, config)

        assert len(result) == 1
        assert result[0].num_hits >= 1

    def test_single_sequence_clustering(self, temp_dir):
        """Test clustering with a single sequence."""
        inputs = MmseqsClusteringInput(
            input_sequences=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"],
        )
        config = MmseqsClusteringConfig(min_seq_id=0.95)
        result = run_mmseqs_clustering(inputs, config)

        assert len(result) == 1
        assert result.num_clusters == 1
        assert result[0].is_representative is True
