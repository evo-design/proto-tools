"""
Tests for Prodigal ORF prediction tool.

Comprehensive tests for the registry-based interface, configuration validation,
input validation, and gene prediction functionality.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import ValidationError

from bio_programming.bio_tools.tools.orf_prediction import (
    ORF,
    ProdigalConfig,
    ProdigalInput,
    ProdigalOutput,
    run_prodigal_prediction,
)
from bio_programming.bio_tools.tools.infra.tool_cache import ToolCache, _program_tool_cache
from bio_programming.bio_tools.tools.tool_registry import ToolRegistry
from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output


class TestProdigalInput:
    """Test ProdigalInput validation and normalization."""

    def test_input_validation_invalid_dna_characters(self):
        """Test that invalid DNA characters are rejected."""
        with pytest.raises(ValidationError, match="Invalid DNA characters"):
            ProdigalInput(input_sequences="ATGCEFGHIJK")

    def test_input_validation_protein_sequence_rejected(self):
        """Test that protein sequence is rejected (contains invalid chars like E, F, I, P)."""
        protein_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTT"
        with pytest.raises(ValidationError, match="Invalid DNA characters"):
            ProdigalInput(input_sequences=protein_seq)

    def test_input_accepts_valid_dna(self):
        """Test that valid DNA sequence is accepted."""
        dna_seq = "ATGCGTAAATAG"
        inputs = ProdigalInput(input_sequences=dna_seq)
        assert inputs.input_sequences == [dna_seq]

    def test_input_accepts_lowercase_dna(self):
        """Test that lowercase DNA is converted to uppercase."""
        dna_seq = "atgcgtaaatag"
        inputs = ProdigalInput(input_sequences=dna_seq)
        assert inputs.input_sequences == ["ATGCGTAAATAG"]

    def test_input_accepts_mixed_case(self):
        """Test that mixed case DNA is converted to uppercase."""
        dna_seq = "AtGcGtAaAtAg"
        inputs = ProdigalInput(input_sequences=dna_seq)
        assert inputs.input_sequences == ["ATGCGTAAATAG"]

    def test_input_normalizes_single_string_to_list(self):
        """Test that single string input is normalized to a list."""
        inputs = ProdigalInput(input_sequences="ATGCGT")
        assert isinstance(inputs.input_sequences, list)
        assert len(inputs.input_sequences) == 1
        assert inputs.input_sequences[0] == "ATGCGT"

    def test_input_accepts_list_of_sequences(self):
        """Test that list of sequences is accepted."""
        sequences = ["ATGCGTAAATAG", "ATGGCATAA"]
        inputs = ProdigalInput(input_sequences=sequences)
        assert inputs.input_sequences == sequences

    def test_input_validates_each_sequence_in_list(self):
        """Test that each sequence in list is validated."""
        sequences = ["ATGCGT", "ATGEFG"]  # Second has invalid chars
        with pytest.raises(ValidationError, match="Invalid DNA characters"):
            ProdigalInput(input_sequences=sequences)

    def test_input_converts_all_sequences_to_uppercase(self):
        """Test that all sequences in list are converted to uppercase."""
        sequences = ["atgcgt", "gcataa"]
        inputs = ProdigalInput(input_sequences=sequences)
        assert inputs.input_sequences == ["ATGCGT", "GCATAA"]


class TestProdigalPrediction:
    """Test Prodigal gene prediction functionality."""

    def test_simple_gene_prediction(self):
        """Test prediction on a simple sequence."""
        sequence = "ATGCGTAAATAA"
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)
        assert result.num_orfs >= 0
        assert len(result.predicted_orfs) == 1

    def test_realistic_gene_prediction(self):
        """Test prediction on a more realistic sequence."""
        sequence = (
            "ATGAAACGTGAATTAGCAGCAGGTATCGATGCAGGTAAACGTGAATTAGCA"
            "GCAGGTATCGATGCAGGTAAACGTGAATTAGCAGCAGGTATCGATGCAGGT"
            "AAACGTGAATTAGCAGCAGGTATCGATGCAGGTAAACGTGAATTAGCAGCA"
            "GGTATCGATGCAGGTTAA"
        )
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)
        assert result.num_orfs >= 0

    def test_prediction_with_meta_mode_false(self):
        """Test prediction with single-genome mode."""
        # Single-genome mode requires longer sequence (100kb+) for training
        sequence = "ATGCGTAAATAA" * 8400  # ~100.8kb
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig(meta_mode=False)
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)

    def test_prediction_with_closed_ends(self):
        """Test prediction with closed ends (complete genome)."""
        sequence = "ATGCGTAAATAA" * 100
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig(closed_ends=True)
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)

    def test_prediction_with_custom_translation_table(self):
        """Test prediction with custom translation table."""
        sequence = "ATGCGTAAATAA" * 100
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig(translation_table=4)
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)

    def test_empty_prediction(self):
        """Test prediction on sequence with no genes."""
        sequence = "AAAAAAAAAA"
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)
        assert result.num_orfs >= 0

    def test_batch_prediction_multiple_sequences(self):
        """Test prediction on multiple sequences."""
        sequences = [
            "ATGCGTAAATAA" * 50,
            "ATGGCATAA" * 50,
            "ATGAAACGT" * 50,
        ]
        inputs = ProdigalInput(input_sequences=sequences)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        assert isinstance(result, ProdigalOutput)
        assert len(result.predicted_orfs) == 3
        # Verify num_orfs matches sum of all sequence results
        total_orfs = sum(len(orfs) for orfs in result.predicted_orfs)
        assert total_orfs == result.num_orfs

    def test_batch_prediction_parallel_processing(self):
        """Test that parallel processing works with multiple threads."""
        sequences = [
            "ATGCGTAAATAA" * 50,
            "ATGGCATAA" * 50,
        ]
        inputs = ProdigalInput(input_sequences=sequences)
        config = ProdigalConfig(num_threads=2)
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        assert isinstance(result, ProdigalOutput)
        assert len(result.predicted_orfs) == 2


class TestProdigalOrfStructure:
    """Test the structure of ProdigalOrf results."""

    def test_results_structure(self):
        """Test that results have expected fields and DataFrame columns when genes are found."""
        sequence = "ATGCGTAAATAA" * 50
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        if result.num_orfs > 0 and result.predicted_orfs:
            orf = result.predicted_orfs[0][0]
            assert isinstance(orf, ORF)

            # Check all expected fields exist
            expected_fields = [
                'parent_id', 'orf_id', 'amino_acid_sequence', 'nucleotide_sequence',
                'amino_acid_length', 'nucleotide_length', 'nucleotide_start', 'nucleotide_end',
                'strand', 'frame', 'gc_content', 'start_type', 'rbs_motif',
                'partial_begin', 'partial_end'
            ]
            for field in expected_fields:
                assert hasattr(orf, field), f"Missing field: {field}"

            # Check data types
            assert isinstance(orf.amino_acid_length, int)
            assert isinstance(orf.nucleotide_length, int)
            assert isinstance(orf.nucleotide_start, int)
            assert isinstance(orf.nucleotide_end, int)
            assert isinstance(orf.gc_content, float)

            # Check DataFrame structure
            df = result.results_df
            assert isinstance(df, pd.DataFrame)
            essential_columns = [
                'parent_id', 'orf_id', 'amino_acid_sequence', 'nucleotide_sequence',
                'amino_acid_length', 'nucleotide_length',
                'nucleotide_start', 'nucleotide_end', 'strand', 'frame',
                'gc_content', 'start_type', 'partial_begin', 'partial_end'
            ]
            for col in essential_columns:
                assert col in df.columns, f"Missing column: {col}"

    def test_gene_id_format(self):
        """Test that gene IDs follow expected format."""
        sequence = "ATGCGTAAATAA" * 50
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        if result.num_orfs > 0:
            orf = result.predicted_orfs[0][0]
            assert orf.parent_id.startswith('seq_')
            assert 'gene_' in orf.orf_id

    def test_strand_format(self):
        """Test that strand is either + or -."""
        sequence = "ATGCGTAAATAA" * 50
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        if result.num_orfs > 0:
            for orf in result.predicted_orfs[0]:
                assert orf.strand in ['+', '-']

    def test_protein_sequence_format(self):
        """Test that protein sequences are valid."""
        sequence = "ATGCGTAAATAA" * 50
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        if result.num_orfs > 0:
            for orf in result.predicted_orfs[0]:
                assert isinstance(orf.amino_acid_sequence, str)
                assert len(orf.amino_acid_sequence) > 0
                # Protein sequences should not end with stop codon (*)
                assert not orf.amino_acid_sequence.endswith('*')


class TestProdigalRegistration:
    """Test Prodigal tool registration."""

    def test_tool_is_registered(self):
        """Test that Prodigal is registered in ToolRegistry."""
        all_tools = ToolRegistry.list_all()
        assert "prodigal-prediction" in {spec.key for spec in all_tools}

    def test_tool_metadata(self):
        """Test that registered tool has correct metadata."""
        all_tools = ToolRegistry.list_all()
        tools_dict = {spec.key: spec for spec in all_tools}
        prodigal_spec = tools_dict["prodigal-prediction"]

        assert "Prokaryotic" in prodigal_spec.description or "ORF" in prodigal_spec.description

    def test_tool_schema_generation(self):
        """Test that JSON schema is generated correctly."""
        schema = ToolRegistry.get_schema("prodigal-prediction")
        assert "properties" in schema
        # Config fields should be in schema
        assert "meta_mode" in schema["properties"]
        assert "translation_table" in schema["properties"]
        assert "closed_ends" in schema["properties"]


class TestProdigalIntegration:
    """Integration tests for Prodigal tool."""

    def test_end_to_end_prediction(self):
        """Test complete workflow from config to results."""
        # Realistic sequence fragment
        sequence = (
            "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGG"
            "GAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTC"
            "GCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAG"
            "TTGCGCAGCCTGAATGGCGAATGGCGCTTTGCCTGGTTTCCGGCACCAGAA"
            "GCGGTGCCGGAAAGCTGGCTGGAGTGCGATCTTCCTGAGGCCGATACTGTC"
            "GTCGTCCCCTCAAACTGGCAGATGCACGGTTACGATGCGCCCATCTACACC"
            "AACGTGACCTATCCCATTACGGTCAATCCGCCGTTTGTTCCCACGGAGAAT"
            "CCGACGGGTTGTTACTCGCTCACATTTAATGTTGATGAAAGCTGGCTACAG"
            "GAAGGCCAGACGCGAATTATTTTTGATGGCGTTAACTCGGCGTTTCATCTG"
            "TGGTGCAACGGGCGCTGGGTCGGTTACGGCCAGGACAGTCGTTTGCCGTCT"
            "TAA"
        )

        # Run prediction
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig(meta_mode=True)
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)

        # Validate results
        assert isinstance(result, ProdigalOutput)

        # If genes found, validate structure
        if result.num_orfs > 0 and result.predicted_orfs:
            orfs = result.predicted_orfs[0]

            # Check first gene
            orf = orfs[0]
            assert orf.parent_id.startswith('seq_')
            assert isinstance(orf.amino_acid_sequence, str)
            assert orf.amino_acid_length > 0
            assert orf.strand in ['+', '-']

    def test_comparison_with_direct_pyrodigal(self):
        """Test that wrapper produces same results as direct pyrodigal usage."""
        try:
            import pyrodigal
        except ImportError:
            pytest.skip("pyrodigal not available in base environment")

        sequence = "ATGCGTAAATAA" * 50

        # Our wrapper
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig(meta_mode=True)
        our_result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(our_result)

        # Direct pyrodigal
        gene_finder = pyrodigal.GeneFinder(meta=True)
        direct_genes = gene_finder.find_genes(sequence.encode('utf-8'))

        # Compare gene counts
        assert our_result.num_orfs == len(direct_genes)

    def test_batch_processing_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        sequences = [
            "ATGCGTAAATAA" * 50,
            "ATGGCATAA" * 50,
        ]

        # Batch processing
        inputs_batch = ProdigalInput(input_sequences=sequences)
        config = ProdigalConfig()
        result_batch = run_prodigal_prediction(inputs_batch, config)

        # Individual processing
        results_individual = []
        for seq in sequences:
            inputs_single = ProdigalInput(input_sequences=seq)
            result_single = run_prodigal_prediction(inputs_single, config)
            results_individual.append(result_single)

        # Compare results
        assert result_batch.num_orfs == sum(r.num_orfs for r in results_individual)
        assert len(result_batch.predicted_orfs) == len(sequences)

        for i, (batch_orfs, single_result) in enumerate(zip(result_batch.predicted_orfs, results_individual)):
            assert len(batch_orfs) == single_result.num_orfs

    def test_lowercase_input_produces_valid_output(self):
        """Test that lowercase input is handled correctly and produces valid results."""
        sequence = "atgcgtaaataa" * 50
        inputs = ProdigalInput(input_sequences=sequence)
        config = ProdigalConfig()
        result = run_prodigal_prediction(inputs, config)

        # Validate output and export functionality
        validate_output(result)
        assert result.num_orfs >= 0


class TestProdigalCaching:
    """Test caching behavior for Prodigal prediction."""

    def test_caching_behavior(self):
        """Test that results are cached and computation is skipped on second call."""
        # Setup cache
        cache = ToolCache()
        token = _program_tool_cache.set(cache)

        try:
            # Create a sequence long enough to be interesting but fast
            sequence = "ATGCGTAAATAA" * 50
            inputs = ProdigalInput(input_sequences=sequence)
            config = ProdigalConfig()

            # Patch the subprocess call to verify it's invoked only when not cached.
            # We use the real method as side_effect so real logic still runs.
            from bio_programming.bio_tools.tools.infra.env_manager import EnvManager

            real_call = EnvManager.call_standalone_script_in_venv

            with patch.object(EnvManager, "call_standalone_script_in_venv", side_effect=real_call, autospec=True) as mock_call:
                # First call - should Compute
                result1 = run_prodigal_prediction(inputs, config)
                assert result1.success is True
                assert mock_call.call_count == 1

                # Capture result for comparison
                orfs1 = result1.predicted_orfs[0]

                # Second call - same inputs - should Cache Hit
                # call_standalone_script_in_venv should NOT be called again
                result2 = run_prodigal_prediction(inputs, config)
                assert result2.success is True
                assert mock_call.call_count == 1  # Still 1

                # Verify results are identical (compare ORF counts and first ORF)
                orfs2 = result2.predicted_orfs[0]
                assert len(orfs1) == len(orfs2)
                if orfs1:
                    assert orfs1[0].amino_acid_sequence == orfs2[0].amino_acid_sequence

                # Third call - different inputs - should Compute
                inputs_diff = ProdigalInput(input_sequences=sequence + "ATGC")
                result3 = run_prodigal_prediction(inputs_diff, config)
                assert result3.success is True
                # Should be called again
                assert mock_call.call_count == 2

        finally:
            _program_tool_cache.reset(token)
