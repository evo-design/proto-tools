"""
Tests for the MinCED CRISPR array detection tool.

Unit tests for data models and input normalization, plus integration
tests (skip_ci) for actual MinCED execution.
"""

from __future__ import annotations

import pytest

from bio_programming_tools.tools.gene_annotation.minced import (
    CrisprArray,
    CrisprRepeatSpacer,
    MincedConfig,
    MincedInput,
    MincedOutput,
    MincedSequenceResult,
    run_minced,
)
from tests.tool_infra_tests.test_export_functionality import (
    validate_export_output,
)


# ============================================================================
# Data Model Tests
# ============================================================================
class TestMincedInput:
    """Tests for MincedInput validation and normalization."""

    def test_single_sequence_normalization(self):
        """Single string should be normalized to list."""
        inp = MincedInput(sequences="ATCGATCG")
        assert isinstance(inp.sequences, list)
        assert len(inp.sequences) == 1
        assert inp.sequences[0] == "ATCGATCG"

    def test_list_of_sequences(self):
        """List of sequences should be preserved."""
        inp = MincedInput(sequences=["ATCG", "GCTA"])
        assert len(inp.sequences) == 2

    def test_optional_sequence_ids(self):
        """Sequence IDs should default to None."""
        inp = MincedInput(sequences=["ATCG"])
        assert inp.sequence_ids is None

    def test_custom_sequence_ids(self):
        """Custom sequence IDs should be preserved."""
        inp = MincedInput(sequences=["ATCG"], sequence_ids=["my_seq"])
        assert inp.sequence_ids == ["my_seq"]


class TestMincedConfig:
    """Tests for MincedConfig defaults and validation."""

    def test_defaults(self):
        """Default config values should be correct."""
        config = MincedConfig()
        assert config.min_num_repeats == 3
        assert config.min_repeat_length == 27

    def test_custom_values(self):
        """Custom config values should be accepted."""
        config = MincedConfig(min_num_repeats=4, min_repeat_length=30)
        assert config.min_num_repeats == 4
        assert config.min_repeat_length == 30

    def test_min_num_repeats_validation(self):
        """min_num_repeats must be >= 2."""
        with pytest.raises(Exception):
            MincedConfig(min_num_repeats=1)

    def test_min_repeat_length_validation(self):
        """min_repeat_length must be >= 10."""
        with pytest.raises(Exception):
            MincedConfig(min_repeat_length=5)


# ============================================================================
# Data Model Properties
# ============================================================================
class TestCrisprArray:
    """Tests for CrisprArray model."""

    def test_num_repeats(self):
        """num_repeats should count repeat-spacer units."""
        rs1 = CrisprRepeatSpacer(position=0, repeat="ATCG", spacer="GCTA")
        rs2 = CrisprRepeatSpacer(position=50, repeat="ATCG", spacer="TTAA")
        rs3 = CrisprRepeatSpacer(position=100, repeat="ATCG")
        array = CrisprArray(repeats_and_spacers=[rs1, rs2, rs3])
        assert array.num_repeats == 3

    def test_spacers_property(self):
        """spacers should extract non-None spacer sequences."""
        rs1 = CrisprRepeatSpacer(position=0, repeat="ATCG", spacer="GCTA")
        rs2 = CrisprRepeatSpacer(position=50, repeat="ATCG", spacer="TTAA")
        rs3 = CrisprRepeatSpacer(position=100, repeat="ATCG")  # Last repeat, no spacer
        array = CrisprArray(repeats_and_spacers=[rs1, rs2, rs3])
        spacers = array.spacers
        assert len(spacers) == 2
        assert "GCTA" in spacers
        assert "TTAA" in spacers

    def test_empty_array(self):
        """Empty array should have 0 repeats and no spacers."""
        array = CrisprArray()
        assert array.num_repeats == 0
        assert array.spacers == []


class TestMincedSequenceResult:
    """Tests for MincedSequenceResult properties."""

    def test_has_crispr_with_arrays(self):
        """has_crispr should be True when arrays exist."""
        rs = CrisprRepeatSpacer(position=0, repeat="ATCG")
        array = CrisprArray(repeats_and_spacers=[rs])
        result = MincedSequenceResult(
            sequence_id="test", crispr_arrays=[array]
        )
        assert result.has_crispr is True
        assert result.num_arrays == 1

    def test_has_crispr_without_arrays(self):
        """has_crispr should be False when no arrays exist."""
        result = MincedSequenceResult(sequence_id="test", crispr_arrays=[])
        assert result.has_crispr is False
        assert result.num_arrays == 0


class TestMincedOutput:
    """Tests for MincedOutput properties."""

    def test_num_sequences_with_crispr(self):
        """num_sequences_with_crispr should count correctly."""
        rs = CrisprRepeatSpacer(position=0, repeat="ATCG")
        array = CrisprArray(repeats_and_spacers=[rs])

        r1 = MincedSequenceResult(sequence_id="seq1", crispr_arrays=[array])
        r2 = MincedSequenceResult(sequence_id="seq2", crispr_arrays=[])
        r3 = MincedSequenceResult(sequence_id="seq3", crispr_arrays=[array])

        output = MincedOutput(results=[r1, r2, r3])
        assert output.num_sequences_with_crispr == 2

    def test_empty_output(self):
        """Empty output should have 0 sequences with CRISPR."""
        output = MincedOutput(results=[])
        assert output.num_sequences_with_crispr == 0


# ============================================================================
# Export Tests
# ============================================================================
class TestMincedExport:
    """Tests for MincedOutput export functionality."""

    @pytest.fixture
    def sample_output(self):
        rs1 = CrisprRepeatSpacer(
            position=100, repeat="ATCGATCG", spacer="GCTAGCTA",
            repeat_length=8, spacer_length=8,
        )
        rs2 = CrisprRepeatSpacer(
            position=200, repeat="ATCGATCG",
            repeat_length=8,
        )
        array = CrisprArray(repeats_and_spacers=[rs1, rs2])
        result = MincedSequenceResult(
            sequence_id="test_seq", crispr_arrays=[array]
        )
        return MincedOutput(results=[result])

    def test_export_csv(self, sample_output, tmp_path):
        """Export to CSV format."""
        sample_output.export(
            name="minced", export_path=str(tmp_path), file_format="csv"
        )
        csv_path = tmp_path / "minced.csv"
        assert validate_export_output(csv_path)

    def test_export_json(self, sample_output, tmp_path):
        """Export to JSON format."""
        sample_output.export(
            name="minced", export_path=str(tmp_path), file_format="json"
        )
        json_path = tmp_path / "minced.json"
        assert validate_export_output(json_path)

    def test_output_format_options(self, sample_output):
        """Check supported output formats."""
        assert "csv" in sample_output.output_format_options
        assert "json" in sample_output.output_format_options
        assert sample_output.output_format_default == "json"


# ============================================================================
# Integration Tests (require minced binary)
# ============================================================================
class TestMincedIntegration:
    """Integration tests that require the MinCED binary."""

    # Known CRISPR-containing sequence fragment (synthetic, for testing)
    CRISPR_SEQUENCE = (
        "ATCGATCGATCGATCGATCGATCGATCG"  # Leader
        + (
            "GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC"  # Repeat (36nt)
            + "AAAAAAAACCCCCCCCTTTTTTTTGGGGGGGG"      # Spacer (31nt)
        ) * 5  # 5 repeat-spacer units
        + "GTTTTAGAGCTATGCTGTTTTGAATGGTCCCAAAAC"  # Final repeat
        + "ATCGATCGATCGATCGATCGATCGATCG"  # Trailer
    )

    @pytest.mark.skip_ci
    def test_run_minced_with_crispr_sequence(self):
        """Run MinCED on a sequence with known CRISPR arrays."""
        inputs = MincedInput(
            sequences=[self.CRISPR_SEQUENCE],
            sequence_ids=["crispr_test"],
        )
        config = MincedConfig(min_num_repeats=3, min_repeat_length=23)
        result = run_minced(inputs, config)

        assert isinstance(result, MincedOutput)
        assert len(result.results) == 1

    @pytest.mark.skip_ci
    def test_run_minced_no_crispr(self):
        """Run MinCED on a sequence without CRISPR arrays."""
        random_seq = "ATCGATCG" * 500
        inputs = MincedInput(sequences=[random_seq])
        config = MincedConfig()
        result = run_minced(inputs, config)

        assert isinstance(result, MincedOutput)
        assert len(result.results) == 1
        assert result.num_sequences_with_crispr == 0
