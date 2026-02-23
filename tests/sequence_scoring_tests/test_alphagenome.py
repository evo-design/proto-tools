"""
test_alphagenome.py

Tests for AlphaGenome open-weights sequence scoring tools.
"""
from __future__ import annotations

import pytest

from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output


_persistent_tool = make_persistent_fixture("alphagenome")


# Smallest supported context length (fastest inference for predictions).
_SHORT = 16_384
_SHORT_MID = _SHORT // 2  # 8_192 – centre of the short interval

# Scoring / ISM operations require a context wider than the scorer's centre
# mask (the default RNA_SEQ interval scorer uses width=200,001 bp).
# Use the smallest supported context that fits: 524,288 bp.
_SCORE = 524_288
_SCORE_MID = _SCORE // 2  # 262_144 – centre of the scoring interval


@pytest.mark.slow
@pytest.mark.uses_gpu
class TestAlphaGenome:
    """GPU tests for AlphaGenome sequence scoring tools."""

    # --- Interval Prediction ---

    @pytest.mark.include_in_env_report(category="sequence_scoring")
    def test_interval_prediction(self):
        """Test interval prediction with multiple output types."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomePredictIntervalConfig,
            AlphaGenomePredictIntervalInput,
            run_alphagenome_predict_interval,
        )

        inputs = AlphaGenomePredictIntervalInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SHORT,
        )
        config = AlphaGenomePredictIntervalConfig(
            requested_outputs=["RNA_SEQ", "ATAC"],
            organism="human",
        )

        result = run_alphagenome_predict_interval(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-predict-interval"
        assert result.chromosome == "chr1"
        assert result.interval_start == 0
        assert result.interval_end == _SHORT
        assert result.requested_outputs == ["RNA_SEQ", "ATAC"]
        assert "predictions" in result.result
        assert result.variant is None

    # --- Variant Prediction ---

    def test_variant_prediction(self):
        """Test variant prediction returns correct metadata and predictions."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomePredictVariantConfig,
            AlphaGenomePredictVariantInput,
            run_alphagenome_predict_variant,
        )

        inputs = AlphaGenomePredictVariantInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SHORT,
            variant_position=_SHORT_MID,
            reference_bases="A",
            alternate_bases="G",
        )
        config = AlphaGenomePredictVariantConfig(
            requested_outputs=["RNA_SEQ"],
            organism="human",
        )

        result = run_alphagenome_predict_variant(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-predict-variant"
        assert result.requested_outputs == ["RNA_SEQ"]
        assert "predictions" in result.result
        assert result.variant == {
            "position": _SHORT_MID,
            "reference_bases": "A",
            "alternate_bases": "G",
        }

    # --- Sequence Prediction ---

    def test_sequence_prediction(self):
        """Test prediction from a raw DNA sequence string."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomePredictSequenceConfig,
            AlphaGenomePredictSequenceInput,
            run_alphagenome_predict_sequence,
        )

        sequence = "ATCG" * (_SHORT // 4)  # 2,048 bp
        inputs = AlphaGenomePredictSequenceInput(sequence=sequence)
        config = AlphaGenomePredictSequenceConfig(
            requested_outputs=["RNA_SEQ"],
            organism="human",
        )

        result = run_alphagenome_predict_sequence(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-predict-sequence"
        assert result.chromosome == "sequence"
        assert result.interval_start == 0
        assert result.interval_end == len(sequence)
        assert result.requested_outputs == ["RNA_SEQ"]
        assert "predictions" in result.result

    # --- Variant Scoring ---

    def test_variant_scoring(self):
        """Test variant scoring with default scorers."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomeScoreVariantConfig,
            AlphaGenomeScoreVariantInput,
            run_alphagenome_score_variant,
        )

        inputs = AlphaGenomeScoreVariantInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
            variant_position=_SCORE_MID,
            reference_bases="A",
            alternate_bases="G",
        )
        config = AlphaGenomeScoreVariantConfig(
            variant_scorers=None,
            organism="human",
        )

        result = run_alphagenome_score_variant(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-score-variant"
        assert isinstance(result.scores, list)
        assert len(result.scores) > 0
        assert isinstance(result.scores[0], dict)

    # --- Interval Scoring ---

    def test_interval_scoring(self):
        """Test interval scoring with default scorers."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomeScoreIntervalConfig,
            AlphaGenomeScoreIntervalInput,
            run_alphagenome_score_interval,
        )

        inputs = AlphaGenomeScoreIntervalInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
        )
        config = AlphaGenomeScoreIntervalConfig(
            interval_scorers=None,
            organism="human",
        )

        result = run_alphagenome_score_interval(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-score-interval"
        assert isinstance(result.scores, list)
        assert len(result.scores) > 0

    # --- In-Silico Mutagenesis (ISM) ---

    def test_ism(self):
        """Test ISM over a small sub-interval with position-based scorers."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomeScoreISMConfig,
            AlphaGenomeScoreISMInput,
            run_alphagenome_score_ism_variants,
        )

        inputs = AlphaGenomeScoreISMInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
            ism_interval_start=_SCORE_MID - 10,
            ism_interval_end=_SCORE_MID + 10,  # 20 bp window
        )
        config = AlphaGenomeScoreISMConfig(
            # Use position-based (CenterMask) scorers that don't require gene
            # annotations near the ISM window. The default (None = all recommended)
            # includes gene-based scorers that return empty for gene-poor regions.
            variant_scorers=["ATAC", "DNASE"],
            organism="human",
        )

        result = run_alphagenome_score_ism_variants(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-score-ism-variants"
        assert isinstance(result.scores, list)
        assert len(result.scores) > 0

    def test_ism_with_variant_context(self):
        """Test ISM with an existing variant applied as background context."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomeScoreISMConfig,
            AlphaGenomeScoreISMInput,
            run_alphagenome_score_ism_variants,
        )

        inputs = AlphaGenomeScoreISMInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
            ism_interval_start=_SCORE_MID - 10,
            ism_interval_end=_SCORE_MID + 10,
            variant_position=_SCORE_MID,
            reference_bases="A",
            alternate_bases="G",
        )
        config = AlphaGenomeScoreISMConfig(
            variant_scorers=["RNA_SEQ"],
            organism="human",
        )

        result = run_alphagenome_score_ism_variants(inputs, config)

        validate_output(result)

        assert len(result.scores) > 0

    # --- Stress Test (131k context) ---

    def test_full_context_interval_prediction(self):
        """Stress test: interval prediction at the 131k context length."""
        from bio_programming_tools.tools.sequence_scoring.alphagenome import (
            AlphaGenomePredictIntervalConfig,
            AlphaGenomePredictIntervalInput,
            run_alphagenome_predict_interval,
        )

        inputs = AlphaGenomePredictIntervalInput(
            chromosome="chr1",
            interval_start=0,
            interval_end=131_072,
        )
        config = AlphaGenomePredictIntervalConfig(
            requested_outputs=["RNA_SEQ"],
            organism="human",
        )

        result = run_alphagenome_predict_interval(inputs, config)

        validate_output(result)

        assert result.tool_id == "alphagenome-predict-interval"
        assert result.interval_end == 131_072
        assert "predictions" in result.result
