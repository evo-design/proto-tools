"""tests/sequence_scoring_tests/test_alphagenome.py

Tests for AlphaGenome."""
from __future__ import annotations

import pytest

from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output


_persistent_tool = make_persistent_fixture("alphagenome")


# Smallest supported context length (fastest inference for predictions).
_SHORT = 16_384
_SHORT_MID = _SHORT // 2  # 8_192 — centre of the short interval

# Scoring / ISM operations require a context wider than the scorer's centre
# mask (the default RNA_SEQ interval scorer uses width=200,001 bp).
# Use the smallest supported context that fits: 524,288 bp.
_SCORE = 524_288
_SCORE_MID = _SCORE // 2  # 262_144 — centre of the scoring interval


# ---------------------------------------------------------------------------
# Integration tests


# ── Interval Prediction ──────────────────────────────────────────────────────


@pytest.mark.include_in_env_report(category="sequence_scoring")
@pytest.mark.uses_gpu
def test_interval_prediction():
    """Test interval prediction with multiple output types."""
    from bio_programming_tools import (
        AlphaGenomeInterval,
        AlphaGenomePredictIntervalsConfig,
        AlphaGenomePredictIntervalsInput,
        run_alphagenome_predict_intervals,
    )

    inputs = AlphaGenomePredictIntervalsInput(
        intervals=AlphaGenomeInterval(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SHORT,
        ),
    )
    config = AlphaGenomePredictIntervalsConfig(
        requested_outputs=["RNA_SEQ", "ATAC"],
        organism="human",
    )

    result = run_alphagenome_predict_intervals(inputs, config)

    assert result.tool_id == "alphagenome-predict-intervals"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert output.chromosome == "chr1"
    assert output.interval_start == 0
    assert output.interval_end == _SHORT
    assert output.requested_outputs == ["RNA_SEQ", "ATAC"]
    assert "predictions" in output.result
    assert output.variant is None


# ── Variant Prediction ───────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_variant_prediction():
    """Test variant prediction returns correct metadata and predictions."""
    from bio_programming_tools import (
        AlphaGenomePredictVariantsConfig,
        AlphaGenomePredictVariantsInput,
        AlphaGenomeVariant,
        run_alphagenome_predict_variants,
    )

    inputs = AlphaGenomePredictVariantsInput(
        variants=AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SHORT,
            variant_position=_SHORT_MID,
            reference_bases="A",
            alternate_bases="G",
        ),
    )
    config = AlphaGenomePredictVariantsConfig(
        requested_outputs=["RNA_SEQ"],
        organism="human",
    )

    result = run_alphagenome_predict_variants(inputs, config)

    assert result.tool_id == "alphagenome-predict-variants"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert output.requested_outputs == ["RNA_SEQ"]
    assert "predictions" in output.result
    assert output.variant == {
        "position": _SHORT_MID,
        "reference_bases": "A",
        "alternate_bases": "G",
    }


# ── Sequence Prediction ──────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_single_sequence_prediction_via_batched_api():
    """Single-sequence prediction should run through batched API."""
    from bio_programming_tools import (
        AlphaGenomePredictSequencesConfig,
        AlphaGenomePredictSequencesInput,
        run_alphagenome_predict_sequences,
    )

    sequence = "ATCG" * (_SHORT // 4)
    inputs = AlphaGenomePredictSequencesInput(sequences=[sequence])
    config = AlphaGenomePredictSequencesConfig(
        requested_outputs=["RNA_SEQ"],
        organism="human",
    )

    result = run_alphagenome_predict_sequences(inputs, config)
    assert result.tool_id == "alphagenome-predict-sequences"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert output.chromosome == "sequence"
    assert output.interval_start == 0
    assert output.interval_end == len(sequence)
    assert output.requested_outputs == ["RNA_SEQ"]
    assert "predictions" in output.result


# ── Variant Scoring ──────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_variant_scoring():
    """Test variant scoring with default scorers."""
    from bio_programming_tools import (
        AlphaGenomeScoreVariantsConfig,
        AlphaGenomeScoreVariantsInput,
        AlphaGenomeVariant,
        run_alphagenome_score_variants,
    )

    inputs = AlphaGenomeScoreVariantsInput(
        variants=AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
            variant_position=_SCORE_MID,
            reference_bases="A",
            alternate_bases="G",
        ),
    )
    config = AlphaGenomeScoreVariantsConfig(
        variant_scorers=None,
        organism="human",
    )

    result = run_alphagenome_score_variants(inputs, config)

    assert result.tool_id == "alphagenome-score-variants"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert isinstance(output.scores, list)
    assert len(output.scores) > 0
    assert isinstance(output.scores[0], dict)


# ── Interval Scoring ─────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_interval_scoring():
    """Test interval scoring with default scorers."""
    from bio_programming_tools import (
        AlphaGenomeInterval,
        AlphaGenomeScoreIntervalsConfig,
        AlphaGenomeScoreIntervalsInput,
        run_alphagenome_score_intervals,
    )

    inputs = AlphaGenomeScoreIntervalsInput(
        intervals=AlphaGenomeInterval(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
        ),
    )
    config = AlphaGenomeScoreIntervalsConfig(
        interval_scorers=None,
        organism="human",
    )

    result = run_alphagenome_score_intervals(inputs, config)

    assert result.tool_id == "alphagenome-score-intervals"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert isinstance(output.scores, list)
    assert len(output.scores) > 0


# ── In-Silico Mutagenesis (ISM) ──────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_ism():
    """Test ISM over a small sub-interval with position-based scorers."""
    from bio_programming_tools import (
        AlphaGenomeISM,
        AlphaGenomeScoreISMConfig,
        AlphaGenomeScoreISMInput,
        run_alphagenome_score_ism_variants_batch,
    )

    inputs = AlphaGenomeScoreISMInput(
        requests=AlphaGenomeISM(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
            ism_interval_start=_SCORE_MID - 10,
            ism_interval_end=_SCORE_MID + 10,  # 20 bp window
        ),
    )
    config = AlphaGenomeScoreISMConfig(
        # Use position-based (CenterMask) scorers that don't require gene
        # annotations near the ISM window. The default (None = all recommended)
        # includes gene-based scorers that return empty for gene-poor regions.
        variant_scorers=["ATAC", "DNASE"],
        organism="human",
    )

    result = run_alphagenome_score_ism_variants_batch(inputs, config)

    assert result.tool_id == "alphagenome-score-ism-variants-batch"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert isinstance(output.scores, list)
    assert len(output.scores) > 0


@pytest.mark.uses_gpu
def test_ism_with_variant_context():
    """Test ISM with an existing variant applied as background context."""
    from bio_programming_tools import (
        AlphaGenomeISM,
        AlphaGenomeScoreISMConfig,
        AlphaGenomeScoreISMInput,
        run_alphagenome_score_ism_variants_batch,
    )

    inputs = AlphaGenomeScoreISMInput(
        requests=AlphaGenomeISM(
            chromosome="chr1",
            interval_start=0,
            interval_end=_SCORE,
            ism_interval_start=_SCORE_MID - 10,
            ism_interval_end=_SCORE_MID + 10,
            variant_position=_SCORE_MID,
            reference_bases="A",
            alternate_bases="G",
        ),
    )
    config = AlphaGenomeScoreISMConfig(
        variant_scorers=["RNA_SEQ"],
        organism="human",
    )

    result = run_alphagenome_score_ism_variants_batch(inputs, config)

    assert len(result) == 1
    validate_output(result)
    assert len(result[0].scores) > 0


# ---------------------------------------------------------------------------
# Slow tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.uses_gpu
def test_interval_prediction_131k_context():
    """Interval prediction at the 131,072 bp context length (third-smallest supported)."""
    from bio_programming_tools import (
        AlphaGenomeInterval,
        AlphaGenomePredictIntervalsConfig,
        AlphaGenomePredictIntervalsInput,
        run_alphagenome_predict_intervals,
    )

    inputs = AlphaGenomePredictIntervalsInput(
        intervals=AlphaGenomeInterval(
            chromosome="chr1",
            interval_start=0,
            interval_end=131_072,
        ),
    )
    config = AlphaGenomePredictIntervalsConfig(
        requested_outputs=["RNA_SEQ"],
        organism="human",
    )

    result = run_alphagenome_predict_intervals(inputs, config)

    assert result.tool_id == "alphagenome-predict-intervals"
    assert len(result) == 1
    validate_output(result)
    output = result[0]
    assert output.interval_end == 131_072
    assert "predictions" in output.result
