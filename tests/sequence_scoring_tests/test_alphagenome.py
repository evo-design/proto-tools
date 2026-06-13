"""tests/sequence_scoring_tests/test_alphagenome.py.

Tests for AlphaGenome.
"""

import pytest

from tests.conftest import benchmark_twice, make_persistent_fixture, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("alphagenome")


# Smallest supported context length (fastest inference for predictions).
_SHORT = 16_384
_SHORT_MID = _SHORT // 2  # 8_192, centre of the short interval

# Scoring / ISM operations require a context wider than the scorer's centre
# mask (the default RNA_SEQ interval scorer uses width=200,001 bp).
# Use the smallest supported context that fits: 524,288 bp.
_SCORE = 524_288
_SCORE_MID = _SCORE // 2  # 262_144, centre of the scoring interval


# ---------------------------------------------------------------------------
# Integration tests


# ── Interval Prediction ──────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_interval_prediction():
    """Test interval prediction with multiple output types."""
    from proto_tools import (
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
    from proto_tools import (
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
    from proto_tools import (
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
    from proto_tools import (
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
    from proto_tools import (
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
    from proto_tools import (
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
    from proto_tools import (
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
    from proto_tools import (
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


# ── Benchmarks ────────────────────────────────────────────────────────────────
# Score tools require >=524 kbp context (default RNA_SEQ scorer needs a
# 200,001 bp centre mask); predict tools use 131 kbp to keep wall time tractable.
_BENCH_PREDICT_LEN = 131_072
_BENCH_SCORE_LEN = 524_288
_BENCH_PREDICT_MID = _BENCH_PREDICT_LEN // 2
_BENCH_SCORE_MID = _BENCH_SCORE_LEN // 2
_BENCH_REQUESTED_OUTPUTS = ["RNA_SEQ", "ATAC", "DNASE"]
_BENCH_BATCH = 2


@pytest.mark.benchmark("alphagenome-predict-intervals")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphagenome_predict_intervals_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark alphagenome-predict-intervals: 2 intervals x 131,072 bp, RNA_SEQ + ATAC + DNASE (cold + warm)."""
    from proto_tools import (
        AlphaGenomeInterval,
        AlphaGenomePredictIntervalsConfig,
        AlphaGenomePredictIntervalsInput,
        run_alphagenome_predict_intervals,
    )

    intervals = [
        AlphaGenomeInterval(
            chromosome="chr1",
            interval_start=i * _BENCH_PREDICT_LEN,
            interval_end=(i + 1) * _BENCH_PREDICT_LEN,
        )
        for i in range(_BENCH_BATCH)
    ]
    inputs = AlphaGenomePredictIntervalsInput(intervals=intervals)
    config = AlphaGenomePredictIntervalsConfig(
        requested_outputs=_BENCH_REQUESTED_OUTPUTS,
        organism="human",
    )

    result = benchmark_twice(request, "alphagenome", lambda: run_alphagenome_predict_intervals(inputs, config))
    validate_output(result, check_export=False)

    assert result.tool_id == "alphagenome-predict-intervals"
    assert len(result) == _BENCH_BATCH
    for output in result.results:
        assert output.interval_end - output.interval_start == _BENCH_PREDICT_LEN
        assert "predictions" in output.result


@pytest.mark.benchmark("alphagenome-predict-sequences")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphagenome_predict_sequences_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark alphagenome-predict-sequences: 2 raw 131,072 bp sequences, RNA_SEQ + ATAC + DNASE (cold + warm)."""
    from proto_tools import (
        AlphaGenomePredictSequencesConfig,
        AlphaGenomePredictSequencesInput,
        run_alphagenome_predict_sequences,
    )

    sequences = random_dna_sequences(n=_BENCH_BATCH, length=_BENCH_PREDICT_LEN, seed=0)
    inputs = AlphaGenomePredictSequencesInput(sequences=sequences)
    config = AlphaGenomePredictSequencesConfig(
        requested_outputs=_BENCH_REQUESTED_OUTPUTS,
        organism="human",
    )

    result = benchmark_twice(request, "alphagenome", lambda: run_alphagenome_predict_sequences(inputs, config))
    validate_output(result, check_export=False)

    assert result.tool_id == "alphagenome-predict-sequences"
    assert len(result) == _BENCH_BATCH
    for output in result.results:
        assert output.interval_end - output.interval_start == _BENCH_PREDICT_LEN
        assert "predictions" in output.result


@pytest.mark.benchmark("alphagenome-predict-variants")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphagenome_predict_variants_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark alphagenome-predict-variants: 2 variants x 131,072 bp interval, RNA_SEQ + ATAC + DNASE (cold + warm)."""
    from proto_tools import (
        AlphaGenomePredictVariantsConfig,
        AlphaGenomePredictVariantsInput,
        AlphaGenomeVariant,
        run_alphagenome_predict_variants,
    )

    variants = [
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=i * _BENCH_PREDICT_LEN,
            interval_end=(i + 1) * _BENCH_PREDICT_LEN,
            variant_position=i * _BENCH_PREDICT_LEN + _BENCH_PREDICT_MID,
            reference_bases="A",
            alternate_bases="G",
        )
        for i in range(_BENCH_BATCH)
    ]
    inputs = AlphaGenomePredictVariantsInput(variants=variants)
    config = AlphaGenomePredictVariantsConfig(
        requested_outputs=_BENCH_REQUESTED_OUTPUTS,
        organism="human",
    )

    result = benchmark_twice(request, "alphagenome", lambda: run_alphagenome_predict_variants(inputs, config))
    validate_output(result, check_export=False)

    assert result.tool_id == "alphagenome-predict-variants"
    assert len(result) == _BENCH_BATCH
    for output in result.results:
        assert "predictions" in output.result


@pytest.mark.benchmark("alphagenome-score-variants")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphagenome_score_variants_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark alphagenome-score-variants: 2 variants x 524,288 bp, default recommended scorers (cold + warm)."""
    from proto_tools import (
        AlphaGenomeScoreVariantsConfig,
        AlphaGenomeScoreVariantsInput,
        AlphaGenomeVariant,
        run_alphagenome_score_variants,
    )

    variants = [
        AlphaGenomeVariant(
            chromosome="chr1",
            interval_start=i * _BENCH_SCORE_LEN,
            interval_end=(i + 1) * _BENCH_SCORE_LEN,
            variant_position=i * _BENCH_SCORE_LEN + _BENCH_SCORE_MID,
            reference_bases="A",
            alternate_bases="G",
        )
        for i in range(_BENCH_BATCH)
    ]
    inputs = AlphaGenomeScoreVariantsInput(variants=variants)
    config = AlphaGenomeScoreVariantsConfig(
        variant_scorers=None,  # all recommended scorers
        organism="human",
    )

    result = benchmark_twice(request, "alphagenome", lambda: run_alphagenome_score_variants(inputs, config))
    validate_output(result)

    assert result.tool_id == "alphagenome-score-variants"
    assert len(result) == _BENCH_BATCH
    for output in result.results:
        assert isinstance(output.scores, list)
        assert len(output.scores) > 0


@pytest.mark.benchmark("alphagenome-score-intervals")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphagenome_score_intervals_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark alphagenome-score-intervals: 2 intervals x 524,288 bp, default recommended scorers (cold + warm)."""
    from proto_tools import (
        AlphaGenomeInterval,
        AlphaGenomeScoreIntervalsConfig,
        AlphaGenomeScoreIntervalsInput,
        run_alphagenome_score_intervals,
    )

    intervals = [
        AlphaGenomeInterval(
            chromosome="chr1",
            interval_start=i * _BENCH_SCORE_LEN,
            interval_end=(i + 1) * _BENCH_SCORE_LEN,
        )
        for i in range(_BENCH_BATCH)
    ]
    inputs = AlphaGenomeScoreIntervalsInput(intervals=intervals)
    config = AlphaGenomeScoreIntervalsConfig(
        interval_scorers=None,  # all recommended scorers
        organism="human",
    )

    result = benchmark_twice(request, "alphagenome", lambda: run_alphagenome_score_intervals(inputs, config))
    validate_output(result)

    assert result.tool_id == "alphagenome-score-intervals"
    assert len(result) == _BENCH_BATCH
    for output in result.results:
        assert isinstance(output.scores, list)
        assert len(output.scores) > 0


@pytest.mark.benchmark("alphagenome-score-ism-variants-batch")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_alphagenome_score_ism_variants_batch_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark alphagenome-score-ism-variants-batch: 50-bp ISM window x 2 position-based scorers over 524 kbp (cold + warm)."""
    from proto_tools import (
        AlphaGenomeISM,
        AlphaGenomeScoreISMConfig,
        AlphaGenomeScoreISMInput,
        run_alphagenome_score_ism_variants_batch,
    )

    inputs = AlphaGenomeScoreISMInput(
        requests=AlphaGenomeISM(
            chromosome="chr1",
            interval_start=0,
            interval_end=_BENCH_SCORE_LEN,
            ism_interval_start=_BENCH_SCORE_MID - 25,
            ism_interval_end=_BENCH_SCORE_MID + 25,  # 50 bp window
        ),
    )
    config = AlphaGenomeScoreISMConfig(
        variant_scorers=["ATAC", "DNASE"],
        organism="human",
    )

    result = benchmark_twice(request, "alphagenome", lambda: run_alphagenome_score_ism_variants_batch(inputs, config))
    validate_output(result)

    assert result.tool_id == "alphagenome-score-ism-variants-batch"
    assert len(result) == 1
    assert isinstance(result[0].scores, list)
    assert len(result[0].scores) > 0
