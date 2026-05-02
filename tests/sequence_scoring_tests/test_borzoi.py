"""tests/sequence_scoring_tests/test_borzoi.py.

Tests for Borzoi regulatory activity prediction tool.
"""

import random

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_scoring.borzoi import (
    BORZOI_CONTEXT,
    BORZOI_OUTPUT,
    BORZOI_OUTPUT_FLANK,
    BORZOI_OUTPUT_RESOLUTION,
)
from tests.conftest import benchmark_twice, make_persistent_fixture, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("borzoi")


def _generate_random_dna_sequence(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence of given length."""
    random.seed(seed)
    return "".join(random.choices("ACGT", k=length))


# -- Input validation ------------------------------------------------------------------


def test_borzoi_input_valid():
    """Test valid Borzoi input is accepted."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiInput

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT)
    inputs = BorzoiInput(sequences=sequence)
    assert inputs.sequences == [sequence]
    assert len(inputs.sequences[0]) == BORZOI_CONTEXT


def test_borzoi_input_accepts_sequence_batches():
    """Borzoi input should normalize sequence batches to a list."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiInput

    seq_a = _generate_random_dna_sequence(BORZOI_CONTEXT)
    seq_b = _generate_random_dna_sequence(BORZOI_CONTEXT, seed=123)

    inputs = BorzoiInput(sequences=[seq_a, seq_b])

    assert inputs.sequences == [seq_a, seq_b]
    assert len(inputs) == 2


def test_borzoi_input_rejects_wrong_length():
    """Test that sequences with invalid length are rejected."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiInput

    # Too short
    with pytest.raises(ValueError, match=f"must have length {BORZOI_CONTEXT}"):
        BorzoiInput(sequences="ATCG" * 100)

    # Too long
    with pytest.raises(ValueError, match=f"must have length {BORZOI_CONTEXT}"):
        BorzoiInput(sequences="ATCG" * 200000)


def test_borzoi_input_accepts_target_aligned_sequences():
    """Target coordinates let Borzoi extract a model window from a larger construct."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiInput, SequenceTargetRange

    sequence = ("C" * 100) + ("A" * BORZOI_CONTEXT) + ("G" * 100)
    target_start = 100 + BORZOI_OUTPUT_FLANK

    target_range = SequenceTargetRange(start=target_start, end=target_start + 10)
    inputs = BorzoiInput(sequences=sequence, target_ranges=target_range)

    assert inputs.sequences == [sequence]
    assert inputs.target_ranges == [target_range]


def test_borzoi_run_extracts_target_aligned_window(monkeypatch):
    """run_borzoi should dispatch exact-context windows and return source coordinates."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        SequenceTargetRange,
        borzoi_prediction,
        run_borzoi,
    )

    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_payloads.append(payload)
        return {"predictions": [[[0.0] * BORZOI_OUTPUT] for _ in payload["sequences"]]}

    monkeypatch.setattr(borzoi_prediction.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    sequence = ("C" * 100) + ("A" * BORZOI_CONTEXT) + ("G" * 100)
    target_start = 100 + BORZOI_OUTPUT_FLANK
    result = run_borzoi(
        BorzoiInput(
            sequences=[sequence], target_ranges=[SequenceTargetRange(start=target_start, end=target_start + 10)]
        ),
        BorzoiConfig(output_tracks=[0], device="cuda"),
    )

    dispatched_sequence = captured_payloads[0]["sequences"][0]
    assert captured_payloads[0]["use_flash_attn"] is True
    assert len(dispatched_sequence) == BORZOI_CONTEXT
    assert dispatched_sequence == sequence[100 : 100 + BORZOI_CONTEXT]
    assert result.results[0].sequence == sequence
    assert result.results[0].context_start == 100
    assert result.results[0].context_end == 100 + BORZOI_CONTEXT
    assert result.results[0].output_start == 100 + BORZOI_OUTPUT_FLANK
    assert result.results[0].output_end == 100 + BORZOI_OUTPUT_FLANK + (BORZOI_OUTPUT * BORZOI_OUTPUT_RESOLUTION)
    assert result.results[0].output_resolution == BORZOI_OUTPUT_RESOLUTION
    assert result.results[0].target_start == target_start
    assert result.results[0].target_end == target_start + 10


# -- Config validation -----------------------------------------------------------------


def test_borzoi_config_rejects_invalid_species():
    """Test that invalid species is rejected."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiConfig

    with pytest.raises(ValidationError, match="Input should be 'human' or 'mouse'"):
        BorzoiConfig(output_tracks=[0], species="zebrafish")


def test_borzoi_config_rejects_invalid_replicate():
    """Test that invalid replicate is rejected."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiConfig

    with pytest.raises(ValidationError, match="Input should be '0', '1', '2' or '3'"):
        BorzoiConfig(output_tracks=[0], replicate="5")


@pytest.mark.parametrize(("species", "expected_flash_attn"), [("human", True), ("mouse", False)])
def test_borzoi_flash_attn_is_derived_from_species(monkeypatch, species, expected_flash_attn):
    """Borzoi selects FlashAttention internally from the species checkpoint."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiConfig, BorzoiInput, borzoi_prediction, run_borzoi

    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_payloads.append(payload)
        return {"predictions": [[[0.0] * BORZOI_OUTPUT] for _ in payload["sequences"]]}

    monkeypatch.setattr(borzoi_prediction.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    run_borzoi(
        BorzoiInput(sequences=["A" * BORZOI_CONTEXT]),
        BorzoiConfig(output_tracks=[0], species=species, device="cuda"),
    )

    assert captured_payloads[0]["use_flash_attn"] is expected_flash_attn


# -- Ensemble config validation --------------------------------------------------------


def test_borzoi_ensemble_config_mouse():
    """Mouse ensemble config is valid without a user-facing FlashAttention switch."""
    from proto_tools.tools.sequence_scoring.borzoi import BorzoiEnsembleConfig

    config = BorzoiEnsembleConfig(output_tracks=[0], species="mouse")
    assert config.species == "mouse"


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
def test_borzoi_prediction_human():
    """Test Borzoi prediction for human genome."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT)
    inputs = BorzoiInput(sequences=[sequence])
    config = BorzoiConfig(
        output_tracks=[0, 1, 2],
        species="human",
        replicate="0",
        avg_output_tracks=True,
        verbose=False,
    )

    result = run_borzoi(inputs, config)

    validate_output(result)

    assert result.tool_id == "borzoi-prediction"
    assert result.species == "human"
    assert result.replicate == "0"
    assert result.output_tracks == [0, 1, 2]
    assert result.avg_output_tracks is True
    assert result.results[0].sequence == sequence
    assert result.results[0].sequence_length == BORZOI_CONTEXT

    # 1 averaged track, 6144 output positions
    assert len(result.results[0].prediction) == 1
    assert len(result.results[0].prediction[0]) == 6144


@pytest.mark.uses_gpu
def test_borzoi_prediction_no_average():
    """Test Borzoi prediction without averaging tracks."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT, seed=123)
    inputs = BorzoiInput(sequences=[sequence])
    config = BorzoiConfig(
        output_tracks=[0, 1, 2, 3, 4],
        species="human",
        replicate="1",
        avg_output_tracks=False,
        verbose=False,
    )

    result = run_borzoi(inputs, config)

    validate_output(result)

    assert result.avg_output_tracks is False
    # 5 individual tracks, 6144 output positions each
    assert len(result.results[0].prediction) == 5
    assert len(result.results[0].prediction[0]) == 6144


@pytest.mark.slow
@pytest.mark.uses_gpu
def test_borzoi_prediction_different_replicates():
    """Test Borzoi prediction with all four replicates and verify they differ."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT, seed=456)
    inputs = BorzoiInput(sequences=[sequence])

    results = []
    for replicate in ["0", "1", "2", "3"]:
        config = BorzoiConfig(
            output_tracks=[0, 1],
            species="human",
            replicate=replicate,
            avg_output_tracks=True,
            verbose=False,
        )
        result = run_borzoi(inputs, config)
        validate_output(result)
        results.append(result)

    # Each replicate: 1 averaged track, 6144 positions
    for result in results:
        assert len(result.results[0].prediction) == 1
        assert len(result.results[0].prediction[0]) == 6144

    # Replicates are trained independently and should not produce identical predictions
    pred_0 = results[0].results[0].prediction[0]
    pred_1 = results[1].results[0].prediction[0]
    assert pred_0 != pred_1, "Different replicates should give different predictions"


@pytest.mark.uses_gpu
def test_borzoi_ensemble_prediction():
    """Test Borzoi ensemble prediction with all replicates."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT)
    inputs = BorzoiInput(sequences=[sequence])
    config = BorzoiEnsembleConfig(
        output_tracks=[0, 1, 2],
        species="human",
        avg_output_tracks=True,
        verbose=False,
    )

    result = run_borzoi_ensemble(inputs, config)

    validate_output(result)

    assert result.tool_id == "borzoi-ensemble"
    assert result.species == "human"
    assert result.output_tracks == [0, 1, 2]
    assert result.avg_output_tracks is True
    assert result.num_replicates == 4
    assert result.results[0].sequence_length == BORZOI_CONTEXT

    # 4 replicates x 1 averaged track x 6144 positions
    assert len(result.results[0].predictions) == 4
    assert len(result.results[0].predictions[0]) == 1
    assert len(result.results[0].predictions[0][0]) == 6144


@pytest.mark.uses_gpu
def test_borzoi_ensemble_no_average():
    """Test Borzoi ensemble prediction without averaging tracks."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT, seed=789)
    inputs = BorzoiInput(sequences=[sequence])
    config = BorzoiEnsembleConfig(
        output_tracks=[0, 1, 2, 3],
        species="human",
        avg_output_tracks=False,
        verbose=False,
    )

    result = run_borzoi_ensemble(inputs, config)

    validate_output(result)

    assert result.avg_output_tracks is False
    # 4 replicates x 4 individual tracks x 6144 positions
    assert len(result.results[0].predictions) == 4
    assert len(result.results[0].predictions[0]) == 4
    assert len(result.results[0].predictions[0][0]) == 6144


@pytest.mark.uses_gpu
def test_borzoi_ensemble_statistics():
    """Test that ensemble replicates vary and numpy statistics are well-shaped."""
    import numpy as np

    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequence = _generate_random_dna_sequence(BORZOI_CONTEXT, seed=999)
    inputs = BorzoiInput(sequences=[sequence])
    config = BorzoiEnsembleConfig(
        output_tracks=[0, 1],
        species="human",
        avg_output_tracks=True,
        verbose=False,
    )

    result = run_borzoi_ensemble(inputs, config)

    validate_output(result)

    predictions_array = np.array(result.results[0].predictions)
    mean_pred = predictions_array.mean(axis=0)
    std_pred = predictions_array.std(axis=0)

    assert mean_pred.shape == (1, 6144)
    assert std_pred.shape == (1, 6144)
    assert std_pred.sum() > 0, "Standard deviation should be non-zero across replicates"


# ── Benchmarks ──────────────────────────────────────────────────────────────


@pytest.mark.benchmark("borzoi-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_borzoi_prediction_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark borzoi-prediction: 4 random 524 kbp sequences, batch_size=4, 5 tracks (cold + warm)."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequences = random_dna_sequences(n=4, length=BORZOI_CONTEXT, seed=0)
    inputs = BorzoiInput(sequences=sequences)
    config = BorzoiConfig(
        output_tracks=[0, 1, 2, 3, 4],
        species="human",
        replicate="0",
        avg_output_tracks=False,
        batch_size=4,
        verbose=False,
    )

    result = benchmark_twice(request, "borzoi", lambda: run_borzoi(inputs, config))
    validate_output(result)

    assert result.tool_id == "borzoi-prediction"
    assert len(result.results) == 4
    for r in result.results:
        assert r.sequence_length == BORZOI_CONTEXT
        assert len(r.prediction) == 5  # 5 individual tracks
        assert len(r.prediction[0]) == 6144


@pytest.mark.benchmark("borzoi-ensemble")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_borzoi_ensemble_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark borzoi-ensemble: 4 random 524 kbp sequences across all 4 replicates, batch_size=4 (cold + warm)."""
    from proto_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequences = random_dna_sequences(n=4, length=BORZOI_CONTEXT, seed=1)
    inputs = BorzoiInput(sequences=sequences)
    config = BorzoiEnsembleConfig(
        output_tracks=[0, 1, 2],
        species="human",
        avg_output_tracks=True,
        batch_size=4,
        verbose=False,
    )

    result = benchmark_twice(request, "borzoi", lambda: run_borzoi_ensemble(inputs, config))
    validate_output(result)

    assert result.tool_id == "borzoi-ensemble"
    assert result.num_replicates == 4
    assert len(result.results) == 4
    for r in result.results:
        assert r.sequence_length == BORZOI_CONTEXT
        assert len(r.predictions) == 4  # 4 replicates
        assert len(r.predictions[0]) == 1  # 1 averaged track
        assert len(r.predictions[0][0]) == 6144
