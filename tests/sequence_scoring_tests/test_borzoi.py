"""Tests for Borzoi regulatory activity prediction tool."""

import random

import pytest
from pydantic import ValidationError

from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output


_persistent_tool = make_persistent_fixture("borzoi")

_BORZOI_CONTEXT = 524_288


def _generate_random_dna_sequence(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence of given length."""
    random.seed(seed)
    return "".join(random.choices("ACGT", k=length))


# -- Input validation ------------------------------------------------------------------


def test_borzoi_input_valid():
    """Test valid Borzoi input is accepted."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiInput

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT)
    inputs = BorzoiInput(sequence=sequence)
    assert len(inputs.sequence) == _BORZOI_CONTEXT


def test_borzoi_input_rejects_wrong_length():
    """Test that sequences with invalid length are rejected."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiInput

    # Too short
    with pytest.raises(ValueError, match=f"must have length {_BORZOI_CONTEXT}"):
        BorzoiInput(sequence="ATCG" * 100)

    # Too long
    with pytest.raises(ValueError, match=f"must have length {_BORZOI_CONTEXT}"):
        BorzoiInput(sequence="ATCG" * 200000)


# -- Config validation -----------------------------------------------------------------


def test_borzoi_config_rejects_invalid_species():
    """Test that invalid species is rejected."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiConfig

    with pytest.raises(ValidationError, match="Input should be 'human' or 'mouse'"):
        BorzoiConfig(output_tracks=[0], species="zebrafish")


def test_borzoi_config_rejects_invalid_replicate():
    """Test that invalid replicate is rejected."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiConfig

    with pytest.raises(ValidationError, match="Input should be '0', '1', '2' or '3'"):
        BorzoiConfig(output_tracks=[0], replicate="5")


def test_borzoi_config_rejects_mouse_with_flash_attn():
    """Test that FlashAttention cannot be used with mouse models."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiConfig

    with pytest.raises(ValueError, match="FlashAttention.*not available for mouse"):
        BorzoiConfig(output_tracks=[0], species="mouse", use_flash_attn=True)


def test_borzoi_config_mouse_without_flash_attn():
    """Mouse config is valid when FlashAttention is disabled."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiConfig

    config = BorzoiConfig(output_tracks=[0], species="mouse", use_flash_attn=False)
    assert config.species == "mouse"
    assert config.use_flash_attn is False


# -- Ensemble config validation --------------------------------------------------------


def test_borzoi_ensemble_config_rejects_mouse_with_flash_attn():
    """Test that FlashAttention cannot be used with mouse models in ensemble."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiEnsembleConfig

    with pytest.raises(ValueError, match="FlashAttention.*not available for mouse"):
        BorzoiEnsembleConfig(output_tracks=[0], species="mouse", use_flash_attn=True)


def test_borzoi_ensemble_config_mouse_without_flash_attn():
    """Ensemble mouse config is valid when FlashAttention is disabled."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import BorzoiEnsembleConfig

    config = BorzoiEnsembleConfig(output_tracks=[0], species="mouse", use_flash_attn=False)
    assert config.species == "mouse"


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.include_in_env_report(category="sequence_scoring")
@pytest.mark.uses_gpu
def test_borzoi_prediction_human():
    """Test Borzoi prediction for human genome."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT)
    inputs = BorzoiInput(sequence=sequence)
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
    assert result.sequence == sequence
    assert result.sequence_length == _BORZOI_CONTEXT

    # 1 averaged track, 6144 output positions
    assert len(result.prediction) == 1
    assert len(result.prediction[0]) == 6144


@pytest.mark.uses_gpu
def test_borzoi_prediction_no_average():
    """Test Borzoi prediction without averaging tracks."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT, seed=123)
    inputs = BorzoiInput(sequence=sequence)
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
    assert len(result.prediction) == 5
    assert len(result.prediction[0]) == 6144


@pytest.mark.slow
@pytest.mark.uses_gpu
def test_borzoi_prediction_different_replicates():
    """Test Borzoi prediction with all four replicates and verify they differ."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import (
        BorzoiConfig,
        BorzoiInput,
        run_borzoi,
    )

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT, seed=456)
    inputs = BorzoiInput(sequence=sequence)

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
        assert len(result.prediction) == 1
        assert len(result.prediction[0]) == 6144

    # Replicates are trained independently and should not produce identical predictions
    pred_0 = results[0].prediction[0]
    pred_1 = results[1].prediction[0]
    assert pred_0 != pred_1, "Different replicates should give different predictions"


@pytest.mark.uses_gpu
def test_borzoi_ensemble_prediction():
    """Test Borzoi ensemble prediction with all replicates."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT)
    inputs = BorzoiInput(sequence=sequence)
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
    assert result.sequence_length == _BORZOI_CONTEXT

    # 4 replicates × 1 averaged track × 6144 positions
    assert len(result.predictions) == 4
    assert len(result.predictions[0]) == 1
    assert len(result.predictions[0][0]) == 6144


@pytest.mark.uses_gpu
def test_borzoi_ensemble_no_average():
    """Test Borzoi ensemble prediction without averaging tracks."""
    from bio_programming_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT, seed=789)
    inputs = BorzoiInput(sequence=sequence)
    config = BorzoiEnsembleConfig(
        output_tracks=[0, 1, 2, 3],
        species="human",
        avg_output_tracks=False,
        verbose=False,
    )

    result = run_borzoi_ensemble(inputs, config)

    validate_output(result)

    assert result.avg_output_tracks is False
    # 4 replicates × 4 individual tracks × 6144 positions
    assert len(result.predictions) == 4
    assert len(result.predictions[0]) == 4
    assert len(result.predictions[0][0]) == 6144


@pytest.mark.uses_gpu
def test_borzoi_ensemble_statistics():
    """Test that ensemble replicates vary and numpy statistics are well-shaped."""
    import numpy as np

    from bio_programming_tools.tools.sequence_scoring.borzoi import (
        BorzoiEnsembleConfig,
        BorzoiInput,
        run_borzoi_ensemble,
    )

    sequence = _generate_random_dna_sequence(_BORZOI_CONTEXT, seed=999)
    inputs = BorzoiInput(sequence=sequence)
    config = BorzoiEnsembleConfig(
        output_tracks=[0, 1],
        species="human",
        avg_output_tracks=True,
        verbose=False,
    )

    result = run_borzoi_ensemble(inputs, config)

    validate_output(result)

    predictions_array = np.array(result.predictions)
    mean_pred = predictions_array.mean(axis=0)
    std_pred = predictions_array.std(axis=0)

    assert mean_pred.shape == (1, 6144)
    assert std_pred.shape == (1, 6144)
    assert std_pred.sum() > 0, "Standard deviation should be non-zero across replicates"
