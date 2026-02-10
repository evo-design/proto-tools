"""
test_borzoi.py

Tests for Borzoi regulatory activity prediction tool.
"""

import pytest
from pydantic import ValidationError

from tests.tool_tests.tool_infra_tests.test_export_functionality import validate_output

# Borzoi constants
BORZOI_CONTEXT = 524_288


def generate_random_dna_sequence(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence of given length."""
    import random
    random.seed(seed)
    return "".join(random.choices("ACGT", k=length))


class TestBorzoiInputValidation:
    """Tests for Borzoi input and config validation."""

    def test_borzoi_input_valid(self):
        """Test valid Borzoi input is accepted."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiInput

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT)
        inputs = BorzoiInput(sequence=sequence)
        assert len(inputs.sequence) == BORZOI_CONTEXT

    def test_borzoi_input_invalid_length(self):
        """Test that sequences with invalid length are rejected."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiInput

        # Too short
        with pytest.raises(ValueError, match=f"must have length {BORZOI_CONTEXT}"):
            BorzoiInput(sequence="ATCG" * 100)

        # Too long
        with pytest.raises(ValueError, match=f"must have length {BORZOI_CONTEXT}"):
            BorzoiInput(sequence="ATCG" * 200000)

    def test_borzoi_config_valid(self):
        """Test valid Borzoi config is accepted."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiConfig

        config = BorzoiConfig(
            output_tracks=[0, 1, 2, 3],
            species="human",
            replicate="0",
            device="cuda",
        )
        assert config.species == "human"
        assert config.output_tracks == [0, 1, 2, 3]
        assert config.replicate == "0"

    def test_borzoi_config_invalid_species(self):
        """Test that invalid species is rejected."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiConfig

        with pytest.raises(ValidationError, match="Input should be 'human' or 'mouse'"):
            BorzoiConfig(output_tracks=[0], species="zebrafish")

    def test_borzoi_config_invalid_replicate(self):
        """Test that invalid replicate is rejected."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiConfig

        with pytest.raises(ValidationError, match="Input should be '0', '1', '2' or '3'"):
            BorzoiConfig(output_tracks=[0], replicate="5")

    def test_borzoi_config_mouse_flash_attn_validation(self):
        """Test that FlashAttention cannot be used with mouse models."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiConfig

        with pytest.raises(ValueError, match="FlashAttention.*not available for mouse"):
            BorzoiConfig(
                output_tracks=[0],
                species="mouse",
                use_flash_attn=True,
            )

        # Should work when flash_attn is disabled
        config = BorzoiConfig(
            output_tracks=[0],
            species="mouse",
            use_flash_attn=False,
        )
        assert config.species == "mouse"
        assert config.use_flash_attn is False


class TestBorzoiPrediction:
    """Tests for Borzoi single-replicate prediction functionality."""

    @pytest.mark.uses_gpu
    def test_borzoi_prediction_human(self):
        """Test Borzoi prediction for human genome."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import (
            BorzoiConfig,
            BorzoiInput,
            run_borzoi,
        )

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT)
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

        # Check output structure
        assert result.tool_id == "borzoi-prediction"
        assert result.species == "human"
        assert result.replicate == "0"
        assert result.output_tracks == [0, 1, 2]
        assert result.avg_output_tracks is True
        assert result.sequence == sequence
        assert result.sequence_length == BORZOI_CONTEXT

        # Check prediction shape (1 track when averaged, 6144 positions)
        assert len(result.prediction) == 1
        assert len(result.prediction[0]) == 6144

    @pytest.mark.uses_gpu
    def test_borzoi_prediction_no_average(self):
        """Test Borzoi prediction without averaging tracks."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import (
            BorzoiConfig,
            BorzoiInput,
            run_borzoi,
        )

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT, seed=123)
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
        # Check prediction shape (5 tracks, 6144 positions)
        assert len(result.prediction) == 5
        assert len(result.prediction[0]) == 6144

    @pytest.mark.uses_gpu
    def test_borzoi_prediction_different_replicates(self):
        """Test Borzoi prediction with different replicates."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import (
            BorzoiConfig,
            BorzoiInput,
            run_borzoi,
        )

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT, seed=456)
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

        # Each replicate should have the same shape (1 track, 6144 positions)
        for result in results:
            assert len(result.prediction) == 1
            assert len(result.prediction[0]) == 6144

        # Different replicates should give different (but similar) predictions
        # They are trained independently so should not be identical
        pred_0 = results[0].prediction[0]
        pred_1 = results[1].prediction[0]
        assert pred_0 != pred_1, "Different replicates should give different predictions"


class TestBorzoiEnsemble:
    """Tests for Borzoi ensemble prediction functionality."""

    @pytest.mark.uses_gpu
    def test_borzoi_ensemble_prediction(self):
        """Test Borzoi ensemble prediction with all replicates."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import (
            BorzoiEnsembleConfig,
            BorzoiInput,
            run_borzoi_ensemble,
        )

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT)
        inputs = BorzoiInput(sequence=sequence)
        config = BorzoiEnsembleConfig(
            output_tracks=[0, 1, 2],
            species="human",
            avg_output_tracks=True,
            verbose=False,
        )

        result = run_borzoi_ensemble(inputs, config)

        validate_output(result)

        # Check output structure
        assert result.tool_id == "borzoi-ensemble"
        assert result.species == "human"
        assert result.output_tracks == [0, 1, 2]
        assert result.avg_output_tracks is True
        assert result.num_replicates == 4
        assert result.sequence_length == BORZOI_CONTEXT

        # Check predictions shape (4 replicates, 1 track when averaged, 6144 positions)
        assert len(result.predictions) == 4
        assert len(result.predictions[0]) == 1
        assert len(result.predictions[0][0]) == 6144

    @pytest.mark.uses_gpu
    def test_borzoi_ensemble_no_average(self):
        """Test Borzoi ensemble prediction without averaging tracks."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import (
            BorzoiEnsembleConfig,
            BorzoiInput,
            run_borzoi_ensemble,
        )

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT, seed=789)
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
        # Check predictions shape (4 replicates, 4 tracks, 6144 positions)
        assert len(result.predictions) == 4
        assert len(result.predictions[0]) == 4
        assert len(result.predictions[0][0]) == 6144

    @pytest.mark.uses_gpu
    def test_borzoi_ensemble_statistics(self):
        """Test computing ensemble statistics from predictions."""
        import numpy as np

        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import (
            BorzoiEnsembleConfig,
            BorzoiInput,
            run_borzoi_ensemble,
        )

        sequence = generate_random_dna_sequence(BORZOI_CONTEXT, seed=999)
        inputs = BorzoiInput(sequence=sequence)
        config = BorzoiEnsembleConfig(
            output_tracks=[0, 1],
            species="human",
            avg_output_tracks=True,
            verbose=False,
        )

        result = run_borzoi_ensemble(inputs, config)

        validate_output(result)

        # Compute ensemble mean and std using numpy
        predictions_array = np.array(result.predictions)
        mean_pred = predictions_array.mean(axis=0)
        std_pred = predictions_array.std(axis=0)

        assert mean_pred.shape == (1, 6144)
        assert std_pred.shape == (1, 6144)

        # Std should be non-zero (replicates should vary)
        assert std_pred.sum() > 0, "Standard deviation should be non-zero across replicates"


class TestBorzoiEnsembleConfigValidation:
    """Tests for Borzoi ensemble config validation."""

    def test_borzoi_ensemble_config_valid(self):
        """Test valid Borzoi ensemble config is accepted."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiEnsembleConfig

        config = BorzoiEnsembleConfig(
            output_tracks=[0, 1, 2],
            species="human",
            avg_output_tracks=True,
        )
        assert config.species == "human"
        assert config.output_tracks == [0, 1, 2]

    def test_borzoi_ensemble_config_mouse_flash_attn_validation(self):
        """Test that FlashAttention cannot be used with mouse models in ensemble."""
        from bio_programming.bio_tools.tools.sequence_scoring.borzoi import BorzoiEnsembleConfig

        with pytest.raises(ValueError, match="FlashAttention.*not available for mouse"):
            BorzoiEnsembleConfig(
                output_tracks=[0],
                species="mouse",
                use_flash_attn=True,
            )

        # Should work when flash_attn is disabled
        config = BorzoiEnsembleConfig(
            output_tracks=[0],
            species="mouse",
            use_flash_attn=False,
        )
        assert config.species == "mouse"
