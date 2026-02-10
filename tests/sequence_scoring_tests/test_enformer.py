"""
test_enformer.py

Tests for Enformer regulatory activity prediction tool.
"""

import pytest
from pydantic import ValidationError

from tests.tool_infra_tests.test_export_functionality import validate_output

# Enformer constants
ENFORMER_CONTEXT = 196_608


def generate_random_dna_sequence(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence of given length."""
    import random
    random.seed(seed)
    return "".join(random.choices("ACGT", k=length))


class TestEnformerInputValidation:
    """Tests for Enformer input and config validation."""

    def test_enformer_input_valid(self):
        """Test valid Enformer input is accepted."""
        from bio_programming_tools.tools.sequence_scoring.enformer import EnformerInput

        sequence = generate_random_dna_sequence(ENFORMER_CONTEXT)
        inputs = EnformerInput(sequence=sequence)
        assert len(inputs.sequence) == ENFORMER_CONTEXT

    def test_enformer_input_invalid_length(self):
        """Test that sequences with invalid length are rejected."""
        from bio_programming_tools.tools.sequence_scoring.enformer import EnformerInput

        # Too short
        with pytest.raises(ValueError, match=f"must have length {ENFORMER_CONTEXT}"):
            EnformerInput(sequence="ATCG" * 100)

        # Too long
        with pytest.raises(ValueError, match=f"must have length {ENFORMER_CONTEXT}"):
            EnformerInput(sequence="ATCG" * 100000)

    def test_enformer_config_valid(self):
        """Test valid Enformer config is accepted."""
        from bio_programming_tools.tools.sequence_scoring.enformer import EnformerConfig

        config = EnformerConfig(
            output_tracks=[0, 1, 2, 3],
            species="human",
            device="cuda",
        )
        assert config.species == "human"
        assert config.output_tracks == [0, 1, 2, 3]

    def test_enformer_config_invalid_species(self):
        """Test that invalid species is rejected."""
        from bio_programming_tools.tools.sequence_scoring.enformer import EnformerConfig

        with pytest.raises(ValidationError, match="Input should be 'human' or 'mouse'"):
            EnformerConfig(output_tracks=[0], species="zebrafish")


class TestEnformerPrediction:
    """Tests for Enformer prediction functionality."""

    @pytest.mark.uses_gpu
    def test_enformer_prediction_human(self):
        """Test Enformer prediction for human genome."""
        from bio_programming_tools.tools.sequence_scoring.enformer import (
            EnformerConfig,
            EnformerInput,
            run_enformer,
        )

        sequence = generate_random_dna_sequence(ENFORMER_CONTEXT)
        inputs = EnformerInput(sequence=sequence)
        config = EnformerConfig(
            output_tracks=[0, 1, 2],
            species="human",
            verbose=False,
        )

        result = run_enformer(inputs, config)

        # Validate output
        validate_output(result)

        # Check output structure
        assert result.tool_id == "enformer-prediction"
        assert result.species == "human"
        assert result.output_tracks == [0, 1, 2]
        assert result.sequence == sequence
        assert result.sequence_length == ENFORMER_CONTEXT

        # Check prediction shape (896 positions, 3 tracks)
        assert len(result.prediction) == 896
        assert len(result.prediction[0]) == 3

    @pytest.mark.uses_gpu
    def test_enformer_prediction_mouse(self):
        """Test Enformer prediction for mouse genome."""
        from bio_programming_tools.tools.sequence_scoring.enformer import (
            EnformerConfig,
            EnformerInput,
            run_enformer,
        )

        sequence = generate_random_dna_sequence(ENFORMER_CONTEXT, seed=123)
        inputs = EnformerInput(sequence=sequence)
        config = EnformerConfig(
            output_tracks=[0, 5, 10],
            species="mouse",
            verbose=False,
        )

        result = run_enformer(inputs, config)

        validate_output(result)

        assert result.species == "mouse"
        assert len(result.prediction) == 896
        assert len(result.prediction[0]) == 3

    @pytest.mark.uses_gpu
    def test_enformer_prediction_single_track(self):
        """Test Enformer prediction with single output track."""
        from bio_programming_tools.tools.sequence_scoring.enformer import (
            EnformerConfig,
            EnformerInput,
            run_enformer,
        )

        sequence = generate_random_dna_sequence(ENFORMER_CONTEXT, seed=456)
        inputs = EnformerInput(sequence=sequence)
        config = EnformerConfig(
            output_tracks=[0],
            species="human",
            verbose=False,
        )

        result = run_enformer(inputs, config)

        validate_output(result)

        assert len(result.prediction) == 896
        assert len(result.prediction[0]) == 1

    @pytest.mark.uses_gpu
    def test_enformer_prediction_many_tracks(self):
        """Test Enformer prediction with many output tracks."""
        from bio_programming_tools.tools.sequence_scoring.enformer import (
            EnformerConfig,
            EnformerInput,
            run_enformer,
        )

        sequence = generate_random_dna_sequence(ENFORMER_CONTEXT, seed=789)
        inputs = EnformerInput(sequence=sequence)
        # Request first 100 human tracks
        config = EnformerConfig(
            output_tracks=list(range(100)),
            species="human",
            verbose=False,
        )

        result = run_enformer(inputs, config)

        validate_output(result)

        assert len(result.prediction) == 896
        assert len(result.prediction[0]) == 100
