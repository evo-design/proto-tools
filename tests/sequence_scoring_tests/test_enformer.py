"""tests/sequence_scoring_tests/test_enformer.py

Tests for Enformer regulatory activity prediction tool."""

import random

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.sequence_scoring.enformer import ENFORMER_CONTEXT
from tests.conftest import make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("enformer")


def _random_dna(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence of given length."""
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=length))


# ── Input validation ──────────────────────────────────────────────────────────

def test_enformer_input_valid():
    """Valid sequence of the required length is accepted."""
    from bio_programming_tools.tools.sequence_scoring.enformer import EnformerInput

    seq = _random_dna(ENFORMER_CONTEXT)
    inp = EnformerInput(sequence=seq)
    assert len(inp.sequence) == ENFORMER_CONTEXT


@pytest.mark.parametrize("seq,label", [
    ("ATCG" * 100, "too short"),
    ("ATCG" * 100_000, "too long"),
])
def test_enformer_input_rejects_wrong_length(seq, label):
    """Sequences that are not exactly ENFORMER_CONTEXT bp are rejected."""
    from bio_programming_tools.tools.sequence_scoring.enformer import EnformerInput

    with pytest.raises(ValueError, match=f"must have length {ENFORMER_CONTEXT}"):
        EnformerInput(sequence=seq)


def test_enformer_input_rejects_empty():
    """Empty sequences are rejected."""
    from bio_programming_tools.tools.sequence_scoring.enformer import EnformerInput

    with pytest.raises((ValueError, ValidationError), match="[Ss]equence"):
        EnformerInput(sequence="")


def test_enformer_input_rejects_invalid_nucleotides():
    """Sequences containing invalid nucleotide characters are rejected."""
    from bio_programming_tools.tools.sequence_scoring.enformer import EnformerInput

    bad_seq = "X" * ENFORMER_CONTEXT
    with pytest.raises((ValueError, ValidationError), match="[Ii]nvalid"):
        EnformerInput(sequence=bad_seq)


# ── Config validation ─────────────────────────────────────────────────────────

def test_enformer_config_default_species():
    """Default species is 'human'."""
    from bio_programming_tools.tools.sequence_scoring.enformer import EnformerConfig

    config = EnformerConfig(output_tracks=[0])
    assert config.species == "human"


def test_enformer_config_rejects_invalid_species():
    """Species values other than 'human' or 'mouse' are rejected."""
    from bio_programming_tools.tools.sequence_scoring.enformer import EnformerConfig

    with pytest.raises(ValidationError, match="Input should be 'human' or 'mouse'"):
        EnformerConfig(output_tracks=[0], species="zebrafish")


# ---------------------------------------------------------------------------
# Integration tests

@pytest.mark.include_in_env_report(category="sequence_scoring")
@pytest.mark.uses_gpu
def test_enformer_prediction_human():
    """Enformer produces a [896, num_tracks] output matrix for human sequences."""
    from bio_programming_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    seq = _random_dna(ENFORMER_CONTEXT)
    inputs = EnformerInput(sequence=seq)
    config = EnformerConfig(output_tracks=[0, 1, 2], species="human", verbose=False)

    result = run_enformer(inputs, config)

    validate_output(result)
    assert result.tool_id == "enformer-prediction"
    assert result.species == "human"
    assert result.output_tracks == [0, 1, 2]
    assert result.sequence == seq
    assert result.sequence_length == ENFORMER_CONTEXT
    assert len(result.prediction) == 896
    assert len(result.prediction[0]) == 3


@pytest.mark.uses_gpu
def test_enformer_prediction_mouse():
    """Enformer produces a [896, num_tracks] output matrix for mouse sequences."""
    from bio_programming_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    seq = _random_dna(ENFORMER_CONTEXT, seed=123)
    inputs = EnformerInput(sequence=seq)
    config = EnformerConfig(output_tracks=[0, 5, 10], species="mouse", verbose=False)

    result = run_enformer(inputs, config)

    validate_output(result)
    assert result.species == "mouse"
    assert len(result.prediction) == 896
    assert len(result.prediction[0]) == 3


@pytest.mark.parametrize("track_indices,expected_n_tracks", [
    ([0], 1),
    (list(range(100)), 100),
])
@pytest.mark.uses_gpu
def test_enformer_prediction_track_count(track_indices, expected_n_tracks):
    """Output matrix second dimension matches the number of requested tracks."""
    from bio_programming_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    seq = _random_dna(ENFORMER_CONTEXT, seed=456)
    inputs = EnformerInput(sequence=seq)
    config = EnformerConfig(
        output_tracks=track_indices, species="human", verbose=False
    )

    result = run_enformer(inputs, config)

    validate_output(result)
    assert len(result.prediction) == 896
    assert len(result.prediction[0]) == expected_n_tracks
