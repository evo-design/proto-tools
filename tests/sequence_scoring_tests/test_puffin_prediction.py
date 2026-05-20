"""Tests for Puffin transcription initiation prediction."""

import random

import pytest

from proto_tools.tools.sequence_scoring.puffin import PUFFIN_MIN_INPUT_LENGTH, PUFFIN_OUTPUT_CHANNELS, TRACK_NAMES
from tests.conftest import benchmark_twice, make_persistent_fixture, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("puffin")


def _random_dna(length: int, seed: int = 42) -> str:
    """Random DNA sequence of given length."""
    return "".join(random.Random(seed).choices("ACGT", k=length))


# ── Input validation ──────────────────────────────────────────────────────────


def test_puffin_prediction_input_normalizes_single_string():
    """A bare string is wrapped to a one-item list."""
    from proto_tools.tools.sequence_scoring.puffin import PuffinPredictionInput

    seq = _random_dna(1650)
    inputs = PuffinPredictionInput(sequences=seq)
    assert inputs.sequences == [seq]


def test_puffin_prediction_input_rejects_short_sequence():
    """Sequences below 651 bp are rejected with a message naming the constants."""
    from proto_tools.tools.sequence_scoring.puffin import PuffinPredictionInput

    with pytest.raises(ValueError, match=str(PUFFIN_MIN_INPUT_LENGTH)) as exc_info:
        PuffinPredictionInput(sequences=_random_dna(PUFFIN_MIN_INPUT_LENGTH - 1))
    assert "325" in str(exc_info.value)


def test_puffin_prediction_input_rejects_invalid_nucleotides():
    """Non-ACGTN characters in any input are rejected."""
    from proto_tools.tools.sequence_scoring.puffin import PuffinPredictionInput

    with pytest.raises(ValueError, match=r"[Ii]nvalid"):
        PuffinPredictionInput(sequences="X" * 1650)


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_puffin_prediction_full_shape_and_coords():
    """End-to-end: per-base 10-channel output with correct shape and coordinates."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinPredictionConfig,
        PuffinPredictionInput,
        run_puffin_prediction,
    )

    seq = _random_dna(1650, seed=0)
    result = run_puffin_prediction(PuffinPredictionInput(sequences=[seq]), PuffinPredictionConfig())

    validate_output(result)
    assert result.tool_id == "puffin-prediction"
    assert result.track_names == list(TRACK_NAMES)

    entry = result.results[0]
    assert entry.sequence == seq
    assert entry.output_length == len(seq) - 650
    assert entry.output_start == 325
    assert entry.output_end == len(seq) - 325
    assert len(entry.predictions) == entry.output_length
    assert all(len(row) == PUFFIN_OUTPUT_CHANNELS for row in entry.predictions)


@pytest.mark.uses_gpu
def test_puffin_prediction_batch_preserves_per_input_results():
    """Multi-sequence input returns one result per sequence in order, with distinct outputs."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinPredictionConfig,
        PuffinPredictionInput,
        run_puffin_prediction,
    )

    seqs = [_random_dna(1650, seed=i) for i in range(3)]
    result = run_puffin_prediction(PuffinPredictionInput(sequences=seqs), PuffinPredictionConfig())

    validate_output(result)
    assert [r.sequence for r in result.results] == seqs
    # Distinct inputs should not collapse to identical outputs.
    centers = [r.predictions[len(r.predictions) // 2] for r in result.results]
    assert centers[0] != centers[1] and centers[1] != centers[2]


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("puffin-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_puffin_prediction_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark: 8 random 1650 bp sequences (cold + warm)."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinPredictionConfig,
        PuffinPredictionInput,
        run_puffin_prediction,
    )

    inputs = PuffinPredictionInput(sequences=random_dna_sequences(n=8, length=1650, seed=0))
    config = PuffinPredictionConfig()
    result = benchmark_twice(request, "puffin", lambda: run_puffin_prediction(inputs, config))

    validate_output(result)
    assert len(result.results) == 8
