"""tests/sequence_scoring_tests/test_enformer.py.

Tests for Enformer regulatory activity prediction tool.
"""

import random

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_scoring.enformer import (
    ENFORMER_CONTEXT,
    ENFORMER_OUTPUT,
    ENFORMER_OUTPUT_FLANK,
    ENFORMER_OUTPUT_RESOLUTION,
)
from tests.conftest import benchmark_twice, make_persistent_fixture, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("enformer")


def _random_dna(length: int, seed: int = 42) -> str:
    """Generate a random DNA sequence of given length."""
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=length))


# ── Input validation ──────────────────────────────────────────────────────────


def test_enformer_input_valid():
    """Valid sequence of the required length is accepted."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerInput

    seq = _random_dna(ENFORMER_CONTEXT)
    inp = EnformerInput(sequences=seq)
    assert inp.sequences == [seq]
    assert len(inp.sequences[0]) == ENFORMER_CONTEXT


def test_enformer_input_accepts_sequence_batches():
    """Enformer input should normalize sequence batches to a list."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerInput

    seq_a = _random_dna(ENFORMER_CONTEXT)
    seq_b = _random_dna(ENFORMER_CONTEXT, seed=123)

    inp = EnformerInput(sequences=[seq_a, seq_b])

    assert inp.sequences == [seq_a, seq_b]
    assert len(inp) == 2


@pytest.mark.parametrize(
    "seq,label",
    [
        ("ATCG" * 100, "too short"),
        ("ATCG" * 100_000, "too long"),
    ],
)
def test_enformer_input_rejects_wrong_length(seq, label):
    """Sequences that are not exactly ENFORMER_CONTEXT bp are rejected."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerInput

    with pytest.raises(ValueError, match=f"must have length {ENFORMER_CONTEXT}"):
        EnformerInput(sequences=seq)


def test_enformer_input_rejects_empty():
    """Empty sequences are rejected."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerInput

    with pytest.raises((ValueError, ValidationError), match=r"[Ss]equence"):
        EnformerInput(sequences="")


def test_enformer_input_rejects_invalid_nucleotides():
    """Sequences containing invalid nucleotide characters are rejected."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerInput

    bad_seq = "X" * ENFORMER_CONTEXT
    with pytest.raises((ValueError, ValidationError), match=r"[Ii]nvalid"):
        EnformerInput(sequences=bad_seq)


def test_enformer_input_accepts_target_aligned_sequences():
    """Target coordinates let Enformer extract a model window from a larger construct."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerInput, SequenceTargetRange

    sequence = ("C" * 100) + ("A" * ENFORMER_CONTEXT) + ("G" * 100)
    target_start = 100 + ENFORMER_OUTPUT_FLANK

    target_range = SequenceTargetRange(start=target_start, end=target_start + 10)
    inputs = EnformerInput(sequences=sequence, target_ranges=target_range)

    assert inputs.sequences == [sequence]
    assert inputs.target_ranges == [target_range]


def test_enformer_run_extracts_target_aligned_window(monkeypatch):
    """run_enformer should dispatch exact-context windows and return source coordinates."""
    from proto_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        SequenceTargetRange,
        enformer_prediction,
        run_enformer,
    )

    captured_payloads = []

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured_payloads.append(payload)
        return {"predictions": [[[0.0] for _ in range(ENFORMER_OUTPUT)] for _ in payload["sequences"]]}

    monkeypatch.setattr(enformer_prediction.ToolInstance, "dispatch", staticmethod(fake_dispatch))

    sequence = ("C" * 100) + ("A" * ENFORMER_CONTEXT) + ("G" * 100)
    target_start = 100 + ENFORMER_OUTPUT_FLANK
    result = run_enformer(
        EnformerInput(
            sequences=[sequence],
            target_ranges=[SequenceTargetRange(start=target_start, end=target_start + 10)],
        ),
        EnformerConfig(output_tracks=[0], device="cpu"),
    )

    dispatched_sequence = captured_payloads[0]["sequences"][0]
    assert len(dispatched_sequence) == ENFORMER_CONTEXT
    assert dispatched_sequence == sequence[100 : 100 + ENFORMER_CONTEXT]
    assert result.results[0].sequence == sequence
    assert result.results[0].context_start == 100
    assert result.results[0].context_end == 100 + ENFORMER_CONTEXT
    assert result.results[0].output_start == 100 + ENFORMER_OUTPUT_FLANK
    assert result.results[0].output_end == 100 + ENFORMER_OUTPUT_FLANK + (ENFORMER_OUTPUT * ENFORMER_OUTPUT_RESOLUTION)
    assert result.results[0].output_resolution == ENFORMER_OUTPUT_RESOLUTION
    assert result.results[0].target_start == target_start
    assert result.results[0].target_end == target_start + 10


# ── Config validation ─────────────────────────────────────────────────────────


def test_enformer_config_default_species():
    """Default species is 'human'."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerConfig

    config = EnformerConfig(output_tracks=[0])
    assert config.species == "human"


def test_enformer_config_rejects_invalid_species():
    """Species values other than 'human' or 'mouse' are rejected."""
    from proto_tools.tools.sequence_scoring.enformer import EnformerConfig

    with pytest.raises(ValidationError, match="Input should be 'human' or 'mouse'"):
        EnformerConfig(output_tracks=[0], species="zebrafish")


# ---------------------------------------------------------------------------
# Integration tests


@pytest.mark.uses_gpu
def test_enformer_prediction_human():
    """Enformer produces a [896, num_tracks] output matrix for human sequences."""
    from proto_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    seq = _random_dna(ENFORMER_CONTEXT)
    inputs = EnformerInput(sequences=[seq])
    config = EnformerConfig(output_tracks=[0, 1, 2], species="human", verbose=False)

    result = run_enformer(inputs, config)

    validate_output(result)
    assert result.tool_id == "enformer-prediction"
    assert result.species == "human"
    assert result.output_tracks == [0, 1, 2]
    assert result.results[0].sequence == seq
    assert result.results[0].sequence_length == ENFORMER_CONTEXT
    assert len(result.results[0].prediction) == 896
    assert len(result.results[0].prediction[0]) == 3


@pytest.mark.uses_gpu
def test_enformer_prediction_mouse():
    """Enformer produces a [896, num_tracks] output matrix for mouse sequences."""
    from proto_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    seq = _random_dna(ENFORMER_CONTEXT, seed=123)
    inputs = EnformerInput(sequences=[seq])
    config = EnformerConfig(output_tracks=[0, 5, 10], species="mouse", verbose=False)

    result = run_enformer(inputs, config)

    validate_output(result)
    assert result.species == "mouse"
    assert len(result.results[0].prediction) == 896
    assert len(result.results[0].prediction[0]) == 3


@pytest.mark.parametrize(
    "track_indices,expected_n_tracks",
    [
        ([0], 1),
        (list(range(100)), 100),
    ],
)
@pytest.mark.uses_gpu
def test_enformer_prediction_track_count(track_indices, expected_n_tracks):
    """Output matrix second dimension matches the number of requested tracks."""
    from proto_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    seq = _random_dna(ENFORMER_CONTEXT, seed=456)
    inputs = EnformerInput(sequences=[seq])
    config = EnformerConfig(output_tracks=track_indices, species="human", verbose=False)

    result = run_enformer(inputs, config)

    validate_output(result)
    assert len(result.results[0].prediction) == 896
    assert len(result.results[0].prediction[0]) == expected_n_tracks


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("enformer-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_enformer_prediction_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark enformer-prediction: 8 random 196,608 bp sequences, batch_size=8, 10 tracks (cold + warm)."""
    from proto_tools.tools.sequence_scoring.enformer import (
        EnformerConfig,
        EnformerInput,
        run_enformer,
    )

    sequences = random_dna_sequences(n=8, length=ENFORMER_CONTEXT, seed=0)
    inputs = EnformerInput(sequences=sequences)
    # 10 tracks keeps the per-position output non-trivial without bloating I/O.
    config = EnformerConfig(
        output_tracks=list(range(10)),
        species="human",
        batch_size=8,
        verbose=False,
    )

    result = benchmark_twice(request, "enformer", lambda: run_enformer(inputs, config))
    validate_output(result)

    assert result.tool_id == "enformer-prediction"
    assert len(result.results) == 8
    for r in result.results:
        assert r.sequence_length == ENFORMER_CONTEXT
        assert len(r.prediction) == 896
        assert len(r.prediction[0]) == 10
