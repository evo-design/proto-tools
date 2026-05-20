"""Tests for Puffin motif-level interpretation of transcription initiation."""

import random

import pytest

from proto_tools.tools.sequence_scoring.puffin import MOTIF_NAMES, PUFFIN_MIN_INPUT_LENGTH
from tests.conftest import benchmark_twice, make_persistent_fixture, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("puffin")

_EXPECTED_MOTIF_KEYS = {f"{name}{strand}" for name in MOTIF_NAMES for strand in ("+", "-")}


def _random_dna(length: int, seed: int = 42) -> str:
    """Random DNA sequence of given length."""
    return "".join(random.Random(seed).choices("ACGT", k=length))


# ── Input validation ──────────────────────────────────────────────────────────


def test_puffin_interpretation_input_normalizes_single_string():
    """A bare string is wrapped to a one-item list."""
    from proto_tools.tools.sequence_scoring.puffin import PuffinInterpretationInput

    seq = _random_dna(1650)
    inputs = PuffinInterpretationInput(sequences=seq)
    assert inputs.sequences == [seq]


def test_puffin_interpretation_input_rejects_short_sequence():
    """Sequences below 651 bp are rejected with a message naming the constants."""
    from proto_tools.tools.sequence_scoring.puffin import PuffinInterpretationInput

    with pytest.raises(ValueError, match=str(PUFFIN_MIN_INPUT_LENGTH)) as exc_info:
        PuffinInterpretationInput(sequences=_random_dna(PUFFIN_MIN_INPUT_LENGTH - 1))
    assert "325" in str(exc_info.value)


def test_puffin_interpretation_input_rejects_invalid_nucleotides():
    """Non-ACGTN characters in any input are rejected."""
    from proto_tools.tools.sequence_scoring.puffin import PuffinInterpretationInput

    with pytest.raises(ValueError, match=r"[Ii]nvalid"):
        PuffinInterpretationInput(sequences="X" * 1650)


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
def test_puffin_interpretation_full_output_schema():
    """End-to-end: prediction + 9-motif x 2-strand decomposition, all tracks at the expected length."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinInterpretationConfig,
        PuffinInterpretationInput,
        run_puffin_interpretation,
    )

    seq = _random_dna(1650, seed=0)
    result = run_puffin_interpretation(PuffinInterpretationInput(sequences=[seq]), PuffinInterpretationConfig())

    validate_output(result)
    assert result.tool_id == "puffin-interpretation"
    assert result.target_signal == "FANTOM_CAGE"
    assert result.reverse_strand is False

    entry = result.results[0]
    expected_length = len(seq) - 650
    assert len(entry.prediction) == expected_length

    motif_dicts = [
        entry.motif_activations,
        entry.motif_effects,
        entry.bp_contribution_per_motif,
        entry.bp_contribution_to_motif_activation,
    ]
    for d in motif_dicts:
        assert set(d.keys()) == _EXPECTED_MOTIF_KEYS
        assert all(len(v) == expected_length for v in d.values())

    for sum_track in (
        entry.sum_motif_effects,
        entry.sum_initiator_effects,
        entry.sum_trinucleotide_effects,
        entry.sum_total_effects,
        entry.bp_contribution,
    ):
        assert len(sum_track) == expected_length


@pytest.mark.uses_gpu
def test_puffin_interpretation_target_signal_actually_changes_output():
    """Switching target_signal produces a different prediction track, not just an echoed label."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinInterpretationConfig,
        PuffinInterpretationInput,
        run_puffin_interpretation,
    )

    inputs = PuffinInterpretationInput(sequences=[_random_dna(1650, seed=3)])
    fantom = run_puffin_interpretation(inputs, PuffinInterpretationConfig(target_signal="FANTOM_CAGE"))
    pro_cap = run_puffin_interpretation(inputs, PuffinInterpretationConfig(target_signal="PRO_CAP"))

    assert fantom.target_signal == "FANTOM_CAGE"
    assert pro_cap.target_signal == "PRO_CAP"
    assert fantom.results[0].prediction != pro_cap.results[0].prediction


@pytest.mark.uses_gpu
def test_puffin_interpretation_reverse_strand_actually_changes_output():
    """Switching reverse_strand produces a different prediction track than the forward strand."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinInterpretationConfig,
        PuffinInterpretationInput,
        run_puffin_interpretation,
    )

    inputs = PuffinInterpretationInput(sequences=[_random_dna(1650, seed=5)])
    fwd = run_puffin_interpretation(inputs, PuffinInterpretationConfig(reverse_strand=False))
    rev = run_puffin_interpretation(inputs, PuffinInterpretationConfig(reverse_strand=True))

    assert fwd.reverse_strand is False
    assert rev.reverse_strand is True
    assert fwd.results[0].prediction != rev.results[0].prediction


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("puffin-interpretation")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_puffin_interpretation_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark: 8 random 1650 bp sequences (cold + warm)."""
    from proto_tools.tools.sequence_scoring.puffin import (
        PuffinInterpretationConfig,
        PuffinInterpretationInput,
        run_puffin_interpretation,
    )

    inputs = PuffinInterpretationInput(sequences=random_dna_sequences(n=8, length=1650, seed=0))
    config = PuffinInterpretationConfig()
    result = benchmark_twice(request, "puffin", lambda: run_puffin_interpretation(inputs, config))

    validate_output(result)
    assert len(result.results) == 8
