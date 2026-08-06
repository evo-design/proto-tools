"""Tests for causal model shared base classes."""

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleInput,
    CausalModelScoringInput,
)

# ── CausalModelSampleInput (renamed sequences → prompts) ────────────────────


def test_sample_input_normalizes_single_string():
    inp = CausalModelSampleInput(prompts="MVLSPADKTNVKAAW")
    assert inp.prompts == ["MVLSPADKTNVKAAW"]


# ── CausalModelScoringInput ──────────────────────────────────────────────────


def test_scoring_input_normalizes_single_string():
    inp = CausalModelScoringInput(sequences="MVLSPADKTNVKAAW")
    assert inp.sequences == ["MVLSPADKTNVKAAW"]


def test_scoring_input_preserves_list():
    inp = CausalModelScoringInput(sequences=["MVLSP", "GGGS"])
    assert inp.sequences == ["MVLSP", "GGGS"]
