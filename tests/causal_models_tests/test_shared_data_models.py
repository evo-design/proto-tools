"""Tests for causal model shared base classes."""

import pytest

from proto_tools.tools.causal_models.shared_data_models import (
    CausalModelSampleConfig,
    CausalModelSampleInput,
    CausalModelScoringConfig,
    CausalModelScoringInput,
)

# ── CausalModelSampleInput (renamed sequences → prompts) ────────────────────


def test_sample_input_normalizes_single_string():
    inp = CausalModelSampleInput(prompts="MVLSPADKTNVKAAW")
    assert inp.prompts == ["MVLSPADKTNVKAAW"]


def test_sample_input_rejects_empty():
    with pytest.raises(ValueError, match="prompts must not be empty"):
        CausalModelSampleInput(prompts=[])


# ── CausalModelSampleConfig (new fields) ────────────────────────────────────


@pytest.mark.parametrize(
    "field,bad_value",
    [("temperature", 0.0), ("temperature", -1.0), ("top_p", 0.0), ("top_p", 1.5), ("batch_size", 0)],
)
def test_sample_config_rejects_invalid_values(field: str, bad_value: float):
    with pytest.raises(ValueError):
        CausalModelSampleConfig(**{field: bad_value})


# ── CausalModelScoringInput ──────────────────────────────────────────────────


def test_scoring_input_normalizes_single_string():
    inp = CausalModelScoringInput(sequences="MVLSPADKTNVKAAW")
    assert inp.sequences == ["MVLSPADKTNVKAAW"]


def test_scoring_input_preserves_list():
    inp = CausalModelScoringInput(sequences=["MVLSP", "GGGS"])
    assert inp.sequences == ["MVLSP", "GGGS"]


def test_scoring_input_rejects_empty():
    with pytest.raises(ValueError, match="sequences must not be empty"):
        CausalModelScoringInput(sequences=[])


# ── CausalModelScoringConfig ─────────────────────────────────────────────────


@pytest.mark.parametrize("field,bad_value", [("batch_size", 0), ("batch_size", -1)])
def test_scoring_config_rejects_invalid_values(field: str, bad_value: int):
    with pytest.raises(ValueError):
        CausalModelScoringConfig(**{field: bad_value})
