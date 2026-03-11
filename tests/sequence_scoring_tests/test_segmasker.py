"""Tests for Segmasker low-complexity region detection tool."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bio_programming_tools.tools.sequence_scoring.segmasker import (
    SegmaskerConfig,
    SegmaskerInput,
    run_segmasker,
)
from tests.tool_infra_tests.test_export_functionality import validate_output


# ── Validation ────────────────────────────────────────────────────────────────


def test_segmasker_input_rejects_missing_sequences():
    with pytest.raises(ValidationError, match="sequences"):
        SegmaskerInput()


def test_segmasker_input_rejects_empty_sequences():
    with pytest.raises(ValidationError, match="At least one sequence"):
        SegmaskerInput(sequences=[])


def test_segmasker_input_rejects_extra_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SegmaskerInput(sequences=["MKTL"], extra_field="x")


def test_segmasker_input_normalizes_single_string():
    inp = SegmaskerInput(sequences="MKTL")
    assert inp.sequences == ["MKTL"]


def test_segmasker_config_defaults():
    config = SegmaskerConfig()
    assert config.window == 15
    assert config.locut == 1.8
    assert config.hicut == 3.4


def test_segmasker_config_rejects_invalid_window():
    with pytest.raises(ValidationError, match="window"):
        SegmaskerConfig(window=0)


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.include_in_env_report(category="sequence_scoring")
def test_segmasker_scores_sequences():
    """Run segmasker on a mix of low-complexity and normal sequences."""
    inputs = SegmaskerInput(
        sequences=[
            "AAAAAAAAAAAAAAAAAAAAAAAAA",  # Low-complexity (polyA)
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",  # Hemoglobin
        ]
    )
    result = run_segmasker(inputs, SegmaskerConfig())

    validate_output(result)
    assert len(result.low_complexity_fractions) == 2
    assert len(result.low_complexity_counts) == 2
    assert len(result.sequence_lengths) == 2
    assert result.sequence_lengths[0] == 25
    assert result.sequence_lengths[1] == 50

    # PolyA should have higher low-complexity fraction than hemoglobin
    assert all(0.0 <= f <= 1.0 for f in result.low_complexity_fractions)
    assert result.low_complexity_fractions[0] >= result.low_complexity_fractions[1]

    # DataFrame should be populated
    assert result.results_df is not None
    assert len(result.results_df) == 2
