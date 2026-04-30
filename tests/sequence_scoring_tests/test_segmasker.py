"""tests/sequence_scoring_tests/test_segmasker.py.

Tests for Segmasker low-complexity region detection tool.
"""

import pytest
from pydantic import ValidationError

from proto_tools.tools.sequence_scoring.segmasker import (
    SegmaskerConfig,
    SegmaskerInput,
    run_segmasker,
)
from tests.conftest import benchmark_twice, random_protein_sequences
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
    assert len(result.results) == 2
    assert result.results[0].sequence_length == 25
    assert result.results[1].sequence_length == 51

    # PolyA should have higher low-complexity fraction than hemoglobin
    assert all(0.0 <= r.low_complexity_fraction <= 1.0 for r in result.results)
    assert result.results[0].low_complexity_fraction >= result.results[1].low_complexity_fraction

    # Every result has all three metrics populated
    assert all(r.low_complexity_count is not None for r in result.results)


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("segmasker-score")
@pytest.mark.slow
def test_segmasker_score_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark segmasker-score: 500 random 500-aa protein sequences on CPU (cold + warm)."""
    sequences = random_protein_sequences(n=500, length=500, seed=0)
    inputs = SegmaskerInput(sequences=sequences)
    config = SegmaskerConfig()

    result = benchmark_twice(request, "segmasker", lambda: run_segmasker(inputs, config))
    validate_output(result)

    assert result.tool_id == "segmasker-score"
    assert len(result.results) == 500
    for r in result.results:
        assert r.sequence_length == 500
        assert 0.0 <= r.low_complexity_fraction <= 1.0
