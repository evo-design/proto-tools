"""Tests for the PARADE 3' UTR stability gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    ParadeStabilityGradientConfig,
    ParadeStabilityGradientInput,
    ParadeStabilityGradientOutput,
    run_parade_stability_gradient,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("parade")

LENGTH = 50
GRADIENT_ROW = [0.1, -0.2, 0.3, -0.4]


def _uniform(length: int = LENGTH, n: int = 1) -> list[list[list[float]]]:
    """``n`` uniform relaxed states of the given length."""
    return [[[0.25, 0.25, 0.25, 0.25]] * length for _ in range(n)]


# ── Validation ────────────────────────────────────────────────────────────────


def test_parade_stability_gradient_requires_one_batch_length():
    """The backend runs one vectorized pass, so a ragged batch is refused up front."""
    with pytest.raises(ValidationError, match="same sequence length"):
        ParadeStabilityGradientInput(inputs=[[[0.25] * 4] * 10, [[0.25] * 4] * 11])


def test_parade_stability_gradient_input_normalizes_a_single_candidate():
    """One relaxed matrix is accepted without wrapping it in a list."""
    assert len(ParadeStabilityGradientInput(inputs=_uniform()[0]).inputs) == 1
    assert len(ParadeStabilityGradientInput(inputs=_uniform(n=3)).inputs) == 3


# ── Dispatch contract ─────────────────────────────────────────────────────────


def test_parade_stability_gradient_dispatch_contract(monkeypatch):
    """The payload targets the stability checkpoint and unbatches into per-candidate results."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        batch = len(payload["logits"])
        return {
            "log_ratios": [1.5, 2.5][:batch],
            "lengths": [LENGTH] * batch,
            "gradients": [[list(GRADIENT_ROW)] * LENGTH for _ in range(batch)],
        }

    monkeypatch.setattr(
        "proto_tools.tools.sequence_scoring.parade.parade_stability_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_parade_stability_gradient(
        ParadeStabilityGradientInput(inputs=_uniform(n=2), temperature=0.7),
        ParadeStabilityGradientConfig(device="cuda"),
    )

    assert captured["toolkit"] == "parade"
    payload = captured["payload"]
    assert payload["operation"] == "stability_gradient", "a distinct operation from the activity gradient"
    assert payload["temperature"] == 0.7
    assert "stability" in payload["checkpoint_filename"], "the stability checkpoint, not an activity one"
    assert "construct_type" not in payload, "stability is not conditioned on a UTR construct"
    assert "loss_terms" not in payload, "stability is a single scalar, not per-cell terms"

    assert len(result.results) == 2, "one result per candidate"
    assert [r.terms[0].term_value for r in result.results] == [1.5, 2.5], "the raw log-ratio, not a signed objective"
    assert result.results[0].metrics["sequence_length"] == LENGTH
    assert result.vocab == ["A", "C", "G", "T"]

    (term,) = result.results[0].terms
    assert term.term.id == "log_ratio"
    assert term.metadata.suggested_optimization_direction == "higher", "a higher log-ratio is more stable"

    # That advisory direction resolves a goal on its own, so composing needs no arguments.
    composed = result.results[0].compose()
    assert composed.contributions[0].dloss_dvalue == -1.0, "the advisory 'higher' resolved to maximize"
    assert composed.objective_loss == pytest.approx(-1.5), "the loss falls as the log-ratio rises"
    assert composed.gradient[0] == [-value for value in GRADIENT_ROW], "the raw gradient, negated"


def test_parade_stability_gradient_forward_mode_returns_no_gradient(monkeypatch):
    """A worker reply without gradients unbatches into results whose gradient is None."""

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        assert payload["compute_gradient"] is False
        return {"log_ratios": [1.0], "lengths": [LENGTH], "gradients": None}

    monkeypatch.setattr(
        "proto_tools.tools.sequence_scoring.parade.parade_stability_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_parade_stability_gradient(
        ParadeStabilityGradientInput(inputs=_uniform()[0]),
        ParadeStabilityGradientConfig(compute_gradient=False),
    )

    (term,) = result.results[0].terms
    assert term.gradient is None and term.grad_l2_norm is None
    assert term.term_value == 1.0, "forward-only keeps the scalar and drops only its derivative"


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_stability_gradient_single_candidate():
    """A real checkpoint returns a finite gradient and the objective tracks the log-ratio."""
    result = run_parade_stability_gradient(
        ParadeStabilityGradientInput(inputs=_uniform()[0]),
        ParadeStabilityGradientConfig(),
    )
    validate_output(result)

    assert result.tool_id == "parade-stability-gradient"
    (candidate,) = result.results
    (term,) = candidate.terms
    assert term.gradient is not None
    assert len(term.gradient) == LENGTH
    assert all(len(row) == 4 for row in term.gradient)
    assert all(math.isfinite(value) for row in term.gradient for value in row)
    assert any(value != 0.0 for row in term.gradient for value in row)
    assert math.isfinite(term.term_value)
    assert result.vocab == ["A", "C", "G", "T"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_stability_gradient_batch_matches_individual_evaluations():
    """Candidates are independent: a batched result matches evaluating each one alone.

    The model runs under ``eval()``, so its BatchNorm layers use running statistics and batching
    is mathematically batch-independent. The tolerance is loose because it is not bitwise: cuDNN
    selects different convolution algorithms for different batch shapes, which moves the result
    by a few parts in 10,000. A real cross-contamination bug would move it far more.
    """
    candidates = [
        [[0.7, 0.1, 0.1, 0.1]] * LENGTH,
        [[0.1, 0.7, 0.1, 0.1]] * LENGTH,
        [[0.25, 0.25, 0.25, 0.25]] * LENGTH,
    ]
    config = ParadeStabilityGradientConfig()

    batched = run_parade_stability_gradient(ParadeStabilityGradientInput(inputs=candidates), config)
    singles = [run_parade_stability_gradient(ParadeStabilityGradientInput(inputs=c), config) for c in candidates]

    assert len(batched.results) == 3
    for index, (together, alone) in enumerate(zip(batched.results, singles, strict=True)):
        assert together.terms[0].term_value == pytest.approx(
            alone.results[0].terms[0].term_value, rel=3e-3, abs=1e-4
        ), f"candidate {index} scored differently in a batch"


@pytest.mark.benchmark("parade-stability-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_parade_stability_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark parade-stability-gradient: one backward pass over a 16-candidate batch."""
    inputs = ParadeStabilityGradientInput(inputs=_uniform(length=200, n=16))
    config = ParadeStabilityGradientConfig()

    result = benchmark_twice(request, "parade", lambda: run_parade_stability_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, ParadeStabilityGradientOutput)
    assert result.tool_id == "parade-stability-gradient"
    assert len(result.results) == 16
    assert all(candidate.terms[0].gradient is not None for candidate in result.results)
