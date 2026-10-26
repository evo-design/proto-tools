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
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

LENGTH = 50


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
            "losses": [-1.5, -2.5][:batch],
            "log_ratios": [1.5, 2.5][:batch],
            "lengths": [LENGTH] * batch,
            "gradients": [[[0.0] * 4] * LENGTH for _ in range(batch)],
        }

    monkeypatch.setattr(
        "proto_tools.tools.sequence_scoring.parade.parade_stability_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_parade_stability_gradient(
        ParadeStabilityGradientInput(inputs=_uniform(n=2), temperature=0.7),
        ParadeStabilityGradientConfig(direction="max", device="cuda"),
    )

    assert captured["toolkit"] == "parade"
    payload = captured["payload"]
    assert payload["operation"] == "stability_gradient", "a distinct operation from the activity gradient"
    assert payload["direction"] == "max"
    assert payload["temperature"] == 0.7
    assert "stability" in payload["checkpoint_filename"], "the stability checkpoint, not an activity one"
    assert "construct_type" not in payload, "stability is not conditioned on a UTR construct"
    assert "loss_terms" not in payload, "stability is a single scalar, not per-cell terms"

    assert len(result.results) == 2, "one result per candidate"
    assert [r.loss for r in result.results] == [-1.5, -2.5]
    assert [r.metrics["log_ratio"] for r in result.results] == [1.5, 2.5]
    assert result.results[0].metrics["sequence_length"] == LENGTH
    assert result.vocab == ["A", "C", "G", "T"]


def test_parade_stability_gradient_forward_mode_returns_no_gradient(monkeypatch):
    """A worker reply without gradients unbatches into results whose gradient is None."""

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        assert payload["compute_gradient"] is False
        return {"losses": [-1.0], "log_ratios": [1.0], "lengths": [LENGTH], "gradients": None}

    monkeypatch.setattr(
        "proto_tools.tools.sequence_scoring.parade.parade_stability_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_parade_stability_gradient(
        ParadeStabilityGradientInput(inputs=_uniform()[0]),
        ParadeStabilityGradientConfig(compute_gradient=False),
    )

    assert result.results[0].gradient is None
    assert result.results[0].loss == -1.0


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_stability_gradient_single_candidate():
    """A real checkpoint returns a finite gradient and the objective tracks the log-ratio."""
    result = run_parade_stability_gradient(
        ParadeStabilityGradientInput(inputs=_uniform()[0]),
        ParadeStabilityGradientConfig(direction="max"),
    )
    validate_output(result)

    assert result.tool_id == "parade-stability-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == LENGTH
    assert all(len(row) == 4 for row in candidate.gradient)
    assert all(math.isfinite(value) for row in candidate.gradient for value in row)
    assert any(value != 0.0 for row in candidate.gradient for value in row)
    assert math.isfinite(candidate.metrics["log_ratio"])
    assert candidate.loss == pytest.approx(-candidate.metrics["log_ratio"], rel=1e-6), (
        "maximizing stability minimizes the negated log-ratio"
    )
    assert result.vocab == ["A", "C", "G", "T"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_parade_stability_gradient_direction_flips_the_objective_sign():
    """`max` and `min` differ only in the sign of the objective and its gradient."""
    inputs = ParadeStabilityGradientInput(inputs=_uniform()[0])

    maximize = run_parade_stability_gradient(inputs, ParadeStabilityGradientConfig(direction="max"))
    minimize = run_parade_stability_gradient(inputs, ParadeStabilityGradientConfig(direction="min"))

    assert maximize.results[0].loss == pytest.approx(-minimize.results[0].loss, rel=1e-5, abs=1e-7)
    assert maximize.results[0].metrics["log_ratio"] == pytest.approx(minimize.results[0].metrics["log_ratio"], rel=1e-5)
    assert maximize.results[0].gradient[0][0] == pytest.approx(-minimize.results[0].gradient[0][0], rel=1e-4, abs=1e-7)


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
        assert together.metrics["log_ratio"] == pytest.approx(
            alone.results[0].metrics["log_ratio"], rel=3e-3, abs=1e-4
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
    assert all(candidate.gradient is not None for candidate in result.results)
