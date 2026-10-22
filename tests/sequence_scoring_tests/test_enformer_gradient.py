"""Tests for the Enformer track-and-bin activity gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    EnformerGradientCandidate,
    EnformerGradientConfig,
    EnformerGradientInput,
    EnformerGradientLossTerm,
    EnformerGradientOutput,
    run_enformer_gradient,
)
from proto_tools.tools.sequence_scoring.enformer.enformer_prediction import ENFORMER_CONTEXT, ENFORMER_OUTPUT
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

CENTER = ENFORMER_CONTEXT // 2


def _context(length: int = ENFORMER_CONTEXT) -> str:
    """A fixed, valid context window of the exact model length."""
    return ("ACGT" * (length // 4 + 1))[:length]


def _candidate(n_positions: int = 4) -> EnformerGradientCandidate:
    """One candidate relaxing ``n_positions`` bases at the centre of the context."""
    return EnformerGradientCandidate(
        sequence=_context(),
        positions=[CENTER + offset for offset in range(n_positions)],
        logits=[[0.25, 0.25, 0.25, 0.25]] * n_positions,
    )


# ── Validation ────────────────────────────────────────────────────────────────


def test_enformer_gradient_candidate_requires_exact_context_length():
    """Enformer reads a fixed window, so a short context is refused rather than padded."""
    with pytest.raises(ValidationError, match=f"exactly {ENFORMER_CONTEXT} bp"):
        EnformerGradientCandidate(sequence="ACGT" * 10, positions=[1], logits=[[0.25] * 4])


def test_enformer_gradient_candidate_requires_aligned_positions():
    """Each relaxed row maps to one designed position, strictly increasing and inside the window."""
    with pytest.raises(ValidationError, match="must align"):
        EnformerGradientCandidate(sequence=_context(), positions=[1, 2], logits=[[0.25] * 4])
    with pytest.raises(ValidationError, match="strictly increasing"):
        EnformerGradientCandidate(sequence=_context(), positions=[5, 5], logits=[[0.25] * 4] * 2)
    with pytest.raises(ValidationError, match="exceeds"):
        EnformerGradientCandidate(sequence=_context(), positions=[ENFORMER_CONTEXT + 1], logits=[[0.25] * 4])


def test_enformer_gradient_loss_term_rejects_inverted_bin_span():
    """A term's bin span must be ordered, otherwise the mean would cover nothing."""
    with pytest.raises(ValidationError, match="precedes"):
        EnformerGradientLossTerm(bin_start=100, bin_end=10)


def test_enformer_gradient_input_normalizes_a_single_candidate():
    """One bundle is accepted without wrapping it in a list."""
    single = EnformerGradientInput(inputs=_candidate())
    assert len(single.inputs) == 1
    assert len(EnformerGradientInput(inputs=[_candidate(), _candidate(2)]).inputs) == 2


# ── Dispatch contract ─────────────────────────────────────────────────────────


def test_enformer_gradient_dispatch_contract(monkeypatch):
    """The payload carries one item per candidate plus the shared objective and relaxation."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {
            "results": [
                {
                    "gradient": [[0.0] * 4] * len(item["positions"]),
                    "loss": -1.5,
                    "metrics": {"loss": -1.5, "designed_positions": len(item["positions"]), "loss_terms": []},
                }
                for item in payload["items"]
            ],
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(
        "proto_tools.tools.sequence_scoring.enformer.enformer_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_enformer_gradient(
        EnformerGradientInput(inputs=[_candidate(4), _candidate(2)], temperature=0.7),
        EnformerGradientConfig(
            loss_terms=[
                EnformerGradientLossTerm(track=11, bin_start=400, bin_end=410, direction="max", weight=2.0),
                EnformerGradientLossTerm(track=12, direction="min"),
            ],
            device="cuda",
        ),
    )

    assert captured["toolkit"] == "enformer"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert [len(item["positions"]) for item in payload["items"]] == [4, 2], "one item per candidate, in order"
    assert payload["items"][0]["positions"][0] == CENTER
    assert len(payload["items"][0]["sequence"]) == ENFORMER_CONTEXT
    assert payload["temperature"] == 0.7
    assert [term["track"] for term in payload["loss_terms"]] == [11, 12]
    assert payload["loss_terms"][0]["weight"] == 2.0
    assert payload["loss_terms"][1]["bin_end"] == ENFORMER_OUTPUT
    assert payload["species"] == "human"
    assert len(result.results) == 2
    assert result.vocab == ["A", "C", "G", "T"]


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_enformer_gradient_single_candidate():
    """A real checkpoint returns a finite gradient covering only the designed positions."""
    result = run_enformer_gradient(
        EnformerGradientInput(inputs=_candidate(8)),
        EnformerGradientConfig(loss_terms=[EnformerGradientLossTerm(track=0, bin_start=440, bin_end=456)]),
    )
    validate_output(result)

    assert result.tool_id == "enformer-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == 8, "one gradient row per designed position"
    assert all(len(row) == 4 for row in candidate.gradient)
    assert all(math.isfinite(value) for row in candidate.gradient for value in row)
    assert any(value != 0.0 for row in candidate.gradient for value in row)
    assert math.isfinite(candidate.loss)
    assert candidate.metrics["designed_positions"] == 8
    (term,) = candidate.metrics["loss_terms"]
    assert term["track"] == 0
    assert math.isfinite(term["mean_activity"])
    assert result.vocab == ["A", "C", "G", "T"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_enformer_gradient_direction_flips_the_objective_sign():
    """`max` and `min` on the same track differ only in the sign of the objective."""
    inputs = EnformerGradientInput(inputs=_candidate(4))
    term = {"track": 0, "bin_start": 440, "bin_end": 456}

    maximize = run_enformer_gradient(
        inputs, EnformerGradientConfig(loss_terms=[EnformerGradientLossTerm(**term, direction="max")])
    )
    minimize = run_enformer_gradient(
        inputs, EnformerGradientConfig(loss_terms=[EnformerGradientLossTerm(**term, direction="min")])
    )

    assert maximize.results[0].loss == pytest.approx(-minimize.results[0].loss, rel=1e-5, abs=1e-6)
    assert maximize.results[0].metrics["loss_terms"][0]["mean_activity"] == pytest.approx(
        minimize.results[0].metrics["loss_terms"][0]["mean_activity"], rel=1e-5
    )


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_enformer_gradient_forward_mode_matches_backward_loss():
    """compute_gradient=False returns the same objective without a gradient."""
    inputs = EnformerGradientInput(inputs=_candidate(4))
    terms = [EnformerGradientLossTerm(track=0, bin_start=440, bin_end=456)]

    backward = run_enformer_gradient(inputs, EnformerGradientConfig(loss_terms=terms))
    forward = run_enformer_gradient(inputs, EnformerGradientConfig(loss_terms=terms, compute_gradient=False))

    assert forward.results[0].gradient is None
    assert backward.results[0].gradient is not None
    assert forward.results[0].loss == pytest.approx(backward.results[0].loss, rel=1e-4, abs=1e-5)


@pytest.mark.benchmark("enformer-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_enformer_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark enformer-gradient: one backward pass over the full 196,608 bp context (cold + warm)."""
    inputs = EnformerGradientInput(inputs=_candidate(16))
    config = EnformerGradientConfig(loss_terms=[EnformerGradientLossTerm(track=0, bin_start=440, bin_end=456)])

    result = benchmark_twice(request, "enformer", lambda: run_enformer_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, EnformerGradientOutput)
    assert result.tool_id == "enformer-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == 16
