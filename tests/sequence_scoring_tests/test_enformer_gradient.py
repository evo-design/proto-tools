"""Tests for the Enformer track-and-bin activity gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    EnformerActivityTerm,
    EnformerGradientCandidate,
    EnformerGradientConfig,
    EnformerGradientInput,
    EnformerGradientOutput,
    run_enformer_gradient,
)
from proto_tools.tools.sequence_scoring.enformer.enformer_prediction import ENFORMER_CONTEXT, ENFORMER_OUTPUT
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("enformer")

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


def test_enformer_gradient_term_rejects_inverted_bin_span():
    """A term's bin span must be ordered, otherwise the mean would cover nothing."""
    with pytest.raises(ValidationError, match="precedes"):
        EnformerActivityTerm(bin_start=100, bin_end=10)


def test_enformer_gradient_rejects_the_same_quantity_twice():
    """One track and bin span is one quantity; asking for it twice is a mistake, not a doubling."""
    with pytest.raises(ValidationError, match="more than once"):
        EnformerGradientConfig(
            gradient_terms=[EnformerActivityTerm(track=3), EnformerActivityTerm(track=3)],
        )


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
                    "terms": [
                        {
                            "term": {k: term[k] for k in ("track", "bin_start", "bin_end")},
                            "value": 1.5 + index,
                            "gradient": [[0.1 * (index + 1)] * 4] * len(item["positions"]),
                        }
                        for index, term in enumerate(payload["gradient_terms"])
                    ],
                    "metrics": {"designed_positions": len(item["positions"])},
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
            gradient_terms=[
                EnformerActivityTerm(track=11, bin_start=400, bin_end=410),
                EnformerActivityTerm(track=12),
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
    assert [term["track"] for term in payload["gradient_terms"]] == [11, 12]
    assert payload["gradient_terms"][1]["bin_end"] == ENFORMER_OUTPUT
    assert payload["species"] == "human"
    assert len(result.results) == 2
    assert result.vocab == ["A", "C", "G", "T"]

    # Each track comes back as its own term, carrying the selectors it was asked for
    first, second = result.results[0].terms
    assert [(t.term.track, t.term.bin_start, t.term.bin_end) for t in (first, second)] == [
        (11, 400, 410),
        (12, 1, ENFORMER_OUTPUT),
    ]
    assert (first.term_value, second.term_value) == (1.5, 2.5)
    assert first.metadata is not None
    assert first.metadata.positions == result.results[0].terms[1].metadata.positions
    assert first.metadata.positions[0] == CENTER, "metadata records where in the context the rows sit"

    # A track can be a target or an off-target, so the tool declines to guess a direction
    with pytest.raises(ValueError, match="needs an explicit goal"):
        result.results[0].compose()


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_enformer_gradient_single_candidate():
    """A real checkpoint returns a finite gradient covering only the designed positions."""
    result = run_enformer_gradient(
        EnformerGradientInput(inputs=_candidate(8)),
        EnformerGradientConfig(gradient_terms=[EnformerActivityTerm(track=0, bin_start=440, bin_end=456)]),
    )
    validate_output(result)

    assert result.tool_id == "enformer-gradient"
    (candidate,) = result.results
    (activity,) = candidate.terms
    assert activity.gradient is not None
    assert len(activity.gradient) == 8, "one gradient row per designed position"
    assert all(len(row) == 4 for row in activity.gradient)
    assert all(math.isfinite(value) for row in activity.gradient for value in row)
    assert any(value != 0.0 for row in activity.gradient for value in row)
    assert math.isfinite(activity.term_value)
    assert activity.term.track == 0
    assert candidate.metrics["designed_positions"] == 8
    assert result.vocab == ["A", "C", "G", "T"]


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_enformer_gradient_forward_mode_matches_backward_activity():
    """compute_gradient=False returns the same activity without a gradient."""
    inputs = EnformerGradientInput(inputs=_candidate(4))
    terms = [EnformerActivityTerm(track=0, bin_start=440, bin_end=456)]

    backward = run_enformer_gradient(inputs, EnformerGradientConfig(gradient_terms=terms))
    forward = run_enformer_gradient(inputs, EnformerGradientConfig(gradient_terms=terms, compute_gradient=False))

    (with_gradient,) = backward.results[0].terms
    (without,) = forward.results[0].terms
    assert without.gradient is None
    assert with_gradient.gradient is not None
    assert without.term_value == pytest.approx(with_gradient.term_value, rel=1e-4, abs=1e-5)


@pytest.mark.benchmark("enformer-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_enformer_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark enformer-gradient: one backward pass over the full 196,608 bp context (cold + warm)."""
    inputs = EnformerGradientInput(inputs=_candidate(16))
    config = EnformerGradientConfig(gradient_terms=[EnformerActivityTerm(track=0, bin_start=440, bin_end=456)])

    result = benchmark_twice(request, "enformer", lambda: run_enformer_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, EnformerGradientOutput)
    assert result.tool_id == "enformer-gradient"
    (candidate,) = result.results
    (activity,) = candidate.terms
    assert activity.gradient is not None
    assert len(activity.gradient) == 16
