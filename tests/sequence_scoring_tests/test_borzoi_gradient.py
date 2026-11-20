"""Tests for the Borzoi track-and-bin activity gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    BorzoiActivityTerm,
    BorzoiGradientCandidate,
    BorzoiGradientConfig,
    BorzoiGradientInput,
    BorzoiGradientOutput,
    run_borzoi_gradient,
)
from proto_tools.tools.sequence_scoring.borzoi.borzoi_prediction import BORZOI_CONTEXT, BORZOI_OUTPUT
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("borzoi")

CENTER = BORZOI_CONTEXT // 2


def _context(length: int = BORZOI_CONTEXT) -> str:
    """A fixed, valid context window of the exact model length."""
    return ("ACGT" * (length // 4 + 1))[:length]


def _candidate(n_positions: int = 4) -> BorzoiGradientCandidate:
    """One candidate relaxing ``n_positions`` bases at the centre of the context."""
    return BorzoiGradientCandidate(
        sequence=_context(),
        positions=[CENTER + offset for offset in range(n_positions)],
        logits=[[0.25, 0.25, 0.25, 0.25]] * n_positions,
    )


# ── Validation ────────────────────────────────────────────────────────────────


def test_borzoi_gradient_candidate_requires_exact_context_length():
    """Borzoi reads a fixed window, so a short context is refused rather than padded."""
    with pytest.raises(ValidationError, match=f"exactly {BORZOI_CONTEXT} bp"):
        BorzoiGradientCandidate(sequence="ACGT" * 10, positions=[1], logits=[[0.25] * 4])


def test_borzoi_gradient_candidate_requires_aligned_positions():
    """Each relaxed row maps to one designed position, strictly increasing and inside the window."""
    with pytest.raises(ValidationError, match="must align"):
        BorzoiGradientCandidate(sequence=_context(), positions=[1, 2], logits=[[0.25] * 4])
    with pytest.raises(ValidationError, match="strictly increasing"):
        BorzoiGradientCandidate(sequence=_context(), positions=[5, 5], logits=[[0.25] * 4] * 2)
    with pytest.raises(ValidationError, match="exceeds"):
        BorzoiGradientCandidate(sequence=_context(), positions=[BORZOI_CONTEXT + 1], logits=[[0.25] * 4])


def test_borzoi_gradient_term_rejects_inverted_bin_span():
    """A term's bin span must be ordered, otherwise the mean would cover nothing."""
    with pytest.raises(ValidationError, match="precedes"):
        BorzoiActivityTerm(bin_start=100, bin_end=10)


def test_borzoi_gradient_input_normalizes_a_single_candidate():
    """One bundle is accepted without wrapping it in a list."""
    single = BorzoiGradientInput(inputs=_candidate())
    assert len(single.inputs) == 1
    assert len(BorzoiGradientInput(inputs=[_candidate(), _candidate(2)]).inputs) == 2


# ── Dispatch contract ─────────────────────────────────────────────────────────


def test_borzoi_gradient_dispatch_contract(monkeypatch):
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
        "proto_tools.tools.sequence_scoring.borzoi.borzoi_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_borzoi_gradient(
        BorzoiGradientInput(inputs=[_candidate(4), _candidate(2)], temperature=0.7),
        BorzoiGradientConfig(
            gradient_terms=[
                BorzoiActivityTerm(track=11, bin_start=3000, bin_end=3010),
                BorzoiActivityTerm(track=12),
            ],
            device="cuda",
        ),
    )

    assert captured["toolkit"] == "borzoi"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert [len(item["positions"]) for item in payload["items"]] == [4, 2], "one item per candidate, in order"
    assert payload["items"][0]["positions"][0] == CENTER
    assert len(payload["items"][0]["sequence"]) == BORZOI_CONTEXT
    assert payload["temperature"] == 0.7
    assert [term["track"] for term in payload["gradient_terms"]] == [11, 12]
    assert payload["gradient_terms"][1]["bin_end"] == BORZOI_OUTPUT
    assert payload["use_flash_attn"] is True, "human species uses FlashAttention, as prediction does"
    assert len(result.results) == 2
    assert result.vocab == ["A", "C", "G", "T"]

    # Each track comes back as its own term, carrying the selectors it was asked for
    first, second = result.results[0].terms
    assert [(t.term.track, t.term.bin_start, t.term.bin_end) for t in (first, second)] == [
        (11, 3000, 3010),
        (12, 1, BORZOI_OUTPUT),
    ]
    assert (first.term_value, second.term_value) == (1.5, 2.5)
    assert first.metadata is not None
    assert first.metadata.positions == result.results[0].terms[1].metadata.positions
    assert first.metadata.positions[0] == CENTER, "metadata records where in the context the rows sit"

    # A track can be a target or an off-target, so the tool declines to guess a direction
    with pytest.raises(ValueError, match="needs an explicit goal"):
        result.results[0].compose()


def test_borzoi_gradient_rejects_flash_attention_on_cpu():
    """The human model needs CUDA for FlashAttention; fail before dispatching."""
    with pytest.raises(ValueError, match="Must run on GPU"):
        run_borzoi_gradient(
            BorzoiGradientInput(inputs=_candidate()),
            BorzoiGradientConfig(species="human", device="cpu"),
        )


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_borzoi_gradient_single_candidate():
    """A real checkpoint returns a finite gradient covering only the designed positions."""
    result = run_borzoi_gradient(
        BorzoiGradientInput(inputs=_candidate(8)),
        BorzoiGradientConfig(gradient_terms=[BorzoiActivityTerm(track=0, bin_start=3060, bin_end=3085)]),
    )
    validate_output(result)

    assert result.tool_id == "borzoi-gradient"
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
def test_borzoi_gradient_forward_mode_matches_backward_activity():
    """compute_gradient=False returns the same activity without a gradient."""
    inputs = BorzoiGradientInput(inputs=_candidate(4))
    terms = [BorzoiActivityTerm(track=0, bin_start=3060, bin_end=3085)]

    backward = run_borzoi_gradient(inputs, BorzoiGradientConfig(gradient_terms=terms))
    forward = run_borzoi_gradient(inputs, BorzoiGradientConfig(gradient_terms=terms, compute_gradient=False))

    (with_gradient,) = backward.results[0].terms
    (without,) = forward.results[0].terms
    assert without.gradient is None
    assert with_gradient.gradient is not None
    assert without.term_value == pytest.approx(with_gradient.term_value, rel=1e-4, abs=1e-5)


@pytest.mark.benchmark("borzoi-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_borzoi_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark borzoi-gradient: one backward pass over the full 524,288 bp context (cold + warm)."""
    inputs = BorzoiGradientInput(inputs=_candidate(16))
    config = BorzoiGradientConfig(gradient_terms=[BorzoiActivityTerm(track=0, bin_start=3060, bin_end=3085)])

    result = benchmark_twice(request, "borzoi", lambda: run_borzoi_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, BorzoiGradientOutput)
    assert result.tool_id == "borzoi-gradient"
    (candidate,) = result.results
    (activity,) = candidate.terms
    assert activity.gradient is not None
    assert len(activity.gradient) == 16
