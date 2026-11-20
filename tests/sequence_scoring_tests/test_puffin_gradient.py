"""Tests for the Puffin transcription-initiation gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    PuffinGradientConfig,
    PuffinGradientInput,
    PuffinGradientOutput,
    PuffinSignalTerm,
    run_puffin_gradient,
)
from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
    PUFFIN_MIN_INPUT_LENGTH,
    PUFFIN_PADDING,
    TRACK_NAMES,
)
from tests.conftest import benchmark_twice, make_persistent_fixture
from tests.tool_infra_tests.test_export_functionality import validate_output

_persistent_tool = make_persistent_fixture("puffin")

LENGTH = PUFFIN_MIN_INPUT_LENGTH + 99
OUTPUT_LENGTH = LENGTH - 2 * PUFFIN_PADDING


def _uniform(length: int = LENGTH) -> list[list[float]]:
    """A uniform relaxed state of the given length."""
    return [[0.25, 0.25, 0.25, 0.25]] * length


# ── Validation ────────────────────────────────────────────────────────────────


def test_puffin_gradient_rejects_sequences_below_one_output_position():
    """Puffin trims 325 bp from each end, so anything shorter scores nothing."""
    with pytest.raises(ValidationError, match=f"at least {PUFFIN_MIN_INPUT_LENGTH} bp"):
        PuffinGradientInput(inputs=_uniform(PUFFIN_MIN_INPUT_LENGTH - 1))

    assert len(PuffinGradientInput(inputs=_uniform(PUFFIN_MIN_INPUT_LENGTH)).inputs) == 1


def test_puffin_gradient_term_rejects_inverted_position_span():
    """A term's span must be ordered when both ends are given."""
    with pytest.raises(ValidationError, match="precedes"):
        PuffinSignalTerm(position_start=100, position_end=10)

    # An open-ended span is allowed and means "to the edge of the output window".
    assert PuffinSignalTerm(position_start=10).position_end is None


def test_puffin_gradient_rejects_the_same_quantity_twice():
    """One track and span is one quantity; asking for it twice is a mistake, not a doubling."""
    with pytest.raises(ValidationError, match="more than once"):
        PuffinGradientConfig(
            gradient_terms=[PuffinSignalTerm(track="GRO_CAP+"), PuffinSignalTerm(track="GRO_CAP+")],
        )


def test_puffin_gradient_input_normalizes_a_single_candidate():
    """One relaxed matrix is accepted without wrapping it in a list."""
    assert len(PuffinGradientInput(inputs=_uniform()).inputs) == 1
    assert len(PuffinGradientInput(inputs=[_uniform(), _uniform(LENGTH + 10)]).inputs) == 2


# ── Dispatch contract ─────────────────────────────────────────────────────────


def test_puffin_gradient_dispatch_contract(monkeypatch):
    """The payload carries one item per candidate plus the objective and the track ordering."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {
            "results": [
                {
                    "terms": [
                        {
                            # The worker resolves an open span against this candidate's window.
                            "term": {
                                "track": term["track"],
                                "position_start": term["position_start"] or 1,
                                "position_end": term["position_end"] or len(item) - 650,
                            },
                            "value": 0.25 + index,
                            "gradient": [[0.1 * (index + 1)] * 4] * len(item),
                        }
                        for index, term in enumerate(payload["gradient_terms"])
                    ],
                    "metrics": {"output_length": len(item) - 650},
                }
                for item in payload["items"]
            ],
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(
        "proto_tools.tools.sequence_scoring.puffin.puffin_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_puffin_gradient(
        PuffinGradientInput(inputs=[_uniform(), _uniform(LENGTH + 10)], temperature=0.8),
        PuffinGradientConfig(
            gradient_terms=[
                PuffinSignalTerm(track="GRO_CAP+", position_start=10, position_end=40),
                PuffinSignalTerm(track="GRO_CAP-"),
            ],
            device="cuda",
        ),
    )

    assert captured["toolkit"] == "puffin"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert [len(item) for item in payload["items"]] == [LENGTH, LENGTH + 10], "one item per candidate, in order"
    assert payload["temperature"] == 0.8
    assert [term["track"] for term in payload["gradient_terms"]] == ["GRO_CAP+", "GRO_CAP-"]
    assert payload["gradient_terms"][1]["position_start"] is None, "an open span stays open over the wire"
    assert payload["track_names"] == list(TRACK_NAMES), "channel order travels with the request"
    assert len(result.results) == 2
    assert result.vocab == ["A", "C", "G", "T"]

    # Each track comes back as its own term, and an open span is reported as the span averaged
    first, second = result.results[0].terms
    assert [(t.term.track, t.term.position_start, t.term.position_end) for t in (first, second)] == [
        ("GRO_CAP+", 10, 40),
        ("GRO_CAP-", 1, OUTPUT_LENGTH),
    ]
    assert (first.term_value, second.term_value) == (0.25, 1.25)
    assert first.metadata is not None
    assert first.metadata.positions == second.metadata.positions, "one basis shared across a candidate's terms"

    # A track can be driven up or suppressed depending on the design, so no direction is assumed
    with pytest.raises(ValueError, match="needs an explicit goal"):
        result.results[0].compose()


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_single_candidate():
    """A real checkpoint returns a finite gradient over the whole relaxed input."""
    result = run_puffin_gradient(
        PuffinGradientInput(inputs=_uniform()),
        PuffinGradientConfig(gradient_terms=[PuffinSignalTerm(track="FANTOM_CAGE+")]),
    )
    validate_output(result)

    assert result.tool_id == "puffin-gradient"
    (candidate,) = result.results
    (signal,) = candidate.terms
    assert signal.gradient is not None
    assert len(signal.gradient) == LENGTH, "Puffin scores every base, so the whole input carries gradient"
    assert all(len(row) == 4 for row in signal.gradient)
    assert all(math.isfinite(value) for row in signal.gradient for value in row)
    assert any(value != 0.0 for row in signal.gradient for value in row)
    assert math.isfinite(signal.term_value)
    assert candidate.metrics["output_length"] == OUTPUT_LENGTH
    assert signal.term.track == "FANTOM_CAGE+"
    assert signal.term.position_start == 1 and signal.term.position_end == OUTPUT_LENGTH


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_strands_are_distinct_channels():
    """The ``+``/``-`` suffix selects different model channels, not the same one twice."""
    both = run_puffin_gradient(
        PuffinGradientInput(inputs=_uniform()),
        PuffinGradientConfig(
            gradient_terms=[PuffinSignalTerm(track="FANTOM_CAGE+"), PuffinSignalTerm(track="FANTOM_CAGE-")]
        ),
    )

    forward, reverse = both.results[0].terms
    assert forward.term_value != pytest.approx(reverse.term_value, rel=1e-9), (
        "forward and reverse strand heads should not produce an identical signal"
    )
    assert forward.gradient != reverse.gradient, "distinct channels must give distinct gradients"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_forward_mode_matches_backward_signal():
    """compute_gradient=False returns the same objective without a gradient."""
    inputs = PuffinGradientInput(inputs=_uniform())
    terms = [PuffinSignalTerm(track="ENCODE_CAGE+")]

    backward = run_puffin_gradient(inputs, PuffinGradientConfig(gradient_terms=terms))
    forward = run_puffin_gradient(inputs, PuffinGradientConfig(gradient_terms=terms, compute_gradient=False))

    (with_gradient,) = backward.results[0].terms
    (without,) = forward.results[0].terms
    assert without.gradient is None
    assert with_gradient.gradient is not None
    assert without.term_value == pytest.approx(with_gradient.term_value, rel=1e-5, abs=1e-7)


@pytest.mark.benchmark("puffin-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_puffin_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark puffin-gradient: one backward pass over a 1,151 bp relaxed input (cold + warm)."""
    inputs = PuffinGradientInput(inputs=_uniform(PUFFIN_MIN_INPUT_LENGTH + 500))
    config = PuffinGradientConfig(gradient_terms=[PuffinSignalTerm(track="FANTOM_CAGE+")])

    result = benchmark_twice(request, "puffin", lambda: run_puffin_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, PuffinGradientOutput)
    assert result.tool_id == "puffin-gradient"
    (candidate,) = result.results
    (signal,) = candidate.terms
    assert signal.gradient is not None
    assert len(signal.gradient) == PUFFIN_MIN_INPUT_LENGTH + 500
