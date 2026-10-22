"""Tests for the Puffin transcription-initiation gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    PuffinGradientConfig,
    PuffinGradientInput,
    PuffinGradientLossTerm,
    PuffinGradientOutput,
    run_puffin_gradient,
)
from proto_tools.tools.sequence_scoring.puffin.puffin_prediction import (
    PUFFIN_MIN_INPUT_LENGTH,
    PUFFIN_PADDING,
    TRACK_NAMES,
)
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output

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


def test_puffin_gradient_loss_term_rejects_inverted_position_span():
    """A term's span must be ordered when both ends are given."""
    with pytest.raises(ValidationError, match="precedes"):
        PuffinGradientLossTerm(position_start=100, position_end=10)

    # An open-ended span is allowed and means "to the edge of the output window".
    assert PuffinGradientLossTerm(position_start=10).position_end is None


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
                    "gradient": [[0.0] * 4] * len(item),
                    "loss": -0.25,
                    "metrics": {"loss": -0.25, "output_length": len(item) - 650, "loss_terms": []},
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
            loss_terms=[
                PuffinGradientLossTerm(track="GRO_CAP+", position_start=10, position_end=40, weight=2.0),
                PuffinGradientLossTerm(track="GRO_CAP-", direction="min"),
            ],
            device="cuda",
        ),
    )

    assert captured["toolkit"] == "puffin"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert [len(item) for item in payload["items"]] == [LENGTH, LENGTH + 10], "one item per candidate, in order"
    assert payload["temperature"] == 0.8
    assert [term["track"] for term in payload["loss_terms"]] == ["GRO_CAP+", "GRO_CAP-"]
    assert payload["loss_terms"][0]["weight"] == 2.0
    assert payload["loss_terms"][1]["position_start"] is None, "an open span stays open over the wire"
    assert payload["track_names"] == list(TRACK_NAMES), "channel order travels with the request"
    assert len(result.results) == 2
    assert result.vocab == ["A", "C", "G", "T"]


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_single_candidate():
    """A real checkpoint returns a finite gradient over the whole relaxed input."""
    result = run_puffin_gradient(
        PuffinGradientInput(inputs=_uniform()),
        PuffinGradientConfig(loss_terms=[PuffinGradientLossTerm(track="FANTOM_CAGE+")]),
    )
    validate_output(result)

    assert result.tool_id == "puffin-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == LENGTH, "Puffin scores every base, so the whole input carries gradient"
    assert all(len(row) == 4 for row in candidate.gradient)
    assert all(math.isfinite(value) for row in candidate.gradient for value in row)
    assert any(value != 0.0 for row in candidate.gradient for value in row)
    assert math.isfinite(candidate.loss)
    assert candidate.metrics["output_length"] == OUTPUT_LENGTH
    (term,) = candidate.metrics["loss_terms"]
    assert term["track"] == "FANTOM_CAGE+"
    assert term["position_start"] == 1 and term["position_end"] == OUTPUT_LENGTH
    assert math.isfinite(term["mean_signal"])


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_strands_are_distinct_channels():
    """The ``+``/``-`` suffix selects different model channels, not the same one twice."""
    forward = run_puffin_gradient(
        PuffinGradientInput(inputs=_uniform()),
        PuffinGradientConfig(loss_terms=[PuffinGradientLossTerm(track="FANTOM_CAGE+")]),
    )
    reverse = run_puffin_gradient(
        PuffinGradientInput(inputs=_uniform()),
        PuffinGradientConfig(loss_terms=[PuffinGradientLossTerm(track="FANTOM_CAGE-")]),
    )

    assert forward.results[0].metrics["loss_terms"][0]["mean_signal"] != pytest.approx(
        reverse.results[0].metrics["loss_terms"][0]["mean_signal"], rel=1e-9
    ), "forward and reverse strand heads should not produce an identical signal"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_direction_flips_the_objective_sign():
    """`max` and `min` on the same track differ only in the sign of the objective."""
    inputs = PuffinGradientInput(inputs=_uniform())

    maximize = run_puffin_gradient(
        inputs, PuffinGradientConfig(loss_terms=[PuffinGradientLossTerm(track="PRO_CAP+", direction="max")])
    )
    minimize = run_puffin_gradient(
        inputs, PuffinGradientConfig(loss_terms=[PuffinGradientLossTerm(track="PRO_CAP+", direction="min")])
    )

    assert maximize.results[0].loss == pytest.approx(-minimize.results[0].loss, rel=1e-5, abs=1e-7)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_puffin_gradient_forward_mode_matches_backward_loss():
    """compute_gradient=False returns the same objective without a gradient."""
    inputs = PuffinGradientInput(inputs=_uniform())
    terms = [PuffinGradientLossTerm(track="ENCODE_CAGE+")]

    backward = run_puffin_gradient(inputs, PuffinGradientConfig(loss_terms=terms))
    forward = run_puffin_gradient(inputs, PuffinGradientConfig(loss_terms=terms, compute_gradient=False))

    assert forward.results[0].gradient is None
    assert backward.results[0].gradient is not None
    assert forward.results[0].loss == pytest.approx(backward.results[0].loss, rel=1e-5, abs=1e-7)


@pytest.mark.benchmark("puffin-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_puffin_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark puffin-gradient: one backward pass over a 1,151 bp relaxed input (cold + warm)."""
    inputs = PuffinGradientInput(inputs=_uniform(PUFFIN_MIN_INPUT_LENGTH + 500))
    config = PuffinGradientConfig(loss_terms=[PuffinGradientLossTerm(track="FANTOM_CAGE+")])

    result = benchmark_twice(request, "puffin", lambda: run_puffin_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, PuffinGradientOutput)
    assert result.tool_id == "puffin-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == PUFFIN_MIN_INPUT_LENGTH + 500
