"""Tests for the SpliceTransformer splice-channel gradient tool."""

import math
from typing import Any

import pytest
from pydantic import ValidationError

from proto_tools.tools import (
    SpliceTransformerGradientCandidate,
    SpliceTransformerGradientConfig,
    SpliceTransformerGradientInput,
    SpliceTransformerGradientLossTerm,
    SpliceTransformerGradientOutput,
    run_splice_transformer_gradient,
)
from proto_tools.tools.rna_splicing.splice_transformer.splice_transformer import (
    CONTEXT_LENGTH,
    SPLICE_TISSUE_CHANNEL_INDEX,
    TARGET_LENGTH,
)
from proto_tools.tools.rna_splicing.splice_transformer.splice_transformer_gradient import SPLICE_CHANNEL_INDEX
from tests.conftest import benchmark_twice
from tests.tool_infra_tests.test_export_functionality import validate_output


def _flank(length: int = CONTEXT_LENGTH) -> str:
    """A fixed flanking context of the exact published length."""
    return ("ACGT" * (length // 4 + 1))[:length]


def _candidate() -> SpliceTransformerGradientCandidate:
    """One candidate with a uniform relaxed target region between fixed contexts."""
    return SpliceTransformerGradientCandidate(
        left_context=_flank(),
        right_context=_flank(),
        logits=[[0.25, 0.25, 0.25, 0.25]] * TARGET_LENGTH,
    )


# ── Channel mapping ───────────────────────────────────────────────────────────


def test_splice_channel_index_matches_the_toolkits_authoritative_map():
    """The gradient's channel names must resolve to the same indices prediction uses.

    A drift here would silently optimize the wrong tissue, so it is pinned against the
    toolkit's own table rather than restated.
    """
    for tissue, index in SPLICE_TISSUE_CHANNEL_INDEX.items():
        if index is None:  # AVERAGE is a reduction, not a channel
            continue
        assert SPLICE_CHANNEL_INDEX[tissue] == index, f"{tissue} maps to the wrong channel"
    assert [SPLICE_CHANNEL_INDEX[name] for name in ("NEITHER", "ACCEPTOR", "DONOR")] == [0, 1, 2]
    assert len(SPLICE_CHANNEL_INDEX) == 18


# ── Validation ────────────────────────────────────────────────────────────────


def test_splice_transformer_gradient_candidate_requires_published_lengths():
    """Contexts and the target region must match what the checkpoint was trained on."""
    with pytest.raises(ValidationError, match=f"left_context must be exactly {CONTEXT_LENGTH}"):
        SpliceTransformerGradientCandidate(
            left_context="ACGT", right_context=_flank(), logits=[[0.25] * 4] * TARGET_LENGTH
        )
    with pytest.raises(ValidationError, match=f"right_context must be exactly {CONTEXT_LENGTH}"):
        SpliceTransformerGradientCandidate(
            left_context=_flank(), right_context="ACGT", logits=[[0.25] * 4] * TARGET_LENGTH
        )
    with pytest.raises(ValidationError, match=f"exactly {TARGET_LENGTH} target"):
        SpliceTransformerGradientCandidate(left_context=_flank(), right_context=_flank(), logits=[[0.25] * 4] * 10)


def test_splice_transformer_gradient_loss_term_rejects_inverted_span():
    """A term's span must be ordered when both ends are given."""
    with pytest.raises(ValidationError, match="precedes"):
        SpliceTransformerGradientLossTerm(position_start=800, position_end=100)


def test_splice_transformer_gradient_input_normalizes_a_single_candidate():
    """One bundle is accepted without wrapping it in a list."""
    assert len(SpliceTransformerGradientInput(inputs=_candidate()).inputs) == 1
    assert len(SpliceTransformerGradientInput(inputs=[_candidate(), _candidate()]).inputs) == 2


# ── Dispatch contract ─────────────────────────────────────────────────────────


def test_splice_transformer_gradient_dispatch_contract(monkeypatch):
    """The payload carries one item per candidate, resolved channel indices, and the context length."""
    captured: dict[str, Any] = {}

    def fake_dispatch(toolkit: str, payload: dict[str, Any], *, instance: Any = None, config: Any = None) -> Any:
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {
            "results": [
                {
                    "gradient": [[0.0] * 4] * len(item["logits"]),
                    "loss": -0.5,
                    "metrics": {"loss": -0.5, "target_length": len(item["logits"]), "loss_terms": []},
                }
                for item in payload["items"]
            ],
            "vocab": ["A", "C", "G", "T"],
        }

    monkeypatch.setattr(
        "proto_tools.tools.rna_splicing.splice_transformer.splice_transformer_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_splice_transformer_gradient(
        SpliceTransformerGradientInput(inputs=[_candidate(), _candidate()], temperature=0.9),
        SpliceTransformerGradientConfig(
            loss_terms=[
                SpliceTransformerGradientLossTerm(channel="DONOR", position_start=400, position_end=600),
                SpliceTransformerGradientLossTerm(channel="BRAIN", direction="min", weight=0.5),
            ],
            device="cuda",
        ),
    )

    assert captured["toolkit"] == "splice_transformer"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert [len(item["logits"]) for item in payload["items"]] == [TARGET_LENGTH, TARGET_LENGTH]
    assert len(payload["items"][0]["left_context"]) == CONTEXT_LENGTH
    assert payload["context_length"] == CONTEXT_LENGTH, "the worker builds its model from this"
    assert payload["temperature"] == 0.9
    assert [term["channel_index"] for term in payload["loss_terms"]] == [2, 6], "names resolve to model channels"
    assert payload["loss_terms"][1]["weight"] == 0.5
    assert len(result.results) == 2
    assert result.vocab == ["A", "C", "G", "T"]


# ── Integration ───────────────────────────────────────────────────────────────


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_splice_transformer_gradient_single_candidate():
    """A real checkpoint returns a finite gradient over the relaxed target region only."""
    result = run_splice_transformer_gradient(
        SpliceTransformerGradientInput(inputs=_candidate()),
        SpliceTransformerGradientConfig(
            loss_terms=[SpliceTransformerGradientLossTerm(channel="ACCEPTOR", position_start=480, position_end=520)]
        ),
    )
    validate_output(result)

    assert result.tool_id == "splice-transformer-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == TARGET_LENGTH, "gradient covers the target region, not the contexts"
    assert all(len(row) == 4 for row in candidate.gradient)
    assert all(math.isfinite(value) for row in candidate.gradient for value in row)
    assert any(value != 0.0 for row in candidate.gradient for value in row)
    assert candidate.metrics["target_length"] == TARGET_LENGTH
    (term,) = candidate.metrics["loss_terms"]
    assert term["channel"] == "ACCEPTOR"
    assert 0.0 <= term["mean_probability"] <= 1.0, "splice classes are softmaxed, so a probability"


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_splice_transformer_gradient_direction_flips_the_objective_sign():
    """`max` and `min` on the same channel differ only in the sign of the objective."""
    inputs = SpliceTransformerGradientInput(inputs=_candidate())
    term = {"channel": "DONOR", "position_start": 480, "position_end": 520}

    maximize = run_splice_transformer_gradient(
        inputs,
        SpliceTransformerGradientConfig(loss_terms=[SpliceTransformerGradientLossTerm(**term, direction="max")]),
    )
    minimize = run_splice_transformer_gradient(
        inputs,
        SpliceTransformerGradientConfig(loss_terms=[SpliceTransformerGradientLossTerm(**term, direction="min")]),
    )

    assert maximize.results[0].loss == pytest.approx(-minimize.results[0].loss, rel=1e-5, abs=1e-7)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_splice_transformer_gradient_forward_mode_matches_backward_loss():
    """compute_gradient=False returns the same objective without a gradient."""
    inputs = SpliceTransformerGradientInput(inputs=_candidate())
    terms = [SpliceTransformerGradientLossTerm(channel="ACCEPTOR")]

    backward = run_splice_transformer_gradient(inputs, SpliceTransformerGradientConfig(loss_terms=terms))
    forward = run_splice_transformer_gradient(
        inputs, SpliceTransformerGradientConfig(loss_terms=terms, compute_gradient=False)
    )

    assert forward.results[0].gradient is None
    assert backward.results[0].gradient is not None
    assert forward.results[0].loss == pytest.approx(backward.results[0].loss, rel=1e-5, abs=1e-7)


@pytest.mark.benchmark("splice-transformer-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_splice_transformer_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark splice-transformer-gradient: one backward pass over a 9,000 bp window (cold + warm)."""
    inputs = SpliceTransformerGradientInput(inputs=_candidate())
    config = SpliceTransformerGradientConfig(loss_terms=[SpliceTransformerGradientLossTerm(channel="ACCEPTOR")])

    result = benchmark_twice(request, "splice_transformer", lambda: run_splice_transformer_gradient(inputs, config))
    validate_output(result)

    assert isinstance(result, SpliceTransformerGradientOutput)
    assert result.tool_id == "splice-transformer-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == TARGET_LENGTH
