"""Tests for the ESM C masked pseudo-log-likelihood gradient tool."""

import math
from pathlib import Path

import numpy as np
import pytest

from proto_tools.tools import (
    ESMCGradientConfig,
    ESMCGradientInput,
    ESMCGradientOutput,
    run_esmc_gradient,
)
from proto_tools.utils import one_hot_protein_logits
from proto_tools.utils.standalone_helpers_source.standalone_helpers.vocabulary import AMINO_ACIDS_LIST
from tests.conftest import benchmark_twice, random_protein_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

VH_SEQ = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKG"


def test_esmc_gradient_dispatch_contract(monkeypatch):
    """The payload carries one item per candidate plus the shared relaxation controls."""
    captured: dict[str, object] = {}

    def fake_dispatch(toolkit, payload, *, instance=None, config=None):
        captured["toolkit"] = toolkit
        captured["payload"] = payload
        return {
            "results": [
                {
                    "gradient": [[0.0] * 20] * len(item),
                    "loss": 0.5,
                    "metrics": {"log_likelihood": -1.0, "avg_log_likelihood": -0.5, "perplexity": np.exp(0.5)},
                }
                for item in payload["items"]
            ],
            "vocab": AMINO_ACIDS_LIST,
        }

    monkeypatch.setattr(
        "proto_tools.tools.masked_models.esmc.esmc_gradient.ToolInstance.dispatch",
        fake_dispatch,
    )

    result = run_esmc_gradient(
        ESMCGradientInput(inputs=[[[0.0] * 20] * 3, [[0.0] * 20] * 4], temperature=0.6),
        ESMCGradientConfig(model_checkpoint="esmc_300m", hard=1.0, batch_size=2, device="cpu"),
    )

    assert captured["toolkit"] == "esmc"
    payload = captured["payload"]
    assert payload["operation"] == "compute_gradient"
    assert [len(item) for item in payload["items"]] == [3, 4], "one payload item per candidate, in order"
    assert payload["temperature"] == 0.6
    assert payload["soft"] == 1.0
    assert payload["hard"] == 1.0
    assert payload["compute_gradient"] is True
    assert payload["batch_size"] == 2
    assert payload["model_checkpoint"] == "esmc_300m"
    assert len(result.results) == 2, "one result per candidate"
    assert result.vocab == AMINO_ACIDS_LIST


@pytest.mark.integration
def test_esmc_gradient_token_layout_matches_the_tokenizer() -> None:
    """The worker's hand-built ``[cls] + residues + [eos]`` must equal the tokenizer's output.

    ``compute_gradient`` assembles the model input itself instead of tokenizing, so a drift in
    the special-token layout would score a sequence the model never sees while every other test
    still passed — the finite-difference check only proves the derivative matches the objective,
    not that the objective is the right one. ``_esmc_token_layout_env_check.py`` runs inside the
    shared ESM env, where the tokenizer lives; a failed assertion surfaces as ``RuntimeError``.
    """
    from proto_tools.utils import run_in_env

    check = Path(__file__).resolve().parent / "_esmc_token_layout_env_check.py"
    output = run_in_env("esmc", script=str(check), timeout=1800)
    assert "ESMC_TOKEN_LAYOUT_OK" in output, output


@pytest.mark.uses_gpu
def test_esmc_gradient_single_sequence():
    """A real checkpoint returns a finite gradient shaped like the input and consistent metrics."""
    logits = one_hot_protein_logits(VH_SEQ[:24], sharpness=2.0)

    result = run_esmc_gradient(ESMCGradientInput(inputs=logits, temperature=0.6), ESMCGradientConfig())
    validate_output(result)

    assert result.tool_id == "esmc-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == 24
    assert all(len(row) == 20 for row in candidate.gradient)
    assert all(math.isfinite(value) for row in candidate.gradient for value in row)
    assert any(value != 0.0 for row in candidate.gradient for value in row)
    assert candidate.loss > 0 and math.isfinite(candidate.loss)
    assert candidate.metrics["avg_log_likelihood"] == pytest.approx(-candidate.loss, rel=1e-6)
    assert candidate.metrics["perplexity"] == pytest.approx(math.exp(candidate.loss), rel=1e-6)
    assert candidate.metrics["sequence_length"] == 24
    assert candidate.metrics["objective"] == "masked_pll"
    assert result.vocab == AMINO_ACIDS_LIST


@pytest.mark.uses_gpu
def test_esmc_gradient_forward_mode_matches_backward_loss():
    """compute_gradient=False returns the same objective without a gradient."""
    inputs = ESMCGradientInput(inputs=one_hot_protein_logits(VH_SEQ[:20], sharpness=2.0), temperature=0.6)

    backward = run_esmc_gradient(inputs, ESMCGradientConfig())
    forward = run_esmc_gradient(inputs, ESMCGradientConfig(compute_gradient=False))

    assert forward.results[0].gradient is None
    assert backward.results[0].gradient is not None
    assert forward.results[0].loss == pytest.approx(backward.results[0].loss, rel=2e-5, abs=2e-5)


@pytest.mark.uses_gpu
@pytest.mark.slow
def test_esmc_gradient_matches_finite_difference():
    """The returned derivative agrees with a central difference of the objective."""
    sequence = VH_SEQ[:12]
    logits = np.asarray(one_hot_protein_logits(sequence, sharpness=2.0))
    temperature = 0.9

    analytic = run_esmc_gradient(
        ESMCGradientInput(inputs=logits.tolist(), temperature=temperature), ESMCGradientConfig()
    )

    epsilon = 0.02
    losses = []
    for delta in (epsilon, -epsilon):
        perturbed = logits.copy()
        perturbed[3, 7] += delta
        losses.append(
            run_esmc_gradient(
                ESMCGradientInput(inputs=perturbed.tolist(), temperature=temperature),
                ESMCGradientConfig(compute_gradient=False),
            )
            .results[0]
            .loss
        )

    derivative = (losses[0] - losses[1]) / (2 * epsilon)
    assert analytic.results[0].gradient[3][7] == pytest.approx(derivative, rel=0.05, abs=5e-4)


@pytest.mark.uses_gpu
def test_esmc_gradient_batch_matches_individual_evaluations():
    """Candidates are independent: a batched result equals evaluating each one alone."""
    candidates = [
        one_hot_protein_logits(VH_SEQ[:18], sharpness=2.0),
        one_hot_protein_logits(VH_SEQ[:18], sharpness=1.0),
        one_hot_protein_logits(VH_SEQ[4:22], sharpness=2.0),
    ]
    config = ESMCGradientConfig()

    batched = run_esmc_gradient(ESMCGradientInput(inputs=candidates), config)
    singles = [run_esmc_gradient(ESMCGradientInput(inputs=candidate), config) for candidate in candidates]

    assert len(batched.results) == len(candidates)
    for index, (together, alone) in enumerate(zip(batched.results, singles, strict=True)):
        assert together.loss == pytest.approx(alone.results[0].loss, rel=1e-5, abs=1e-6), (
            f"candidate {index} scored differently in a batch"
        )


@pytest.mark.benchmark("esmc-gradient")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_esmc_gradient_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark esmc-gradient: 10 backward passes over a length-300 logits matrix (cold + warm)."""
    sequence = random_protein_sequences(n=1, length=300, seed=5)[0]
    inputs = ESMCGradientInput(inputs=one_hot_protein_logits(sequence, sharpness=2.0), temperature=0.6)
    config = ESMCGradientConfig(model_checkpoint="esmc_300m", batch_size=32)

    def run() -> ESMCGradientOutput:
        last: ESMCGradientOutput | None = None
        for _ in range(10):
            last = run_esmc_gradient(inputs, config)
        assert last is not None
        return last

    result = benchmark_twice(request, "esmc", run)
    validate_output(result)

    assert result.tool_id == "esmc-gradient"
    (candidate,) = result.results
    assert candidate.gradient is not None
    assert len(candidate.gradient) == 300
    assert all(len(row) == 20 for row in candidate.gradient)
    assert candidate.loss > 0
