"""tests/rna_splicing_tests/test_rna_splicing.py.

Tests for SpliceTransformer.
"""

import pytest

from proto_tools.tools.rna_splicing.splice_transformer import (
    CONTEXT_LENGTH,
    TARGET_LENGTH,
    SpliceTransformerConfig,
    SpliceTransformerInput,
    run_splice_transformer,
)
from tests.conftest import benchmark_twice, random_dna_sequences
from tests.tool_infra_tests.test_export_functionality import validate_output

# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_splice_transformer_and_check(device: str) -> None:
    """Run SpliceTransformer on a poly-A sequence and verify output shape."""
    inp = SpliceTransformerInput(
        target_seqs=["A" * TARGET_LENGTH],
        left_contexts=["A" * CONTEXT_LENGTH],
        right_contexts=["A" * CONTEXT_LENGTH],
    )
    config = SpliceTransformerConfig(device=device)

    output = run_splice_transformer(inp, config)

    assert output.success is True, f"SpliceTransformer failed: {output}"
    assert len(output.prediction) == 1, f"Expected 1 batch, got {len(output.prediction)}"
    assert len(output.prediction[0]) == TARGET_LENGTH, (
        f"Expected target_length {TARGET_LENGTH}, got {len(output.prediction[0])}"
    )
    assert len(output.prediction[0][0]) == 18, f"Expected 18 channels, got {len(output.prediction[0][0])}"


# ---------------------------------------------------------------------------
# Validator coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,left,right,match",
    [
        ("A" * (TARGET_LENGTH - 1), "A" * CONTEXT_LENGTH, "A" * CONTEXT_LENGTH, "target_seqs"),
        ("A" * TARGET_LENGTH, "A" * (CONTEXT_LENGTH - 1), "A" * CONTEXT_LENGTH, "left_contexts"),
        ("A" * TARGET_LENGTH, "A" * CONTEXT_LENGTH, "A" * (CONTEXT_LENGTH + 1), "right_contexts"),
    ],
    ids=["target-too-short", "left-too-short", "right-too-long"],
)
def test_splice_transformer_input_rejects_wrong_lengths(target: str, left: str, right: str, match: str) -> None:
    """The model is pinned to 1000 bp targets and 4000 bp contexts; mismatches must raise pre-dispatch."""
    with pytest.raises(ValueError, match=match):
        SpliceTransformerInput(target_seqs=[target], left_contexts=[left], right_contexts=[right])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_splice_transformer_cpu() -> None:
    """Test SpliceTransformer on CPU."""
    _run_splice_transformer_and_check(device="cpu")


@pytest.mark.uses_gpu
def test_splice_transformer_gpu():
    """Test SpliceTransformer on GPU."""
    _run_splice_transformer_and_check(device="cuda")


# ── Benchmark ─────────────────────────────────────────────────────────────────


@pytest.mark.benchmark("splice-transformer-prediction")
@pytest.mark.slow
@pytest.mark.uses_gpu
def test_splice_transformer_prediction_benchmark(request: pytest.FixtureRequest) -> None:
    """Benchmark splice-transformer-prediction: 16 items at target=1000 + 4000 bp flanks each, batched (cold + warm)."""
    # Each item is 1000 bp target + 4000 bp left/right context = 9000 bp; 16 items
    # is a comfortable transformer batch on a modern GPU and exercises the
    # batched forward pass at realistic high-throughput screening scale.
    targets = random_dna_sequences(n=16, length=TARGET_LENGTH, seed=0)
    lefts = random_dna_sequences(n=16, length=CONTEXT_LENGTH, seed=1)
    rights = random_dna_sequences(n=16, length=CONTEXT_LENGTH, seed=2)

    inputs = SpliceTransformerInput(target_seqs=targets, left_contexts=lefts, right_contexts=rights)
    config = SpliceTransformerConfig()

    result = benchmark_twice(request, "splice_transformer", lambda: run_splice_transformer(inputs, config))
    validate_output(result)

    assert result.tool_id == "splice-transformer-prediction"
    assert len(result.prediction) == 16
    assert len(result.prediction[0]) == TARGET_LENGTH
    assert len(result.prediction[0][0]) == 18
