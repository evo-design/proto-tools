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
