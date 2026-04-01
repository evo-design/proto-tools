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
    assert output.prediction.shape == (1, TARGET_LENGTH, 18), (
        f"Expected shape (1, {TARGET_LENGTH}, 18), got {output.prediction.shape}"
    )


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
