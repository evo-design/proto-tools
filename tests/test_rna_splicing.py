"""
test_rna_splicing.py

Test models for predicting RNA splicing (e.g., SpliceTransformer).
"""

import pytest

from bio_programming_tools.tools.rna_splicing.splice_transformer import (
    CONTEXT_LENGTH as SPLICE_TRANSFORMER_CONTEXT_LENGTH,
)
from bio_programming_tools.tools.rna_splicing.splice_transformer import (
    TARGET_LENGTH as SPLICE_TRANSFORMER_TARGET_LENGTH,
)
from bio_programming_tools.tools.rna_splicing.splice_transformer import (
    SpliceTransformerConfig,
    SpliceTransformerInput,
    run_splice_transformer,
)


def _test_splice_transformer(device: str) -> None:
    """Simply tests that SpliceTransformer can be run without issue."""
    splice_transformer_input = SpliceTransformerInput(
        target_seqs=["A" * SPLICE_TRANSFORMER_TARGET_LENGTH],
        left_contexts=["A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH],
        right_contexts=["A" * SPLICE_TRANSFORMER_CONTEXT_LENGTH],
    )
    splice_transformer_config = SpliceTransformerConfig(device=device)

    output = run_splice_transformer(
        splice_transformer_input,
        splice_transformer_config,
    )

    assert output.success is True, f"SpliceTransformer failed: {output}"

    assert len(output.prediction.shape) == 3, "Expected three dimensions for Splice Transformer output"
    assert output.prediction.shape[0] == 1, "Expected batch size of 1 for Splice Transformer output"
    assert output.prediction.shape[1] == SPLICE_TRANSFORMER_TARGET_LENGTH, "Expected dimension 2 to be the target length"
    assert output.prediction.shape[2] == 18, "Expected 18 dimensions in dimension 3"


@pytest.mark.skip_ci
def test_splice_transformer_cpu() -> None:
    """Test SpliceTransformer on cpu."""
    _test_splice_transformer(device="cpu")


@pytest.mark.uses_gpu
def test_splice_transformer_gpu():
    """Test SpliceTransformer on gpu."""
    _test_splice_transformer(device="cuda")
