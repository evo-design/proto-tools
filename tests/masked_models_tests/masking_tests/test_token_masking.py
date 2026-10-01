"""Regression tests for atomic-token masking of mixed genomic sequences."""

import pytest

from proto_tools.transforms.masking import MaskingStrategy, RandomMaskingStrategy
from proto_tools.transforms.masking.maskers import split_tokens


@pytest.mark.parametrize("width,sequence", [(1, "ACDEFGHIKLMNPQRSTVWY"), (3, "ATGGCTTTTCAGAACGGT")])
@pytest.mark.parametrize("method", ["random", "entropy", "max-logit"])
def test_atomic_and_fixed_width_paths_agree(width, sequence, method):
    """Factoring selection does not change seeded protein or codon masking."""
    tokens = split_tokens(sequence, width)
    strategy = MaskingStrategy(method=method, mask_fraction=0.5, fixed_positions=[1])

    def logits(_sequences):
        return [[[float(i), 0.0, -1.0] for i in range(len(tokens))]]

    fixed_width = strategy.mask([sequence], position_score_fn=logits, token_size=width, seed=42)
    atomic = strategy.mask_tokens([tokens], position_score_fn=logits, seed=42)
    assert fixed_width == ["".join(atomic[0])]
    assert atomic[0][0] == tokens[0]
    assert len("".join(atomic[0])) == len(sequence)


def test_variable_width_tokens_preserve_protected_context():
    """Markers count once, remain ineligible, and do not shift fixed positions."""
    tokens = ["<+>", "M", "K", "<->", "a", "c", "X"]
    eligibility = [[False, True, True, False, True, True, False]]
    result = RandomMaskingStrategy(mask_fraction=1.0, fixed_positions=[3]).mask_tokens(
        [tokens], eligibility=eligibility, seed=7
    )
    assert result == [["<+>", "_", "K", "<->", "_", "_", "X"]]
    assert tokens == ["<+>", "M", "K", "<->", "a", "c", "X"]


def test_model_informed_masking_uses_atomic_logit_rows():
    """Entropy is aligned to tokens even when one token spans three characters."""
    tokens = ["<+>", "M", "a", "<->", "K"]
    seen = []

    def logits(sequences):
        seen.extend(sequences)
        return [[[0.0, 0.0], [30.0, 0.0], [0.0, 0.0], [0.0, 0.0], [30.0, 0.0]]]

    result = MaskingStrategy(method="entropy", temperature=0.001, num_mutations=1).mask_tokens(
        [tokens], eligibility=[[False, True, True, False, True]], position_score_fn=logits, seed=4
    )
    assert seen == ["<+>Ma<->K"]
    assert result == [["<+>", "M", "_", "<->", "K"]]


def test_atomic_masking_counts_only_eligible_sites():
    """Requested counts cannot consume protected marker or ambiguity positions."""
    with pytest.raises(ValueError, match="only 1 mutable"):
        RandomMaskingStrategy(num_mutations=2).mask_tokens([["<+>", "M", "X"]], eligibility=[[False, True, False]])
    assert RandomMaskingStrategy().mask_tokens([["<+>", "X"]], eligibility=[[False, False]]) == [["<+>", "X"]]


@pytest.mark.parametrize("eligibility", [[], [[True]]])
def test_atomic_masking_rejects_misaligned_eligibility(eligibility):
    with pytest.raises(ValueError, match=r"(one entry|token count)"):
        RandomMaskingStrategy().mask_tokens([["<+>", "M"]], eligibility=eligibility)


def test_atomic_masking_rejects_misaligned_model_logits():
    with pytest.raises(ValueError, match="token count"):
        MaskingStrategy(method="entropy").mask_tokens(
            [["<+>", "M"]], position_score_fn=lambda _sequences: [[[0.0, 0.0]]]
        )
