"""Tests for per-token eligibility in masking of mixed genomic sequences."""

import pytest

from proto_tools.transforms.masking import MaskingStrategy, RandomMaskingStrategy


def _eligibility(sequence):
    """Flag canonical protein and DNA tokens; markers and ambiguity codes are protected."""
    return [token in "ACDEFGHIKLMNPQRSTVWYacgt" for token in sequence]


def test_ineligible_tokens_and_fixed_positions_are_never_masked():
    sequence = "+MK-acX"
    result = RandomMaskingStrategy(mask_fraction=1.0, fixed_positions=[3]).mask(
        [sequence], eligibility=[_eligibility(sequence)], seed=7
    )
    assert result == ["+_K-__X"]


def test_mask_fraction_counts_only_eligible_tokens():
    sequence = "+MKLA-acgt"
    (result,) = RandomMaskingStrategy(mask_fraction=0.5).mask([sequence], eligibility=[_eligibility(sequence)], seed=3)
    assert result.count("_") == 4
    assert result[0] == "+"
    assert result[5] == "-"


def test_all_true_eligibility_matches_unrestricted_masking():
    strategy = RandomMaskingStrategy(mask_fraction=0.5)
    sequences = ["MKTLLIFLA", "ACDEFG"]
    unrestricted = strategy.mask(sequences, seed=11)
    restricted = strategy.mask(sequences, eligibility=[[True] * len(s) for s in sequences], seed=11)
    assert restricted == unrestricted


def test_eligibility_reduces_available_count():
    with pytest.raises(ValueError, match="only 1 mutable"):
        RandomMaskingStrategy(num_mutations=2).mask(["+MX"], eligibility=[[False, True, False]])
    assert RandomMaskingStrategy().mask(["+X"], eligibility=[[False, False]]) == ["+X"]


@pytest.mark.parametrize(
    ("eligibility", "error"),
    [([], "one row per sequence"), ([[True], [True]], "one row per sequence"), ([[True]], "one entry per token")],
)
def test_misaligned_eligibility_raises(eligibility, error):
    with pytest.raises(ValueError, match=error):
        RandomMaskingStrategy().mask(["+M"], eligibility=eligibility)


def test_model_scored_masking_receives_sequences_and_skips_ineligible_sites():
    sequence = "+Ma-K"
    seen = []

    def logits(sequences):
        seen.extend(sequences)
        # Uniform rows on markers would win on entropy if eligibility were ignored.
        return [[[0.0, 0.0], [30.0, 0.0], [0.0, 0.0], [0.0, 0.0], [30.0, 0.0]]]

    result = MaskingStrategy(method="entropy", temperature=0.001, num_mutations=1).mask(
        [sequence], eligibility=[_eligibility(sequence)], position_score_fn=logits, seed=4
    )
    assert seen == [sequence]
    assert result == ["+M_-K"]
