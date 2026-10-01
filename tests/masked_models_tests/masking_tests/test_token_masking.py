"""Tests for protected tokens in masking of mixed genomic sequences."""

import pytest

from proto_tools.transforms.masking import MaskingStrategy, RandomMaskingStrategy

# Strand markers and ambiguous protein symbols, as protected by the mixed-model samplers.
_PROTECTED = frozenset("+-XBUZO")


def test_protected_tokens_and_fixed_positions_are_never_masked():
    result = RandomMaskingStrategy(mask_fraction=1.0, fixed_positions=[3]).mask(
        ["+MK+ac-X"], protected_tokens=_PROTECTED, seed=7
    )
    assert result == ["+_K+__-X"]


def test_mask_fraction_counts_only_unprotected_tokens():
    (result,) = RandomMaskingStrategy(mask_fraction=0.5).mask(["+MKLA+acgt"], protected_tokens=_PROTECTED, seed=3)
    assert result.count("_") == 4
    assert result[0] == "+"
    assert result[5] == "+"


@pytest.mark.parametrize("protected_tokens", [None, frozenset()])
def test_no_protected_tokens_matches_unrestricted_masking(protected_tokens):
    strategy = RandomMaskingStrategy(mask_fraction=0.5)
    sequences = ["MKTLLIFLA", "ACDEFG"]
    assert strategy.mask(sequences, protected_tokens=protected_tokens, seed=11) == strategy.mask(sequences, seed=11)


def test_protected_tokens_reduce_available_count():
    with pytest.raises(ValueError, match="only 1 mutable"):
        RandomMaskingStrategy(num_mutations=2).mask(["+MX"], protected_tokens=_PROTECTED)
    assert RandomMaskingStrategy().mask(["+X"], protected_tokens=_PROTECTED) == ["+X"]


def test_protected_tokens_match_whole_multi_character_tokens():
    # Codon-level masking protects the stop codon while every other codon is masked.
    result = RandomMaskingStrategy(mask_fraction=1.0).mask(
        ["ATGAAATAA"], protected_tokens=frozenset({"TAA"}), token_size=3, seed=1
    )
    assert result == ["______TAA"]


def test_model_scored_masking_receives_sequences_and_skips_protected_sites():
    sequence = "+M+a-K"
    seen = []

    def logits(sequences):
        seen.extend(sequences)
        # Uniform rows on markers would win on entropy if protection were ignored.
        return [[[0.0, 0.0], [30.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [30.0, 0.0]]]

    result = MaskingStrategy(method="entropy", temperature=0.001, num_mutations=1).mask(
        [sequence], protected_tokens=_PROTECTED, position_score_fn=logits, seed=4
    )
    assert seen == [sequence]
    assert result == ["+M+_-K"]
