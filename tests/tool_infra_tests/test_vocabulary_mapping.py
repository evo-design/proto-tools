"""tests/tool_infra_tests/test_vocabulary_mapping.py.

Tests for ``standalone_helpers.vocabulary.gather_indices``, the shared mapping between proto's
column order and a model's own vocabulary order.

Coverage:

- Both directions of a permutation pair are mutual inverses.
- A superset source (an alphabet with extra symbols, or a full tokenizer) maps one-directionally.
- A ``Mapping`` source, as a tokenizer vocabulary is, is looked up directly.
- ``offset`` shifts every index, for models reserving leading positions.
- Missing and repeated symbols raise rather than returning a silently wrong permutation.

CPU-only and environment-free.
"""

from __future__ import annotations

import pytest
from standalone_helpers.vocabulary import AMINO_ACIDS_LIST, gather_indices

# Stands in for any model's native ordering; this exercises the mapping, not a real vocabulary.
NATIVE_ORDER = list("ARNDCQEGHILKMFPSTWYV")


def _apply(axis: list[str], indices: list[int]) -> list[str]:
    """Reorder ``axis`` by ``indices``, the way a tensor gather would."""
    return [axis[index] for index in indices]


# ── Direction and round trip ─────────────────────────────────────────────────


def test_gather_indices_reorders_into_target_order() -> None:
    """Gathering a source-ordered axis by the result yields target order."""
    indices = gather_indices(NATIVE_ORDER, AMINO_ACIDS_LIST)
    assert _apply(AMINO_ACIDS_LIST, indices) == NATIVE_ORDER


def test_permutation_directions_are_mutual_inverses() -> None:
    """Proto -> native -> proto is the identity for a permutation pair.

    This is the invariant that makes an inverted permutation impossible to express: the two
    directions differ only in argument order, so there is no separate hand-written inverse to
    get backwards.
    """
    to_native = gather_indices(NATIVE_ORDER, AMINO_ACIDS_LIST)
    from_native = gather_indices(AMINO_ACIDS_LIST, NATIVE_ORDER)

    native_ordered = _apply(AMINO_ACIDS_LIST, to_native)
    assert _apply(native_ordered, from_native) == AMINO_ACIDS_LIST


# ── Superset sources ─────────────────────────────────────────────────────────


def test_superset_source_maps_one_directionally() -> None:
    """A source carrying extra symbols maps the smaller target, but not the reverse.

    ProteinMPNN's alphabet is the AlphaFold order plus ``X``; tokenizer vocabularies are much
    larger supersets. Only the direction targeting the smaller vocabulary is total.
    """
    alphabet_with_x = [*NATIVE_ORDER, "X"]

    indices = gather_indices(AMINO_ACIDS_LIST, alphabet_with_x)
    assert _apply(alphabet_with_x, indices) == AMINO_ACIDS_LIST

    with pytest.raises(ValueError, match=r"missing \['X'\]"):
        gather_indices(alphabet_with_x, AMINO_ACIDS_LIST)


# ── Mapping sources (tokenizers) ─────────────────────────────────────────────


def test_mapping_source_is_looked_up_directly() -> None:
    """A tokenizer vocabulary maps symbols to ids that need not be contiguous or ordered."""
    tokenizer_vocab = {"<cls>": 0, "<pad>": 1, **{aa: 10 + index for index, aa in enumerate(AMINO_ACIDS_LIST)}}

    indices = gather_indices(AMINO_ACIDS_LIST, tokenizer_vocab, label="esm2")

    assert indices == [10 + index for index in range(20)]


def test_mapping_source_reports_absent_symbols() -> None:
    """A tokenizer that does not cover the proto vocabulary fails loudly."""
    incomplete = {aa: index for index, aa in enumerate(AMINO_ACIDS_LIST) if aa != "C"}

    with pytest.raises(ValueError, match=r"esm2: source_vocab is missing \['C'\]"):
        gather_indices(AMINO_ACIDS_LIST, incomplete, label="esm2")


# ── Offset and error cases ───────────────────────────────────────────────────


def test_offset_shifts_every_index() -> None:
    """``offset`` covers models that reserve leading indices, such as OpenFold's index 0."""
    plain = gather_indices(AMINO_ACIDS_LIST, NATIVE_ORDER)
    shifted = gather_indices(AMINO_ACIDS_LIST, NATIVE_ORDER, offset=1)

    assert shifted == [index + 1 for index in plain]


def test_repeated_source_symbol_is_rejected() -> None:
    """An ambiguous source order cannot define a mapping."""
    with pytest.raises(ValueError, match="repeats 'A'"):
        gather_indices(["A"], ["A", "C", "A"])


def test_empty_target_is_rejected() -> None:
    """An empty target vocabulary is a caller mistake, not an empty mapping."""
    with pytest.raises(ValueError, match="at least one symbol"):
        gather_indices([], AMINO_ACIDS_LIST)
