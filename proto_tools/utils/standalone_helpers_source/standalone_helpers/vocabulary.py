"""proto_tools/utils/standalone_helpers_source/standalone_helpers/vocabulary.py.

Index mapping between proto's column order and a model's own vocabulary order.

Re-exported from the package ``__init__``; standalones may also import the submodule
directly as ``from standalone_helpers.vocabulary import gather_indices``.
"""

from collections.abc import Mapping, Sequence

from .proto_logging import get_logger

logger = get_logger(__name__)

# ── Proto's interface vocabulary orders ─────────────────────────────────────
# These are the column orders every tool must accept inputs in and return gradients and
# embeddings in. The parent's matching source is ``proto_tools/utils/sequence.py``, and a
# consistency test pins the two together.
#
# A model's own native ordering does NOT belong here: it is that model's ground truth and lives
# beside its other files in the toolkit's ``standalone/`` directory, mapped onto the orders
# below with ``gather_indices``. Those orderings are fixed by separate upstreams and only
# coincide by accident, so sharing one constant between models would couple them wrongly.

AMINO_ACIDS_LIST: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
"""Canonical 20 standard amino acids in alphabetical order; proto's protein column order."""

DNA_NUCLEOTIDES: str = "ACGT"
"""Standard DNA nucleotides; proto's nucleotide column order."""

RNA_NUCLEOTIDES: str = "ACGU"
"""Standard RNA nucleotides."""

PROTEIN_TOKENS: str = "".join(AMINO_ACIDS_LIST)
"""Uppercase protein tokens in the mixed protein/DNA vocabulary."""

DNA_TOKENS: str = DNA_NUCLEOTIDES.lower()
"""Lowercase DNA tokens in the mixed protein/DNA vocabulary."""

AMBIGUOUS_PROTEIN_TOKENS: str = "XBUZO"
"""Protein symbols accepted in input but excluded from the relaxed vocabulary."""

MIXED_VOCAB: list[str] = list(PROTEIN_TOKENS + DNA_TOKENS)
"""Mixed protein/DNA column order: protein in 0-19, lowercase DNA in 20-23."""


def gather_indices(
    target_vocab: Sequence[str],
    source_vocab: Sequence[str] | Mapping[str, int],
    *,
    offset: int = 0,
    label: str = "vocabulary",
) -> list[int]:
    """Return indices into ``source_vocab`` that reorder a source-ordered axis into target order.

    ``result[i]`` is the position of ``target_vocab[i]`` within ``source_vocab``, plus ``offset``.
    A ``Mapping`` source is looked up directly, which is what a tokenizer vocabulary is; a
    ``Sequence`` source is indexed by position.

    The argument order carries the direction, so one call covers every use:

    * permute a proto-ordered matrix into native column order —
      ``matrix[:, gather_indices(native, proto)]``
    * bring a native-ordered gradient back to proto columns —
      ``gradient[:, gather_indices(proto, native)]``
    * gather embedding rows in proto order —
      ``weights[gather_indices(proto, native)]``

    The first two are mutual inverses when the vocabularies are permutations of each other. When
    the source is a strict superset (a full tokenizer, or an alphabet carrying extra symbols),
    only the direction that targets the smaller vocabulary is total.

    Args:
        target_vocab (Sequence[str]): Symbols in the order the result should produce.
        source_vocab (Sequence[str] | Mapping[str, int]): Symbols in the axis's current order, or
            a mapping from symbol to index such as a tokenizer vocabulary.
        offset (int): Added to every returned index, for models that reserve leading indices.
        label (str): Name used in error messages, typically the toolkit or model.

    Returns:
        list[int]: One index per entry of ``target_vocab``, in target order.

    Raises:
        ValueError: If ``target_vocab`` is empty, a target symbol is absent from ``source_vocab``,
            or a ``Sequence`` source repeats a symbol.
    """
    if not target_vocab:
        raise ValueError(f"{label}: target_vocab must contain at least one symbol")

    if isinstance(source_vocab, Mapping):
        lookup: Mapping[str, int] = source_vocab
    else:
        wanted = set(target_vocab)
        positions: dict[str, int] = {}
        for index, symbol in enumerate(source_vocab):
            if symbol not in wanted:
                continue
            if symbol in positions:
                raise ValueError(
                    f"{label}: source_vocab repeats {symbol!r} at positions {positions[symbol]} and {index}; "
                    "an ambiguous source order cannot define a mapping"
                )
            positions[symbol] = index
        lookup = positions

    missing = [symbol for symbol in target_vocab if symbol not in lookup]
    if missing:
        raise ValueError(f"{label}: source_vocab is missing {missing!r}; it must cover every symbol in target_vocab")
    return [lookup[symbol] + offset for symbol in target_vocab]
