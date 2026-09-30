"""Case-sensitive protein/DNA tokenization shared by mixed genomic language models."""

import math
from itertools import pairwise

from .proto_logging import get_logger

logger = get_logger(__name__)

PROTEIN_TOKENS = "ACDEFGHIKLMNPQRSTVWY"
DNA_TOKENS = "acgt"
AMBIGUOUS_PROTEIN_TOKENS = "XBUZO"
MIXED_VOCAB = list(PROTEIN_TOKENS + DNA_TOKENS)
STRAND_TOKENS = "+-"
# Upstream tokenizers spell the one-character strand markers as special tokens.
UPSTREAM_STRAND_TOKENS = {"+": "<+>", "-": "<->"}


def normalize_mixed_sequence(sequence: str) -> str:
    """Rewrite upstream ``<+>``/``<->`` strand markers as one-character ``+``/``-``."""
    for short, upstream in UPSTREAM_STRAND_TOKENS.items():
        sequence = sequence.replace(upstream, short)
    return sequence


def tokenize_mixed_sequence(sequence: str, *, allow_masks: bool = False) -> list[str]:
    """Split a prepared locus into one-character tokens without altering its biological content.

    Strand markers may be written ``+``/``-`` or ``<+>``/``<->``; both become one ``+``/``-``
    token. Each marker must be followed by a biological token, and ``-`` by protein. No
    automatic strand insertion, case conversion, translation, whitespace removal, or reverse
    complementation is performed.
    """
    tokens = list(normalize_mixed_sequence(sequence))
    alphabet = PROTEIN_TOKENS + DNA_TOKENS + AMBIGUOUS_PROTEIN_TOKENS + STRAND_TOKENS
    for index, char in enumerate(tokens):
        if char not in alphabet and not (allow_masks and char == "_"):
            raise ValueError(f"Invalid mixed-sequence character at token position {index + 1}: {char!r}")
    if not any(token not in STRAND_TOKENS for token in tokens):
        raise ValueError("A mixed sequence must contain at least one biological position")
    for index, (current, following) in enumerate(pairwise(tokens)):
        # Each marker starts a non-empty element; intergenic DNA is always on the + strand.
        if current in STRAND_TOKENS and following in STRAND_TOKENS:
            raise ValueError(f"Consecutive strand markers at token positions {index + 1}-{index + 2}")
        if current == "-" and following in DNA_TOKENS:
            raise ValueError(f"DNA follows a '-' marker at token position {index + 2}; intergenic DNA uses '+'")
    return tokens


def one_hot_mixed_logits(sequence: str, *, sharpness: float = 20.0) -> list[list[float]]:
    """Build token-aligned 24-column logits; fixed context rows contain zeros."""
    if not math.isfinite(sharpness):
        raise ValueError("sharpness must be finite")
    rows = []
    for token in tokenize_mixed_sequence(sequence):
        row = [0.0] * len(MIXED_VOCAB)
        if token in MIXED_VOCAB:
            row[MIXED_VOCAB.index(token)] = sharpness
        rows.append(row)
    return rows
