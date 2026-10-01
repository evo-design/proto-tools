"""Case-sensitive protein/DNA tokenization shared by mixed genomic language models."""

import math

from .proto_logging import get_logger

logger = get_logger(__name__)

PROTEIN_TOKENS = "ACDEFGHIKLMNPQRSTVWY"
DNA_TOKENS = "acgt"
AMBIGUOUS_PROTEIN_TOKENS = "XBUZO"
MIXED_VOCAB = list(PROTEIN_TOKENS + DNA_TOKENS)
STRAND_TOKENS = ("<+>", "<->")


def tokenize_mixed_sequence(sequence: str, *, allow_masks: bool = False) -> list[str]:
    """Split a prepared locus into atomic tokens without altering its biological content.

    Strand markers occupy one position. No automatic strand insertion, case conversion,
    translation, whitespace removal, or reverse complementation is performed.
    """
    tokens: list[str] = []
    index = 0
    alphabet = PROTEIN_TOKENS + DNA_TOKENS + AMBIGUOUS_PROTEIN_TOKENS
    while index < len(sequence):
        marker = sequence[index : index + 3]
        if marker in STRAND_TOKENS:
            tokens.append(marker)
            index += 3
            continue
        char = sequence[index]
        if char not in alphabet and not (allow_masks and char == "_"):
            raise ValueError(f"Invalid mixed-sequence character at position {index + 1}: {char!r}")
        tokens.append(char)
        index += 1
    if not any(token not in STRAND_TOKENS for token in tokens):
        raise ValueError("A mixed sequence must contain at least one biological position")
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
