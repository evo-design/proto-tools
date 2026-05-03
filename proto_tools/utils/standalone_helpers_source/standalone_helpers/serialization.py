"""Shared bio constants and serialization helpers for standalone inference scripts."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

AMINO_ACIDS_LIST: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
"""Canonical 20 standard amino acids in alphabetical order."""

DNA_NUCLEOTIDES: str = "ACGT"
"""Standard DNA nucleotides."""

RNA_NUCLEOTIDES: str = "ACGU"
"""Standard RNA nucleotides."""


def serialize_output(value: Any) -> Any:
    """Recursively serialize tensors and arrays to JSON-safe Python types.

    Handles PyTorch tensors, JAX arrays, numpy arrays, and scalar-like objects.
    Checks ``.tolist()`` before ``.cpu()`` to avoid an unnecessary intermediate
    CPU copy when serializing GPU tensors.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: serialize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_output(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return value.item()
    if not isinstance(value, (str, int, float, bool)):
        logger.warning(
            "serialize_output: passing through unrecognized type %s; downstream JSON encoding may fail",
            type(value).__name__,
        )
    return value
