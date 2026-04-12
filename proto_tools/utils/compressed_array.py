"""Compressed array utilities for IPC deserialization.

Provides decompression of arrays compressed by
``standalone_helpers.compress_array()``. The compression side lives in
``standalone_helpers_source/standalone_helpers.py`` (copied into each tool's
isolated subprocess environment); the decompression side lives here in
the main ``proto_tools`` package (parent process).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_COMPRESSED_ARRAY_SENTINEL = "__compressed_array__"


def is_compressed_array(obj: Any) -> bool:
    """Check if *obj* is a compressed array dict.

    Args:
        obj (Any): Value to check.

    Returns:
        bool: ``True`` if *obj* is a dict with the compressed array sentinel.
    """
    return isinstance(obj, dict) and obj.get(_COMPRESSED_ARRAY_SENTINEL) is True


def decompress_array(compressed: dict[str, Any]) -> Any:
    """Decompress a compressed array dict back to a numpy ndarray.

    The input dict must have been produced by
    ``standalone_helpers.compress_array()``, containing keys
    ``__compressed_array__``, ``data``, ``dtype``, ``shape``, and ``version``.

    Args:
        compressed (dict[str, Any]): Compressed array dict with sentinel.

    Returns:
        Any: Reconstructed ``numpy.ndarray`` (contiguous, writable copy).

    Raises:
        ValueError: If the format version is unsupported.
    """
    import base64
    import zlib

    import numpy as np

    version = compressed.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported compressed array version: {version}")

    encoded = compressed["data"]
    dtype = compressed["dtype"]
    shape = tuple(compressed["shape"])

    raw_compressed = base64.b85decode(encoded.encode("ascii"))
    raw_bytes = zlib.decompress(raw_compressed)
    arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
    return arr.copy()


def decompress_result(obj: Any, *, to_list: bool = False) -> Any:
    """Recursively walk a result structure and decompress any compressed arrays.

    Leaves non-compressed values (primitives, plain dicts, plain lists)
    unchanged, so this function is safe to call on results that were
    produced before compression was enabled (backward compatible).

    Args:
        obj (Any): Any JSON-deserialized object (dict, list, or primitive).
        to_list (bool): If ``True``, convert decompressed arrays to nested
            Python lists (``list[list[float]]``). If ``False`` (default),
            return numpy arrays.

    Returns:
        Any: The same structure with compressed array dicts replaced by
            numpy arrays (or lists if *to_list* is ``True``).
    """
    if is_compressed_array(obj):
        arr = decompress_array(obj)
        return arr.tolist() if to_list else arr
    if isinstance(obj, dict):
        return {k: decompress_result(v, to_list=to_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decompress_result(item, to_list=to_list) for item in obj]
    return obj
