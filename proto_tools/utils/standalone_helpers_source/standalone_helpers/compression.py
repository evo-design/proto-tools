"""Array compression helpers for large-array subprocess IPC.

Replaces ``numpy.ndarray.tolist()`` + ``json.dumps`` (billions of Python
float objects + gigabytes of JSON text) with a compact wire format:
``base85(zlib(array.tobytes()))``. The returned dict is JSON-safe and
survives the existing subprocess IPC pipeline unchanged. Decompression
lives on the parent side in ``proto_tools.utils.compressed_array``.
"""

from typing import Any

_COMPRESSED_ARRAY_SENTINEL = "__compressed_array__"
_COMPRESS_MIN_SIZE = 1000


def compress_array(arr: Any) -> dict[str, Any]:
    """Compress a numpy/jax array for JSON-safe IPC transport.

    Uses zlib compression on raw float32 bytes, then base85 encoding for
    JSON safety. The returned dict contains only JSON-primitive values
    (str, int, bool, list[int]) so it survives ``json.dump``/``json.load``
    roundtrips and Pydantic ``model_dump(mode="json")`` unchanged.

    For a ``(16384, 667)`` float32 array (typical AlphaGenome RNA_SEQ output):
    raw bytes = 43.7 MB, compressed ≈ 15-25 MB, base85 ≈ 20-31 MB — versus
    ~220 MB as JSON text via ``.tolist()``. Compression takes <0.5s versus
    ~30s for ``.tolist()`` + ``json.dumps``.

    Args:
        arr (Any): Array-like with ``.tobytes()``, ``.dtype``, ``.shape``
            attributes (numpy ndarray, jax Array, etc.).

    Returns:
        dict[str, Any]: JSON-serializable compressed representation with
            sentinel key ``__compressed_array__`` for detection.
    """
    import base64
    import zlib

    import numpy as np

    np_arr = np.asarray(arr, dtype=np.float32)
    if not np_arr.flags["C_CONTIGUOUS"]:
        np_arr = np.ascontiguousarray(np_arr)

    raw_bytes = np_arr.tobytes()
    compressed = zlib.compress(raw_bytes, level=1)
    encoded = base64.b85encode(compressed).decode("ascii")

    return {
        _COMPRESSED_ARRAY_SENTINEL: True,
        "data": encoded,
        "dtype": "float32",
        "shape": list(np_arr.shape),
        "version": 1,
    }


def is_compressed_array(obj: Any) -> bool:
    """Check if *obj* is a compressed array dict produced by :func:`compress_array`."""
    return isinstance(obj, dict) and obj.get(_COMPRESSED_ARRAY_SENTINEL) is True
