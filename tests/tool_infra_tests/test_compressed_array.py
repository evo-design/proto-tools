"""Tests for compressed array serialization roundtrip."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

# ============================================================================
# Roundtrip correctness
# ============================================================================


def test_roundtrip_bit_exact():
    """Compress -> JSON serialize -> JSON deserialize -> decompress = original (bit-exact)."""
    from proto_tools.utils.compressed_array import decompress_array
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    rng = np.random.default_rng(42)
    original = rng.standard_normal((1000, 100)).astype(np.float32)

    compressed = compress_array(original)
    json_str = json.dumps(compressed)
    reloaded = json.loads(json_str)
    recovered = decompress_array(reloaded)

    np.testing.assert_array_equal(original, recovered)


def test_roundtrip_large_array():
    """Roundtrip on a realistic AlphaGenome-sized array (16k x 667)."""
    from proto_tools.utils.compressed_array import decompress_array
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    rng = np.random.default_rng(42)
    original = rng.standard_normal((16384, 667)).astype(np.float32)

    compressed = compress_array(original)
    recovered = decompress_array(compressed)

    np.testing.assert_array_equal(original, recovered)


def test_roundtrip_1d_array():
    """1D arrays roundtrip correctly."""
    from proto_tools.utils.compressed_array import decompress_array
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    original = np.arange(5000, dtype=np.float32)
    recovered = decompress_array(compress_array(original))
    np.testing.assert_array_equal(original, recovered)


def test_roundtrip_float64_input():
    """Float64 inputs are normalized to float32; values are preserved within tolerance."""
    from proto_tools.utils.compressed_array import decompress_array
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    original = np.array([[1.0, 2.5], [3.0, 4.5]], dtype=np.float64)
    recovered = decompress_array(compress_array(original))

    assert recovered.dtype == np.float32
    np.testing.assert_allclose(recovered, original, rtol=1e-7)


# ============================================================================
# JSON safety
# ============================================================================


def test_compressed_dict_is_json_serializable():
    """The compressed dict survives json.dumps / json.loads."""
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    arr = np.ones((100, 50), dtype=np.float32)
    compressed = compress_array(arr)

    json_str = json.dumps(compressed)
    reloaded = json.loads(json_str)

    assert reloaded["__compressed_array__"] is True
    assert reloaded["shape"] == [100, 50]
    assert reloaded["dtype"] == "float32"
    assert reloaded["version"] == 1
    assert isinstance(reloaded["data"], str)


# ============================================================================
# Size and speed
# ============================================================================


def test_compression_reduces_json_size():
    """Compressed JSON is at least 3x smaller than list-of-lists JSON."""
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    rng = np.random.default_rng(42)
    arr = rng.standard_normal((1000, 100)).astype(np.float32)

    list_json = json.dumps(arr.tolist())
    compressed_json = json.dumps(compress_array(arr))

    assert len(compressed_json) < len(list_json) / 3


@pytest.mark.slow
def test_compressed_json_smaller_and_faster_than_list_json():
    """Full pipeline: compress+json.dumps is faster and smaller than tolist+json.dumps."""
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    rng = np.random.default_rng(42)
    arr = rng.standard_normal((4000, 200)).astype(np.float32)

    t0 = time.monotonic()
    list_json = json.dumps(arr.tolist())
    list_time = time.monotonic() - t0

    t0 = time.monotonic()
    compressed_json = json.dumps(compress_array(arr))
    compress_time = time.monotonic() - t0

    assert len(compressed_json) < len(list_json) / 3, (
        f"Compressed JSON ({len(compressed_json):,}) should be <1/3 of list JSON ({len(list_json):,})"
    )
    assert compress_time < list_time, (
        f"compress+json.dumps ({compress_time:.2f}s) should be faster than tolist+json.dumps ({list_time:.2f}s)"
    )


# ============================================================================
# Sentinel detection
# ============================================================================


def test_is_compressed_array_positive():
    """Correctly identifies compressed array dicts."""
    from proto_tools.utils.compressed_array import is_compressed_array

    assert is_compressed_array(
        {
            "__compressed_array__": True,
            "data": "",
            "shape": [1],
            "dtype": "float32",
            "version": 1,
        }
    )


def test_is_compressed_array_negative():
    """Does not falsely identify non-compressed objects."""
    from proto_tools.utils.compressed_array import is_compressed_array

    assert not is_compressed_array({"values": [[1, 2]]})
    assert not is_compressed_array([1, 2, 3])
    assert not is_compressed_array("hello")
    assert not is_compressed_array(42)
    assert not is_compressed_array(None)


# ============================================================================
# Recursive decompression
# ============================================================================


def test_decompress_result_recursive():
    """decompress_result walks nested dicts and replaces compressed arrays."""
    from proto_tools.utils.compressed_array import decompress_result
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    nested = {
        "predictions": {
            "reference": {
                "rna_seq": {
                    "values": compress_array(arr),
                    "metadata": [{"name": "track1"}, {"name": "track2"}],
                    "resolution": 1,
                }
            }
        }
    }

    result = decompress_result(nested, to_list=False)

    np.testing.assert_array_equal(result["predictions"]["reference"]["rna_seq"]["values"], arr)
    assert result["predictions"]["reference"]["rna_seq"]["metadata"] == [
        {"name": "track1"},
        {"name": "track2"},
    ]
    assert result["predictions"]["reference"]["rna_seq"]["resolution"] == 1


def test_decompress_result_to_list():
    """to_list=True produces nested Python lists."""
    from proto_tools.utils.compressed_array import decompress_result
    from proto_tools.utils.standalone_helpers_source.standalone_helpers import compress_array

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    compressed = {"values": compress_array(arr)}

    result = decompress_result(compressed, to_list=True)

    assert isinstance(result["values"], list)
    assert isinstance(result["values"][0], list)
    assert result["values"] == [[1.0, 2.0], [3.0, 4.0]]


def test_decompress_result_backward_compat():
    """Uncompressed data passes through decompress_result unchanged."""
    from proto_tools.utils.compressed_array import decompress_result

    original = {
        "values": [[1.0, 2.0], [3.0, 4.0]],
        "metadata": [{"name": "track1"}],
        "scalar": 42,
        "text": "hello",
    }

    result = decompress_result(original, to_list=False)
    assert result == original


# ============================================================================
# Error handling
# ============================================================================


def test_unsupported_version_raises():
    """Unsupported version number raises ValueError."""
    from proto_tools.utils.compressed_array import decompress_array

    with pytest.raises(ValueError, match="Unsupported compressed array version"):
        decompress_array(
            {
                "__compressed_array__": True,
                "data": "",
                "shape": [1],
                "dtype": "float32",
                "version": 99,
            }
        )
