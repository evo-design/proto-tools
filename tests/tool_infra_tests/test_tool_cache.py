"""tests/tool_infra_tests/test_tool_cache.py

Tests for tool_cache."""

from typing import List, Union

import pytest
from pydantic import BaseModel

from proto_tools.utils import BaseConfig
from proto_tools.utils.tool_cache import (
    CacheStripResult,
    ToolCache,
    _get_obj_size,
    _program_tool_cache,
    _serialize_for_cache_key,
    cache_stitch_items,
    cache_store_items,
    cache_strip_items,
    deduplicate_items,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def _setup_cache():
    """Set up cache in contextvar before each test, clear after."""
    cache = ToolCache()
    _program_tool_cache.set(cache)
    yield cache
    _program_tool_cache.set(None)


@pytest.fixture
def empty_cache():
    return ToolCache()


# ── CacheStripResult tests ──────────────────────────────────────────────────

def test_cache_strip_result_all_cached():
    """all_cached property returns True when uncached_items is empty."""
    result = CacheStripResult(
        uncached_items=[],
        uncached_indices=[],
        cached_results={0: "a", 1: "b"},
        cache_keys=[],
    )
    assert result.all_cached is True


def test_cache_strip_result_not_all_cached():
    """all_cached property returns False when there are uncached items."""
    result = CacheStripResult(
        uncached_items=["x"],
        uncached_indices=[2],
        cached_results={0: "a", 1: "b"},
        cache_keys=["key_x"],
    )
    assert result.all_cached is False


def test_cache_strip_result_empty():
    """Empty CacheStripResult is all_cached (vacuously true)."""
    result = CacheStripResult()
    assert result.all_cached is True


# ── cache_strip_items tests ──────────────────────────────────────────────────

def test_cache_strip_items_no_cache():
    """Returns None when no active cache exists."""
    _program_tool_cache.set(None)
    result = cache_strip_items("test-tool", ["a", "b"], None)
    assert result is None


def test_cache_strip_items_all_miss(_setup_cache):
    """All items are uncached on first call."""
    items = ["item_a", "item_b", "item_c"]
    result = cache_strip_items("test-tool", items, None)

    assert result is not None
    assert len(result.uncached_items) == 3
    assert len(result.cached_results) == 0
    assert result.all_cached is False
    assert len(result.cache_keys) == 3


def test_cache_strip_items_all_hit(_setup_cache):
    """All items are cached after a store."""
    config = None
    items = ["item_a", "item_b"]

    # First call: all miss
    strip1 = cache_strip_items("test-tool", items, config)
    assert strip1.all_cached is False

    # Simulate storing results
    cache_store_items("test-tool", strip1.cache_keys, ["res_a", "res_b"])

    # Second call: all hit
    strip2 = cache_strip_items("test-tool", items, config)
    assert strip2 is not None
    assert strip2.all_cached is True
    assert strip2.cached_results == {0: "res_a", 1: "res_b"}


def test_cache_strip_items_partial_hit(_setup_cache):
    """Some items cached, some not."""
    config = None

    # Store one item
    strip1 = cache_strip_items("test-tool", ["a"], config)
    cache_store_items("test-tool", strip1.cache_keys, ["result_a"])

    # Query with mix of cached and new
    strip2 = cache_strip_items("test-tool", ["a", "b", "c"], config)
    assert strip2 is not None
    assert len(strip2.cached_results) == 1
    assert strip2.cached_results[0] == "result_a"
    assert strip2.uncached_items == ["b", "c"]
    assert strip2.uncached_indices == [1, 2]
    assert len(strip2.cache_keys) == 2


# ── cache_store_items tests ──────────────────────────────────────────────────

def test_cache_store_items_no_cache():
    """Does nothing when no active cache exists."""
    _program_tool_cache.set(None)
    # Should not raise
    cache_store_items("test-tool", ["key1"], ["val1"])


def test_cache_store_items_populates_cache(_setup_cache):
    """Stored items are retrievable from cache."""
    cache_store_items("test-tool", ["k1", "k2"], ["v1", "v2"])

    cache = _setup_cache
    assert cache.get("test-tool", "k1") == "v1"
    assert cache.get("test-tool", "k2") == "v2"

    info = cache.get_info()
    assert info["total_entries"] == 2


# ── cache_stitch_items tests ─────────────────────────────────────────────────

def test_cache_stitch_items_all_cached():
    """Stitching with no computed items returns cached items in order."""
    strip = CacheStripResult(
        uncached_items=[],
        uncached_indices=[],
        cached_results={0: "a", 1: "b", 2: "c"},
        cache_keys=[],
    )
    result = cache_stitch_items(strip, [], 3)
    assert result == ["a", "b", "c"]


def test_cache_stitch_items_all_computed():
    """Stitching with no cached items returns computed items in order."""
    strip = CacheStripResult(
        uncached_items=["x", "y", "z"],
        uncached_indices=[0, 1, 2],
        cached_results={},
        cache_keys=["k1", "k2", "k3"],
    )
    result = cache_stitch_items(strip, ["rx", "ry", "rz"], 3)
    assert result == ["rx", "ry", "rz"]


def test_cache_stitch_items_mixed():
    """Stitching interleaves cached and computed items correctly."""
    strip = CacheStripResult(
        uncached_items=["b", "d"],
        uncached_indices=[1, 3],
        cached_results={0: "cached_a", 2: "cached_c"},
        cache_keys=["kb", "kd"],
    )
    result = cache_stitch_items(strip, ["computed_b", "computed_d"], 4)
    assert result == ["cached_a", "computed_b", "cached_c", "computed_d"]


# ── Roundtrip: strip → store → stitch ────────────────────────────────────────

def test_strip_store_stitch_roundtrip(_setup_cache):
    """Full roundtrip: strip cached, store computed, stitch back."""
    config = None

    # Call 1: cache items A, B
    items1 = ["A", "B"]
    strip1 = cache_strip_items("rt-tool", items1, config)
    computed1 = ["result_A", "result_B"]
    cache_store_items("rt-tool", strip1.cache_keys, computed1)
    stitched1 = cache_stitch_items(strip1, computed1, len(items1))
    assert stitched1 == ["result_A", "result_B"]

    # Call 2: A cached, C new
    items2 = ["A", "C"]
    strip2 = cache_strip_items("rt-tool", items2, config)
    assert strip2.cached_results[0] == "result_A"
    assert strip2.uncached_items == ["C"]

    computed2 = ["result_C"]
    cache_store_items("rt-tool", strip2.cache_keys, computed2)
    stitched2 = cache_stitch_items(strip2, computed2, len(items2))
    assert stitched2 == ["result_A", "result_C"]


def test_strip_different_configs_separate_cache(_setup_cache):
    """Different configs produce different cache entries."""
    class Cfg(BaseConfig):
        multiplier: int = 2

    config_a = Cfg(multiplier=2)
    config_b = Cfg(multiplier=3)

    items = ["X"]

    # Store with config_a
    strip_a = cache_strip_items("cfg-tool", items, config_a)
    cache_store_items("cfg-tool", strip_a.cache_keys, ["result_2x"])

    # Same item with config_b should miss
    strip_b = cache_strip_items("cfg-tool", items, config_b)
    assert strip_b.all_cached is False
    assert strip_b.uncached_items == ["X"]


# ── Clear configuration logic tests ─────────────────────────────────────────

class _MockOptimizer:
    """Mock class to test the configuration logic."""
    def __init__(self, clear_config: Union[bool, List[str], int]):
        self.clear_tool_cache = clear_config
        self.tool_cache = ToolCache()

    def _clear_tool_cache(self) -> None:
        """The logic we want to test."""
        if not self.clear_tool_cache:
            return

        # Case 1: Byte threshold (int).
        if isinstance(self.clear_tool_cache, int) and not isinstance(self.clear_tool_cache, bool):
            threshold_bytes = self.clear_tool_cache

            if self.tool_cache.current_size > threshold_bytes:
                self.tool_cache.prune(threshold_bytes)

        # Case 2: Clear all (bool).
        elif isinstance(self.clear_tool_cache, bool):
            self.tool_cache.clear()

        # Case 3: Clear all for specific tools (List[str]).
        elif isinstance(self.clear_tool_cache, list):
            for tool in self.clear_tool_cache:
                self.tool_cache.clear(tool)

        else:
            raise ValueError(f"Invalid type of clear_tool_cache: {type(self.clear_tool_cache)}")


def test_config_bool_true_clears_all():
    opt = _MockOptimizer(clear_config=True)
    opt.tool_cache.set("t1", "k", "val")

    opt._clear_tool_cache()

    assert opt.tool_cache.current_size == 0
    assert opt.tool_cache.get("t1", "k") is None


def test_config_list_clears_specific():
    opt = _MockOptimizer(clear_config=["t1"])
    opt.tool_cache.set("t1", "k", "val")
    opt.tool_cache.set("t2", "k", "val")

    opt._clear_tool_cache()

    # t1 should be gone.
    assert opt.tool_cache.get("t1", "k") is None
    # t2 should remain.
    assert opt.tool_cache.get("t2", "k") == "val"


def test_config_int_threshold_below_limit():
    """Size is BELOW threshold, should NOT clear."""
    # Set threshold to 1000 bytes.
    opt = _MockOptimizer(clear_config=1000)

    # Add small data (e.g., 50 bytes).
    opt.tool_cache.set("t1", "k", "small_data")
    initial_size = opt.tool_cache.current_size

    # Ensure we are actually testing a valid scenario.
    assert initial_size < 1000

    opt._clear_tool_cache()

    # Should still be there.
    assert opt.tool_cache.current_size == initial_size
    assert opt.tool_cache.get("t1", "k") == "small_data"


def test_config_int_prunes_lru():
    """Test that we only remove enough to fit under the limit, removing the oldest items first."""
    old_data = "A" * 80
    old_data_size = _get_obj_size(old_data)

    new_data = "B" * 100
    new_data_size = _get_obj_size(new_data)

    opt = _MockOptimizer(clear_config=new_data_size)
    opt.tool_cache.set("tool", "old_key", old_data)
    opt.tool_cache.set("tool", "new_key", new_data)

    # Pre-check: Both exist.
    assert opt.tool_cache.get("tool", "old_key") is not None
    assert opt.tool_cache.get("tool", "new_key") is not None
    assert opt.tool_cache.current_size == old_data_size + new_data_size

    # 3. Trigger Prune.
    opt._clear_tool_cache()

    # 4. Verification.
    assert opt.tool_cache.current_size <= new_data_size

    # The old key should be gone (LRU eviction).
    assert opt.tool_cache.get("tool", "old_key") is None

    # The NEW key should still be there.
    assert opt.tool_cache.get("tool", "new_key") == new_data


# ── ToolCache internals tests ──────────────────────────────────────────────

def test_initial_state(empty_cache):
    assert empty_cache.current_size == 0
    assert empty_cache.get("tool", "key") is None


def test_set_increases_size(empty_cache):
    empty_cache.set("calculator", "2+2", 4)
    size_after_add = empty_cache.current_size
    assert size_after_add > 0
    assert empty_cache.get("calculator", "2+2") == 4


def test_overwrite_updates_size_correctly(empty_cache):
    # 1. Add small item.
    empty_cache.set("tool", "key", "a")
    size_small = empty_cache.current_size

    # 2. Overwrite with massive item.
    large_string = "a" * 1000
    empty_cache.set("tool", "key", large_string)
    size_large = empty_cache.current_size

    assert size_large > size_small

    # 3. Overwrite with small item again.
    empty_cache.set("tool", "key", "b")
    size_final = empty_cache.current_size

    # Size should have gone back down.
    assert size_final < size_large


def test_clear_all_resets_size(empty_cache):
    empty_cache.set("tool1", "k1", "data")
    empty_cache.set("tool2", "k2", "data")

    assert empty_cache.current_size > 0

    empty_cache.clear()

    assert empty_cache.current_size == 0
    assert empty_cache._cache == {}


def test_clear_specific_tool_adjusts_size(empty_cache):
    empty_cache.set("tool1", "k1", "data_1")

    empty_cache.set("tool2", "k2", "data_2")
    size_2 = empty_cache.current_size

    # Clear tool 2.
    empty_cache.clear("tool2")

    assert empty_cache.current_size < size_2
    assert empty_cache.get("tool2", "k2") is None
    assert empty_cache.get("tool1", "k1") == "data_1"


def test_circular_reference_safety():
    """Ensure recursive size calc doesn't crash on circular refs."""
    cache = ToolCache()

    d = {}
    d['self'] = d  # Circular reference.

    try:
        cache.set("tool", "circ", d)
    except RecursionError:
        pytest.fail("Recursive size calculation crashed on circular reference")

    assert cache.current_size > 0


# ── deduplicate_items unit tests ─────────────────────────────────────────────

def test_deduplicate_items_all_unique():
    """All unique items should be returned unchanged."""
    items = ["a", "b", "c"]
    result = deduplicate_items(items, key_fn=str)

    assert result.unique_items == ["a", "b", "c"]
    assert len(result.unique_keys) == 3
    assert result.index_map == [(0, 0), (1, 1), (2, 2)]


def test_deduplicate_items_all_duplicates():
    """All-duplicate input should return a single unique item."""
    items = ["x", "x", "x"]
    result = deduplicate_items(items, key_fn=str)

    assert result.unique_items == ["x"]
    assert len(result.unique_keys) == 1
    assert result.index_map == [(0, 0), (1, 0), (2, 0)]


def test_deduplicate_items_mixed():
    """Mixed input should preserve first-occurrence order."""
    items = ["a", "b", "a", "c", "b"]
    result = deduplicate_items(items, key_fn=str)

    assert result.unique_items == ["a", "b", "c"]
    assert result.index_map == [(0, 0), (1, 1), (2, 0), (3, 2), (4, 1)]


def test_deduplicate_items_empty():
    """Empty input should return empty results."""
    result = deduplicate_items([], key_fn=str)

    assert result.unique_items == []
    assert result.unique_keys == []
    assert result.index_map == []


def test_deduplicate_items_preserves_order():
    """First occurrence order is preserved in unique_items."""
    items = ["c", "a", "c", "b", "a"]
    result = deduplicate_items(items, key_fn=str)

    assert result.unique_items == ["c", "a", "b"]
    # Verify index_map lets us reconstruct original list
    reconstructed = [result.unique_items[uid] for _, uid in result.index_map]
    assert reconstructed == items


def test_deduplicate_items_with_pydantic_models():
    """Test dedup with Pydantic models using _serialize_for_cache_key."""

    class Item(BaseModel):
        value: int

    items = [Item(value=1), Item(value=2), Item(value=1), Item(value=3)]
    result = deduplicate_items(items, key_fn=_serialize_for_cache_key)

    assert len(result.unique_items) == 3
    assert [it.value for it in result.unique_items] == [1, 2, 3]
    assert result.index_map == [(0, 0), (1, 1), (2, 0), (3, 2)]
