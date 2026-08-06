"""tests/tool_infra_tests/test_tool_cache.py.

Tests for tool_cache.
"""

import pytest
from pydantic import BaseModel, Field

from proto_tools.utils import BaseConfig, ConfigField
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
from proto_tools.utils.tool_io import BaseToolInput, BaseToolOutput, InputField

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    from proto_tools.tools.tool_registry import ToolRegistry

    original_registry = ToolRegistry._registry.copy()
    ToolRegistry._registry.clear()
    yield ToolRegistry
    ToolRegistry._registry = original_registry


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


def test_cache_store_items_rejects_length_mismatch(_setup_cache):
    """Mismatched keys/results lengths violate the documented 1:1 contract and must fail fast."""
    with pytest.raises(ValueError):
        cache_store_items("test-tool", ["k1", "k2"], ["only_one"])


# ── cache_stitch_items tests ─────────────────────────────────────────────────


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

    def __init__(self, clear_config: bool | list[str] | int):
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
    d["self"] = d  # Circular reference.

    try:
        cache.set("tool", "circ", d)
    except RecursionError:
        pytest.fail("Recursive size calculation crashed on circular reference")

    assert cache.current_size > 0


# ── deduplicate_items unit tests ─────────────────────────────────────────────


def test_deduplicate_items_all_duplicates():
    """All-duplicate input should return a single unique item."""
    items = ["x", "x", "x"]
    result = deduplicate_items(items, key_fn=str)

    assert result.unique_items == ["x"]
    assert len(result.unique_keys) == 1
    assert result.index_map == [(0, 0), (1, 0), (2, 0)]


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


# ── Nested-config inheritance and cache keys ────────────────────────────────


class _NestedCfg(BaseConfig):
    """Config nested inside another config."""

    dataset: str = ConfigField(default="a", title="Dataset", description="Dataset")


class _OuterCfg(BaseConfig):
    """Config holding a nested config directly and in a list."""

    nested: _NestedCfg = ConfigField(default_factory=_NestedCfg, title="Nested", description="Nested")
    many: list[_NestedCfg] = ConfigField(default_factory=list, title="Many", description="Many")


@pytest.mark.parametrize(
    "nested,expected",
    [(None, 2), (_NestedCfg(verbose=0), 0)],
    ids=["unset_inherits", "explicit_kept"],
)
def test_nested_config_inherits_verbose_unless_it_sets_its_own(nested, expected):
    """A nested config takes the outer verbosity only when it did not name its own."""
    config = _OuterCfg(verbose=2) if nested is None else _OuterCfg(verbose=2, nested=nested)

    assert config.nested.verbose == expected


def test_nested_config_in_a_collection_inherits_verbose():
    """Inheritance reaches configs held in list fields, not only direct attributes."""
    config = _OuterCfg(verbose=3, many=[_NestedCfg()])

    assert config.many[0].verbose == 3


def test_reassigning_the_outer_value_reaches_nested_configs():
    """Changing the outer verbosity flows down again, because inheritance never marks a field set."""
    config = _OuterCfg(verbose=1)

    config.verbose = 3

    assert config.nested.verbose == 3


@pytest.mark.parametrize(
    "field,value", [("verbose", 3), ("timeout", 99), ("device", "cuda")], ids=["verbose", "timeout", "device"]
)
def test_nested_non_key_fields_do_not_split_the_cache(field, value):
    """A nested setting that cannot change the result is excluded from the cache key."""
    baseline = _OuterCfg()
    varied = _OuterCfg(nested=_NestedCfg(**{field: value}))

    assert baseline.cache_key() == varied.cache_key()


def test_nested_result_affecting_fields_still_split_the_cache():
    """A nested setting that does change the result still reaches the cache key."""
    baseline = _OuterCfg()
    varied = _OuterCfg(nested=_NestedCfg(dataset="b"))

    assert baseline.cache_key() != varied.cache_key()


# ── Output fields beyond the iterable one, through the cache ────────────────


class _ItemsIn(BaseToolInput):
    """Iterable input."""

    items: list[str] = InputField(title="Items", description="Items")


class _RichItem(BaseModel):
    """One result and everything produced for it."""

    value: str = Field(description="Per-item value")
    score: int = Field(description="Per-item score")


class _RichOut(BaseToolOutput):
    """Per-item results plus an invariant required field."""

    results: list[_RichItem] = Field(default_factory=list, description="One entry per item")
    run_label: str = Field(description="Invariant across items, required")

    @property
    def output_format_options(self) -> list[str]:
        """Supported export formats."""
        return ["json"]

    @property
    def output_format_default(self) -> str:
        """Default export format."""
        return "json"

    def _export_output(self, export_path, file_format) -> None:
        """No-op export."""
        return


def _register_rich_tool(registry, calls: list[int]):
    """Register a cacheable iterable tool carrying both kinds of extra output field."""

    @registry.register(
        key="rich-cache-tool",
        label="Rich",
        category="test",
        input_class=_ItemsIn,
        config_class=BaseConfig,
        output_class=_RichOut,
        description="Aligned + invariant output fields",
        iterable_input_fields=["items"],
        iterable_output_field="results",
        max_chunk_size=None,
        cacheable=True,
    )
    def run_tool(inputs, config=None, instance=None):
        calls.append(len(inputs.items))
        return _RichOut(
            results=[_RichItem(value=f"out_{i}", score=len(i)) for i in inputs.items],
            run_label="fixed",
        )

    return registry.get("rich-cache-tool").function


def test_full_cache_hit_restores_invariant_and_aligned_fields(clean_registry, _setup_cache):
    """A repeat call served from cache returns the same object shape as the computed one."""
    calls: list[int] = []
    fn = _register_rich_tool(clean_registry, calls)

    first = fn(_ItemsIn(items=["a", "bb"]))
    second = fn(_ItemsIn(items=["a", "bb"]))

    assert calls == [2]  # second call was served from cache
    assert second.results == first.results
    assert [r.score for r in second.results] == [1, 2]
    assert second.run_label == "fixed"


def test_partial_cache_hit_stitches_per_item_values(clean_registry, _setup_cache):
    """Cached and freshly computed items are stitched back in input order."""
    calls: list[int] = []
    fn = _register_rich_tool(clean_registry, calls)

    fn(_ItemsIn(items=["a", "bb"]))
    result = fn(_ItemsIn(items=["a", "bb", "ccc"]))

    assert calls == [2, 1]  # only the new item was computed
    assert [r.value for r in result.results] == ["out_a", "out_bb", "out_ccc"]
    assert [r.score for r in result.results] == [1, 2, 3]
    assert result.run_label == "fixed"


def test_dedup_expansion_restores_every_original_position(clean_registry, _setup_cache):
    """A duplicated input gets its full item back in both positions."""
    calls: list[int] = []
    fn = _register_rich_tool(clean_registry, calls)

    result = fn(_ItemsIn(items=["a", "bb", "a"]))

    assert calls == [2]  # 'a' deduped away
    assert [r.value for r in result.results] == ["out_a", "out_bb", "out_a"]
    assert [r.score for r in result.results] == [1, 2, 1]
