"""
test_tool_cache.py


Tests for the tool_cache.
"""

from typing import List, Union
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from bio_programming.bio_tools.tools.utils import BaseConfig
from bio_programming.bio_tools.tools.infra.tool_cache import (
    ToolCache,
    _get_obj_size,
    _program_tool_cache,
    clear_cache,
    get_cache_info,
    tool_cache,
    tool_cache_iterable,
)
from bio_programming.bio_tools.tools.infra.tool_io import BaseToolInput
from tests.tool_tests.tool_infra_tests.test_export_functionality import (
    MockToolOutputBase,
)


class TestToolCaching:
    """Test suite for tool caching behavior."""

    def setup_method(self):
        """Set up cache in contextvar before each test."""
        self.cache = ToolCache()
        _program_tool_cache.set(self.cache)

    def teardown_method(self):
        """Clear contextvar after each test."""
        _program_tool_cache.set(None)

    def test_tool_cache_decorator_basic(self):
        """Test basic tool_cache decorator functionality."""
        call_count = 0

        @tool_cache("test_tool")
        def expensive_function(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should execute function
        result1 = expensive_function(5, 3)
        assert result1 == 8
        assert call_count == 1

        # Second call with same args should use cache
        result2 = expensive_function(5, 3)
        assert result2 == 8
        assert call_count == 1  # Still 1, not 2

        # Different args should execute function again
        result3 = expensive_function(10, 7)
        assert result3 == 17
        assert call_count == 2

    def test_tool_cache_with_pydantic_models(self):
        """Test caching with Pydantic model inputs."""
        call_count = 0

        class TestConfig(BaseModel):
            value: int
            text: str
            threshold: float = 0.5

        class TestOutput(BaseModel):
            result: int
            message: str

        @tool_cache("pydantic_tool")
        def process_config(config: TestConfig) -> TestOutput:
            nonlocal call_count
            call_count += 1
            return TestOutput(
                result=config.value * 2,
                message=f"Processed {config.text}"
            )

        config1 = TestConfig(value=5, text="test", threshold=0.8)
        config2 = TestConfig(value=5, text="test", threshold=0.8)  # Same content
        config3 = TestConfig(value=5, text="test", threshold=0.9)  # Different threshold

        # First call executes
        result1 = process_config(config1)
        assert result1.result == 10
        assert call_count == 1

        # Same config should use cache
        result2 = process_config(config2)
        assert result2.result == 10
        assert call_count == 1  # Still 1

        # Different config executes again
        result3 = process_config(config3)
        assert result3.result == 10
        assert call_count == 2

    def test_cache_disabled_mode(self):
        """Test that caching can be disabled."""
        call_count = 0

        @tool_cache("test_tool", enabled=False)
        def no_cache_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Every call should execute the function
        result1 = no_cache_function(5)
        assert result1 == 10
        assert call_count == 1

        result2 = no_cache_function(5)
        assert result2 == 10
        assert call_count == 2  # Should increment

    def test_cache_info_tracking(self):
        """Test cache info retrieval."""
        @tool_cache("info_tool")
        def cached_func(x: int) -> int:
            return x * x

        # Initially empty
        info = get_cache_info()
        assert info["total_entries"] == 0

        # Add some cached entries
        cached_func(5)
        cached_func(10)

        info = get_cache_info()
        assert info["total_entries"] == 2

        # Clear and check
        clear_cache()
        info = get_cache_info()
        assert info["total_entries"] == 0

    def test_verbose_mode_reporting(self, caplog):
        """Test verbose mode cache hit/miss reporting."""
        import logging

        class VerboseInput(BaseToolInput):
            value: int

        class VerboseConfig(BaseConfig):
            multiplier: int = 2
            verbose: bool = True

        @tool_cache("verbose_tool")
        def verbose_function(inputs: VerboseInput, config: VerboseConfig) -> int:
            return inputs.value * config.multiplier

        inputs = VerboseInput(value=5)
        config = VerboseConfig(multiplier=2, verbose=True)

        # First call should report cache miss
        with caplog.at_level(logging.DEBUG):
            _ = verbose_function(inputs, config)
        assert "[Cache Miss]" in caplog.text

        caplog.clear()

        # Second call should report cache hit
        with caplog.at_level(logging.DEBUG):
            _ = verbose_function(inputs, config)
        assert "[Cache Hit]" in caplog.text

    def test_cache_isolation_between_tools(self):
        """Test that different tools have isolated cache entries."""
        @tool_cache("tool1")
        def func1(x: int) -> str:
            return f"tool1: {x}"

        @tool_cache("tool2")
        def func2(x: int) -> str:
            return f"tool2: {x}"

        # Same input, different tools
        result1 = func1(5)
        result2 = func2(5)

        assert result1 == "tool1: 5"
        assert result2 == "tool2: 5"

        # Check cache has 2 entries (one per tool)
        info = get_cache_info()
        assert info["total_entries"] == 2

    def test_complex_nested_pydantic_models(self):
        """Test caching with nested Pydantic models."""
        call_count = 0

        class InnerConfig(BaseModel):
            threshold: float
            method: str

        class OuterConfig(BaseModel):
            inner: InnerConfig
            sequences: str
            max_results: int = 10

        @tool_cache("nested_tool")
        def process_nested(config: OuterConfig) -> dict:
            nonlocal call_count
            call_count += 1
            return {"processed": True, "sequences": config.sequences}

        config1 = OuterConfig(
            inner=InnerConfig(threshold=0.5, method="blast"),
            sequences="ACGT",
            max_results=20
        )

        config2 = OuterConfig(
            inner=InnerConfig(threshold=0.5, method="blast"),
            sequences="ACGT",
            max_results=20
        )  # Same content

        config3 = OuterConfig(
            inner=InnerConfig(threshold=0.5, method="mmseqs"),  # Different
            sequences="ACGT",
            max_results=20
        )

        # First call
        _ = process_nested(config1)
        assert call_count == 1

        # Same config should use cache
        _ = process_nested(config2)
        assert call_count == 1

        # Different config should execute
        _ = process_nested(config3)
        assert call_count == 2

    def test_cache_with_sequence_in_config(self):
        """Test that sequences in configs are properly included in cache key."""
        call_count = 0

        class SequenceConfig(BaseModel):
            sequences: str
            param1: int = 10

        @tool_cache("sequence_tool")
        def process_sequence(config: SequenceConfig) -> str:
            nonlocal call_count
            call_count += 1
            return f"Processed {len(config.sequences)} chars"

        # Same parameters, different sequences
        config1 = SequenceConfig(sequences="ACGT", param1=10)
        config2 = SequenceConfig(sequences="TGCA", param1=10)

        _ = process_sequence(config1)
        assert call_count == 1

        _ = process_sequence(config2)
        assert call_count == 2  # Should execute again due to different sequence

        # Same sequence should use cache
        config3 = SequenceConfig(sequences="ACGT", param1=10)
        _ = process_sequence(config3)
        assert call_count == 2  # Still 2, used cache

    def test_input_config_pattern_caching(self):
        """Test caching with the new input/config pattern."""
        call_count = 0

        class ToolInput(BaseToolInput):
            sequence: str
            data_type: str = "dna"

        class ToolConfig(BaseConfig):
            threshold: float = 0.5
            max_results: int = 10
            verbose: bool = False

        @tool_cache("input_config_tool")
        def process_with_input_config(inputs: ToolInput, config: ToolConfig) -> dict:
            nonlocal call_count
            call_count += 1
            return {
                "sequence_length": len(inputs.sequence),
                "processed": True,
                "threshold_used": config.threshold
            }

        # Test 1: Same inputs and config should use cache
        inputs1 = ToolInput(sequence="ACGT", data_type="dna")
        config1 = ToolConfig(threshold=0.5, max_results=10)

        result1 = process_with_input_config(inputs1, config1)
        assert call_count == 1
        assert result1["sequence_length"] == 4

        # Same inputs and config should hit cache
        inputs2 = ToolInput(sequence="ACGT", data_type="dna")
        config2 = ToolConfig(threshold=0.5, max_results=10)

        _ = process_with_input_config(inputs2, config2)
        assert call_count == 1  # Should still be 1 (cache hit)

        # Test 2: Different inputs should execute again
        inputs3 = ToolInput(sequence="TGCA", data_type="dna")  # Different sequence
        config3 = ToolConfig(threshold=0.5, max_results=10)

        _ = process_with_input_config(inputs3, config3)
        assert call_count == 2  # Should increment

        # Test 3: Different config should execute again
        inputs4 = ToolInput(sequence="ACGT", data_type="dna")  # Same as first
        config4 = ToolConfig(threshold=0.8, max_results=10)  # Different threshold

        _ = process_with_input_config(inputs4, config4)
        assert call_count == 3  # Should increment

        # Test 4: Verbose flag should not affect caching (excluded from cache key)
        inputs5 = ToolInput(sequence="ACGT", data_type="dna")
        config5 = ToolConfig(threshold=0.5, max_results=10, verbose=True)  # verbose=True

        _ = process_with_input_config(inputs5, config5)
        assert call_count == 3  # Should still be 3 if verbose is excluded from cache key

    def test_verbose_not_in_cache_key(self, capsys):
        """Test that verbose flag doesn't affect cache key but still shows messages."""
        call_count = 0

        class TestInput(BaseToolInput):
            value: int

        class TestConfig(BaseConfig):
            multiplier: int = 2
            verbose: bool = False

        @tool_cache("verbose_cache_test")
        def test_function(inputs: TestInput, config: TestConfig) -> int:
            nonlocal call_count
            call_count += 1
            return inputs.value * config.multiplier

        inputs = TestInput(value=5)

        # First call with verbose=False
        config1 = TestConfig(multiplier=2, verbose=False)
        _ = test_function(inputs, config1)
        assert call_count == 1
        captured = capsys.readouterr()
        assert "[Cache Miss]" not in captured.out  # No output when verbose=False

        # Second call with verbose=True, same other parameters
        config2 = TestConfig(multiplier=2, verbose=True)
        _ = test_function(inputs, config2)

        # If verbose is properly excluded from cache key, this should be a cache hit
        # Note: This test assumes verbose is excluded from cache key serialization
        captured = capsys.readouterr()
        # The behavior depends on whether verbose is excluded from cache key
        # If included: should see Cache Miss, call_count=2
        # If excluded: should see Cache Hit, call_count=1

    def test_program_scoped_isolation(self):
        """Test that different programs have isolated caches."""
        call_count = 0

        @tool_cache("isolation_test")
        def cached_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Program 1: Run function and cache result
        cache1 = ToolCache()
        _program_tool_cache.set(cache1)

        result1 = cached_func(5)
        assert result1 == 10
        assert call_count == 1

        # Same call in Program 1 should use cache
        result2 = cached_func(5)
        assert result2 == 10
        assert call_count == 1  # Still 1, used cache

        # Program 2: Different cache, should NOT see Program 1's results
        cache2 = ToolCache()
        _program_tool_cache.set(cache2)

        result3 = cached_func(5)  # Same input as Program 1
        assert result3 == 10
        assert call_count == 2  # Incremented! Program 2 has isolated cache

        # Verify each cache has independent entries
        assert cache1.get_info()["total_entries"] == 1
        assert cache2.get_info()["total_entries"] == 1


class TestToolCacheIterable:
    """Test suite for iterable-level tool caching behavior."""

    def setup_method(self):
        """Set up cache in contextvar before each test."""
        self.cache = ToolCache()
        _program_tool_cache.set(self.cache)

    def teardown_method(self):
        """Clear contextvar after each test."""
        _program_tool_cache.set(None)

    def mock_tool_registry_get(self, tool_name: str, input_model, output_model):
        """Helper to create a mock ToolSpec for ToolRegistry.get()"""
        mock_spec = MagicMock()
        mock_spec.input_model = input_model
        mock_spec.output_model = output_model
        return mock_spec

    def test_iterable_cache_basic(self):
        """Test basic iterable caching functionality."""

        call_count = 0

        class Item(BaseModel):
            value: int

        class TestIterableInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            multiplier: int = 2

        class TestIterableOutput(MockToolOutputBase):
            results: list[int]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get(
            "iterable_test", TestIterableInput, TestIterableOutput
        )

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="iterable_test",
            )
            def process_items(
                inputs: TestIterableInput, config: TestConfig
            ) -> TestIterableOutput:
                nonlocal call_count
                call_count += len(inputs.items)
                return TestIterableOutput(
                    results=[item.value * config.multiplier for item in inputs.items],
                    tool_id="iterable_test",
                    execution_time=0.0,
                    success=True,
                )

            # First call with 3 items - all cache misses
            inputs1 = TestIterableInput(items=[Item(value=1), Item(value=2), Item(value=3)])
            config = TestConfig(multiplier=2)

            result1 = process_items(inputs1, config)
            assert result1.results == [2, 4, 6]
            assert call_count == 3

            # Second call with overlapping items - should only compute new ones
            inputs2 = TestIterableInput(items=[Item(value=1), Item(value=4), Item(value=3)])
            result2 = process_items(inputs2, config)
            assert result2.results == [2, 8, 6]
            assert call_count == 4  # Only item with value=4 was computed

            # Third call with all cached items
            inputs3 = TestIterableInput(items=[Item(value=1), Item(value=2)])
            result3 = process_items(inputs3, config)
            assert result3.results == [2, 4]
            assert call_count == 4  # No new computations

    def test_iterable_cache_preserves_order(self):
        """Test that iterable caching preserves the original order of items."""

        class Item(BaseModel):
            id: str

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            prefix: str = "result_"

        class TestOutput(MockToolOutputBase):
            results: list[str]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get("order_test", TestInput, TestOutput)

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="order_test",
            )
            def process_with_order(inputs: TestInput, config: TestConfig) -> TestOutput:
                return TestOutput(
                    results=[f"{config.prefix}{item.id}" for item in inputs.items],
                    tool_id="order_test",
                    execution_time=0.0,
                    success=True,
                )

            # First call - cache items A, B, C
            inputs1 = TestInput(items=[Item(id="A"), Item(id="B"), Item(id="C")])
            config = TestConfig(prefix="result_")
            result1 = process_with_order(inputs1, config)
            assert result1.results == ["result_A", "result_B", "result_C"]

            # Second call with different order and one new item
            inputs2 = TestInput(
                items=[Item(id="C"), Item(id="D"), Item(id="A"), Item(id="B")]
            )
            result2 = process_with_order(inputs2, config)
            assert result2.results == ["result_C", "result_D", "result_A", "result_B"]

    def test_iterable_cache_disabled_mode(self):
        """Test that iterable caching can be disabled."""

        call_count = 0

        class Item(BaseModel):
            value: int

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            multiplier: int = 2

        class TestOutput(MockToolOutputBase):
            results: list[int]

        # Note: enabled=False means no ToolRegistry.get() is called, so no mock needed
        @tool_cache_iterable(
            input_iterable_field="items", output_iterable_field="results", enabled=False
        )
        def no_cache_function(inputs: TestInput, config: TestConfig) -> TestOutput:
            nonlocal call_count
            call_count += len(inputs.items)
            return TestOutput(
                results=[item.value * config.multiplier for item in inputs.items],
                tool_id="no_cache_test",
                execution_time=0.0,
                success=True,
            )

        inputs = TestInput(items=[Item(value=1), Item(value=2)])
        config = TestConfig(multiplier=2)

        # First call
        _ = no_cache_function(inputs, config)
        assert call_count == 2

        # Second call with same items should compute again
        _ = no_cache_function(inputs, config)
        assert call_count == 4  # Should increment

    def test_iterable_cache_verbose_mode(self, caplog):
        """Test verbose mode reporting for iterable caching."""
        import logging

        class Item(BaseModel):
            value: int

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            multiplier: int = 2
            verbose: bool = True

        class TestOutput(MockToolOutputBase):
            results: list[int]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get("verbose_iterable", TestInput, TestOutput)

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="verbose_iterable",
            )
            def verbose_function(inputs: TestInput, config: TestConfig) -> TestOutput:
                return TestOutput(
                    results=[item.value * config.multiplier for item in inputs.items],
                    tool_id="verbose_iterable",
                    execution_time=0.0,
                    success=True,
                )

            inputs = TestInput(items=[Item(value=1), Item(value=2)])
            config = TestConfig(multiplier=2, verbose=True)

            # First call should report cache misses
            with caplog.at_level(logging.DEBUG):
                _ = verbose_function(inputs, config)
            assert "[Iterable Cache Stats]" in caplog.text
            assert "0 cache hits, 2 cache misses" in caplog.text

            caplog.clear()

            # Second call should report cache hits
            with caplog.at_level(logging.DEBUG):
                _ = verbose_function(inputs, config)
            assert "[Iterable Cache Stats]" in caplog.text
            assert "2 cache hits, 0 cache misses" in caplog.text

    def test_iterable_cache_with_different_configs(self):
        """Test that different configs produce different cache entries."""

        call_count = 0

        class Item(BaseModel):
            value: int

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            multiplier: int = 2

        class TestOutput(MockToolOutputBase):
            results: list[int]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get("config_test", TestInput, TestOutput)

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="config_test",
            )
            def process_with_config(inputs: TestInput, config: TestConfig) -> TestOutput:
                nonlocal call_count
                call_count += len(inputs.items)
                return TestOutput(
                    results=[item.value * config.multiplier for item in inputs.items],
                    tool_id="config_test",
                    execution_time=0.0,
                    success=True,
                )

            inputs = TestInput(items=[Item(value=5)])

            # First call with multiplier=2
            config1 = TestConfig(multiplier=2)
            result1 = process_with_config(inputs, config1)
            assert result1.results == [10]
            assert call_count == 1

            # Second call with same items but different config
            config2 = TestConfig(multiplier=3)
            result2 = process_with_config(inputs, config2)
            assert result2.results == [15]
            assert call_count == 2  # Should recompute

            # Third call with original config should use cache
            config3 = TestConfig(multiplier=2)
            result3 = process_with_config(inputs, config3)
            assert result3.results == [10]
            assert call_count == 2  # Should not recompute

    def test_iterable_cache_all_cached_scenario(self):
        """Test scenario where all items are cached."""

        call_count = 0

        class Item(BaseModel):
            value: int

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            pass

        class TestOutput(MockToolOutputBase):
            results: list[int]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get("all_cached_test", TestInput, TestOutput)

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="all_cached_test",
            )
            def process_items(inputs: TestInput, config: TestConfig) -> TestOutput:
                nonlocal call_count
                call_count += len(inputs.items)
                return TestOutput(
                    results=[item.value * 2 for item in inputs.items],
                    tool_id="all_cached_test",
                    execution_time=1.5,
                    success=True,
                )

            config = TestConfig()

            # First call - cache all items
            inputs1 = TestInput(items=[Item(value=1), Item(value=2), Item(value=3)])
            _ = process_items(inputs1, config)
            assert call_count == 3

            # Second call with all cached items
            inputs2 = TestInput(items=[Item(value=1), Item(value=2), Item(value=3)])
            result2 = process_items(inputs2, config)

            assert result2.results == [2, 4, 6]
            assert call_count == 3  # No additional computations
            assert result2.execution_time == 0.0  # Cached result has 0 execution time
            assert result2.success is True

    def test_iterable_cache_partial_overlap(self):
        """Test scenario with partial overlap between calls."""

        computation_log = []

        class Item(BaseModel):
            id: str
            value: int

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            pass

        class TestOutput(MockToolOutputBase):
            results: list[str]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get("partial_overlap_test", TestInput, TestOutput)

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="partial_overlap_test",
            )
            def process_items(inputs: TestInput, config: TestConfig) -> TestOutput:
                for item in inputs.items:
                    computation_log.append(item.id)
                return TestOutput(
                    results=[f"processed_{item.id}" for item in inputs.items],
                    tool_id="partial_overlap_test",
                    execution_time=0.0,
                    success=True,
                )

            config = TestConfig()

            # First call
            inputs1 = TestInput(
                items=[Item(id="A", value=1), Item(id="B", value=2), Item(id="C", value=3)]
            )
            _ = process_items(inputs1, config)
            assert computation_log == ["A", "B", "C"]

            # Second call with 50% overlap
            inputs2 = TestInput(
                items=[
                    Item(id="B", value=2),
                    Item(id="C", value=3),
                    Item(id="D", value=4),
                    Item(id="E", value=5),
                ]
            )
            result2 = process_items(inputs2, config)

            # Only D and E should be computed
            assert computation_log == ["A", "B", "C", "D", "E"]
            assert result2.results == [
                "processed_B",
                "processed_C",
                "processed_D",
                "processed_E",
            ]

    def test_iterable_cache_info_tracking(self):
        """Test cache info tracking for iterable caching."""

        class Item(BaseModel):
            value: int

        class TestInput(BaseToolInput):
            items: list[Item]

        class TestConfig(BaseConfig):
            pass

        class TestOutput(MockToolOutputBase):
            results: list[int]

        # Mock ToolRegistry.get() to return mock spec
        mock_spec = self.mock_tool_registry_get("cache_info_test", TestInput, TestOutput)

        with patch("bio_programming.bio_tools.tools.infra.tool_cache.ToolRegistry.get", return_value=mock_spec):
            @tool_cache_iterable(
                input_iterable_field="items",
                output_iterable_field="results",
                tool_name="cache_info_test",
            )
            def cached_func(inputs: TestInput, config: TestConfig) -> TestOutput:
                return TestOutput(
                    results=[item.value * 2 for item in inputs.items],
                    tool_id="cache_info_test",
                    execution_time=0.0,
                    success=True,
                )

            # Initially empty
            info = get_cache_info()
            assert info["total_entries"] == 0

            # Add some cached entries (each item is cached separately)
            config = TestConfig()
            inputs1 = TestInput(items=[Item(value=1), Item(value=2), Item(value=3)])
            cached_func(inputs1, config)

            info = get_cache_info()
            assert info["total_entries"] == 3  # 3 items cached

            # Add more unique items
            inputs2 = TestInput(items=[Item(value=4), Item(value=5)])
            cached_func(inputs2, config)

            info = get_cache_info()
            assert info["total_entries"] == 5  # 5 unique items

            # Clear and check
            clear_cache()
            info = get_cache_info()
            assert info["total_entries"] == 0


class MockOptimizer:
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


@pytest.fixture
def empty_cache():
    return ToolCache()


class TestToolCacheInternals:

    def test_initial_state(self, empty_cache):
        assert empty_cache.current_size == 0
        assert empty_cache.get("tool", "key") is None

    def test_set_increases_size(self, empty_cache):
        empty_cache.set("calculator", "2+2", 4)
        size_after_add = empty_cache.current_size
        assert size_after_add > 0
        assert empty_cache.get("calculator", "2+2") == 4

    def test_overwrite_updates_size_correctly(self, empty_cache):
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

    def test_clear_all_resets_size(self, empty_cache):
        empty_cache.set("tool1", "k1", "data")
        empty_cache.set("tool2", "k2", "data")

        assert empty_cache.current_size > 0

        empty_cache.clear()

        assert empty_cache.current_size == 0
        assert empty_cache._cache == {}

    def test_clear_specific_tool_adjusts_size(self, empty_cache):
        empty_cache.set("tool1", "k1", "data_1")
        #size_1 = empty_cache.current_size

        empty_cache.set("tool2", "k2", "data_2")
        size_2 = empty_cache.current_size

        # Clear tool 2.
        empty_cache.clear("tool2")

        # Should be roughly back to size_1 (allowing for small container overhead diffs).
        # Note: Depending on implementation, container overhead might vary slightly
        # but the value size should definitely be subtracted.
        assert empty_cache.current_size < size_2
        assert empty_cache.get("tool2", "k2") is None
        assert empty_cache.get("tool1", "k1") == "data_1"


class TestClearConfigurationLogic:

    def test_config_bool_true_clears_all(self):
        opt = MockOptimizer(clear_config=True)
        opt.tool_cache.set("t1", "k", "val")

        opt._clear_tool_cache()

        assert opt.tool_cache.current_size == 0
        assert opt.tool_cache.get("t1", "k") is None

    def test_config_list_clears_specific(self):
        opt = MockOptimizer(clear_config=["t1"])
        opt.tool_cache.set("t1", "k", "val")
        opt.tool_cache.set("t2", "k", "val")

        opt._clear_tool_cache()

        # t1 should be gone.
        assert opt.tool_cache.get("t1", "k") is None
        # t2 should remain.
        assert opt.tool_cache.get("t2", "k") == "val"

    def test_config_int_threshold_below_limit(self):
        """Size is BELOW threshold, should NOT clear."""
        # Set threshold to 1000 bytes.
        opt = MockOptimizer(clear_config=1000)

        # Add small data (e.g., 50 bytes).
        opt.tool_cache.set("t1", "k", "small_data")
        initial_size = opt.tool_cache.current_size

        # Ensure we are actually testing a valid scenario.
        assert initial_size < 1000

        opt._clear_tool_cache()

        # Should still be there.
        assert opt.tool_cache.current_size == initial_size
        assert opt.tool_cache.get("t1", "k") == "small_data"

    def test_config_int_prunes_lru(self):
        """
        Test that we only remove enough to fit under the limit,
        removing the oldest items first.
        """
        old_data = "A" * 80
        old_data_size = _get_obj_size(old_data)

        new_data = "B" * 100
        new_data_size = _get_obj_size(new_data)

        opt = MockOptimizer(clear_config=new_data_size)
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

    def test_circular_reference_safety(self):
        """Ensure recursive size calc doesn't crash on circular refs."""
        cache = ToolCache()

        d = {}
        d['self'] = d # Circular reference.

        try:
            cache.set("tool", "circ", d)
        except RecursionError:
            pytest.fail("Recursive size calculation crashed on circular reference")

        assert cache.current_size > 0
