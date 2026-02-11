"""
tool_cache.py

Tool cache utilities for caching expensive tool operations.

This module provides program-scoped caching capabilities for tools via a decorator
that transparently handles result caching based on input parameters. Each Program
instance maintains its own isolated cache using Python's contextvars.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from bio_programming_tools.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Context variable for program-scoped cache
_program_tool_cache: ContextVar[ToolCache | None] = ContextVar(
    "_program_tool_cache", default=None
)

T = TypeVar("T")


def _get_obj_size(obj: Any, seen: set[int] = None) -> int:
    """Calculates size recursively."""
    if seen is None:
        seen = set()

    try:
        obj_id = id(obj)
    except Exception:
        return 0
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # For data arrays.
    if hasattr(obj, "nbytes"):  # NumPy / Pandas
        return obj.nbytes
    if hasattr(obj, "element_size") and hasattr(obj, "nelement"):  # PyTorch
        return obj.element_size() * obj.nelement()

    # Standard Python objects.
    try:
        size = sys.getsizeof(obj)
    except Exception:
        return 0

    if isinstance(obj, dict):
        size += sum(
            _get_obj_size(k, seen) + _get_obj_size(v, seen) for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set)):
        size += sum(_get_obj_size(i, seen) for i in obj)

    return size


class ToolCache:
    """
    Program-scoped cache for tool results.

    Each Program/Optimizer instance creates its own ToolCache, ensuring
    isolation between different optimization runs.
    """

    def __init__(self):
        """Initialize empty cache storage."""
        self._cache: dict[str, dict[str, Any]] = {}
        self._current_size_bytes = 0

    @property
    def current_size(self) -> int:
        """Return the currently tracked size in bytes."""
        return self._current_size_bytes

    def get(self, tool_name: str, cache_key: str) -> Any | None:
        """
        Get cached result for a tool and cache key.

        Args:
            tool_name: Name of the tool
            cache_key: Cache key for the specific invocation

        Returns:
            Cached result if available, None otherwise
        """
        tool_cache = self._cache.get(tool_name)
        if tool_cache is None:
            return None

        if cache_key in tool_cache:
            # Mark as most-recently used.
            # Move tool to the end of the main cache.
            val = self._cache.pop(tool_name)
            self._cache[tool_name] = val
            # Move the specific entry the end of the tool's cache dict.
            result = tool_cache.pop(cache_key)
            tool_cache[cache_key] = result
            return result

        return None

    def set(self, tool_name: str, cache_key: str, result: Any) -> None:
        """
        Store result in cache for a tool and cache key.

        Args:
            tool_name: Name of the tool
            cache_key: Cache key for the specific invocation
            result: Result to cache
        """
        # If the tool exists, move it to the end of the cache (mark as recently used).
        if tool_name in self._cache:
            self._cache[tool_name] = self._cache.pop(tool_name)
        else:
            self._cache[tool_name] = {}

        tool_entries = self._cache[tool_name]

        # If key exists, remove it first so the new insertion goes to the end.
        if cache_key in tool_entries:
            old_val = tool_entries.pop(cache_key)
            self._current_size_bytes -= _get_obj_size(old_val)

        tool_entries[cache_key] = result
        self._current_size_bytes += _get_obj_size(result)
        logger.debug(f"ToolCache.set: {tool_name}, size={self._current_size_bytes} bytes")

    def clear(self, tool_name: str | None = None) -> int:
        """
        Clear cache entries.

        Args:
            tool_name: If provided, clear only this tool's cache.
                       If None, clear all cache entries.

        Returns:
            Number of entries cleared
        """
        if tool_name:
            count = len(self._cache.get(tool_name, {}))
            popped = self._cache.pop(tool_name, None)
            if popped is not None:
                for cache_key in popped:
                    self._current_size_bytes -= _get_obj_size(popped[cache_key])
            return count
        else:
            count = sum(len(cache_dict) for cache_dict in self._cache.values())
            self._cache.clear()
            self._current_size_bytes = 0
            return count

    def prune(self, target_size: int = 0) -> None:
        """
        Removes items until the cache size is below target_size.
        Uses LRU eviction strategy (prioritizes least recently used entries).

        Args:
            target_size: The desired cache size in bytes.
        """
        if self._current_size_bytes <= target_size:
            return

        logger.debug(f"ToolCache.prune: {self._current_size_bytes} bytes -> target {target_size} bytes")
        # Create a list of keys to avoid runtime error during modification.
        tools_to_check = list(self._cache.keys())

        for tool in tools_to_check:
            if self._current_size_bytes <= target_size:
                break

            tool_entries = self._cache[tool]
            keys_to_remove = list(tool_entries.keys())

            for key in keys_to_remove:
                if self._current_size_bytes <= target_size:
                    break

                val = tool_entries[key]
                size = _get_obj_size(val)

                del tool_entries[key]
                self._current_size_bytes -= size

            # Clean up empty tool dictionaries.
            if not tool_entries:
                del self._cache[tool]

    def get_info(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including total entries and estimated size
        """
        total_entries = sum(len(cache_dict) for cache_dict in self._cache.values())
        return {
            "total_entries": total_entries,
            "cache_size_bytes": self._current_size_bytes,
        }


def _serialize_for_cache_key(obj: Any) -> str:
    """
    Convert any object to a string representation suitable for cache key generation.

    Handles Pydantic models, basic types, lists, dicts, etc.
    """
    if isinstance(obj, BaseModel):
        # For Pydantic models, convert to dict first then JSON for deterministic serialization
        # Exclude verbose field since it only affects logging, not computation results
        model_dict = obj.model_dump(exclude_none=True, exclude={"verbose"})
        return json.dumps(model_dict, sort_keys=True, default=str)
    elif isinstance(obj, (dict, list, tuple)):
        # For collections, use JSON with sorted keys
        return json.dumps(obj, sort_keys=True, default=str)
    else:
        # For basic types, convert to string
        return str(obj)


def _generate_cache_key(tool_name: str, *args, **kwargs) -> str:
    """
    Generate a deterministic cache key for tool operations.

    Args:
        tool_name: Name of the tool
        *args: Positional arguments to the tool
        **kwargs: Keyword arguments to the tool

    Returns:
        A deterministic hash key for the cache
    """
    # Create a list of all parameters in a deterministic order
    key_parts = [tool_name]

    # Add positional arguments
    for arg in args:
        key_parts.append(_serialize_for_cache_key(arg))

    # Add keyword arguments in sorted order
    for key in sorted(kwargs.keys()):
        key_parts.append(f"{key}={_serialize_for_cache_key(kwargs[key])}")

    # Generate MD5 hash of the combined key parts
    combined = "|".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def tool_cache(tool_name: str | None = None, enabled: bool = True) -> Callable:
    """
    Decorator that adds transparent caching to tool functions. Reads cache from
    the program-scoped contextvar set by the Optimizer.

    This decorator caches the entire output based on the complete input and config.
    For tools that process batches of independent items, consider using
    @tool_cache_iterable instead for more granular caching.

    Args:
        tool_name: Optional name for the tool. If not provided, uses the function name.
            When provided, use the registry key (kebab-case, e.g. "mafft-align") for consistency.
            Tool functions use the run_ prefix (e.g. run_mafft_align); registry keys use kebab-case.
        enabled: Whether caching is enabled. Useful for testing.

    Example:
        @tool(
            key="sequence-analysis",
            label="Sequence Analysis",
            input=SequenceInput,
            config=AnalysisConfig,
            output=AnalysisOutput,
            description="Analyze DNA sequences",
            category="analysis"
        )
        @tool_cache("sequence-analysis")
        def analyze_sequence(
            inputs: SequenceInput,
            config: AnalysisConfig
        ) -> AnalysisOutput:
            # Expensive computation here
            result = expensive_analysis(inputs.sequence, config.threshold)
            return AnalysisOutput(
                score=result.score,
            )

        # First call - cache miss, computation runs
        result1 = analyze_sequence(
            SequenceInput(sequence="ACGT"),
            AnalysisConfig(threshold=0.5)
        )

        # Second call with same inputs - cache hit, returns cached result
        result2 = analyze_sequence(
            SequenceInput(sequence="ACGT"),
            AnalysisConfig(threshold=0.5)
        )

        # Different inputs - cache miss, computation runs
        result3 = analyze_sequence(
            SequenceInput(sequence="TGCA"),
            AnalysisConfig(threshold=0.5)
        )
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        actual_tool_name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # If caching is disabled, just run the function
            if not enabled:
                return func(*args, **kwargs)

            # Get cache from contextvar
            cache = _program_tool_cache.get()
            if cache is None:
                # No cache set - run without caching
                return func(*args, **kwargs)

            # Bind arguments to function signature to normalize positional/keyword args
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Apply defaults for correct cache key generation

            # Generate cache key using normalized arguments
            cache_key = _generate_cache_key(actual_tool_name, *args, **kwargs)

            # Check cache
            cached_result = cache.get(actual_tool_name, cache_key)
            if cached_result is not None:
                logger.debug(f"[Cache Hit] {actual_tool_name}: Using cached result")
                return cached_result

            # Run the function
            logger.debug(f"[Cache Miss] {actual_tool_name}: Computing result")

            result = func(*args, **kwargs)

            # Cache the result
            cache.set(actual_tool_name, cache_key, result)

            return result

        # Add utility methods to the wrapper
        wrapper.clear_cache = lambda: clear_tool_cache(actual_tool_name)
        wrapper.cache_info = lambda: get_cache_info(actual_tool_name)

        return wrapper

    return decorator


def tool_cache_iterable(
    input_iterable_field: str,
    output_iterable_field: str,
    tool_name: str | None = None,
    enabled: bool = True,
) -> Callable:
    """
    Decorator that adds iterable-level caching to tool functions that process
    independent items in batches.

    Unlike @tool_cache which caches the entire output based on all inputs,
    this decorator caches individual items within an iterable. This is ideal
    for batch processing tools where each item is processed independently
    (e.g., structure prediction for multiple protein complexes).

    The decorator:
    1. Checks cache for each item in the input iterable individually
    2. Only passes uncached items to the underlying function
    3. Combines cached and newly computed results in original order
    4. Caches newly computed items for future calls

    Args:
        input_iterable_field: Name of the field in the input Pydantic model that contains the iterable
            of items to process as independent cache entries.
        output_iterable_field: Name of the field in the output Pydantic model that contains the iterable
            of output items corresponding to the same order as the input iterable.
        tool_name: Optional name for the tool. If not provided, uses the function name.
        enabled: Whether caching is enabled. Useful for testing.

    Example:
        @tool(
            key="esmfold",
            label="ESMFold Structure Prediction",
            input=ESMFoldInput,
            config=ESMFoldConfig,
            output=ESMFoldOutput,
            description="Protein structure prediction using ESMFold",
            category="structure_prediction",
            requires_gpu=True
        )
        @tool_cache_iterable(
            input_iterable_field="complexes",
            output_iterable_field="structures",
            tool_name="esmfold"
        )
        def run_esmfold(
            inputs: ESMFoldInput,
            config: ESMFoldConfig
        ) -> ESMFoldOutput:
            # Process complexes and return structures
            structures = [predict_structure(c) for c in inputs.complexes]
            return ESMFoldOutput(
                structures=structures,
            )

        # First call with 3 complexes - all cache misses
        result1 = run_esmfold(
            ESMFoldInput(complexes=[complex_A, complex_B, complex_C]),
            ESMFoldConfig()
        )
        # Computes all 3 structures, caches each individually

        # Second call with overlapping complexes - partial cache hits
        result2 = run_esmfold(
            ESMFoldInput(complexes=[complex_A, complex_D, complex_C]),
            ESMFoldConfig()
        )
        # Returns cached A and C, only computes D
        # result2.structures = [cached_A, computed_D, cached_C]

        # Third call with all cached - all cache hits
        result3 = run_esmfold(
            ESMFoldInput(complexes=[complex_B, complex_A]),
            ESMFoldConfig()
        )
        # Returns all cached, no computation
        # result3.execution_time = 0.0

    Notes:
        - Order is preserved: output[i] always corresponds to input[i]
        - Config changes invalidate cache for all items
        - Verbose mode shows cache hit/miss statistics per call
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        actual_tool_name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # If caching is disabled, just run the function
            if not enabled:
                return func(*args, **kwargs)

            # Get cache from contextvar
            cache = _program_tool_cache.get()
            if cache is None:
                # No cache set - run without caching
                return func(*args, **kwargs)

            # Bind arguments to function signature to normalize positional/keyword args
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()  # Apply defaults for correct cache key generation

            # Extract inputs and config
            inputs = bound_args.arguments.get("inputs")
            config = bound_args.arguments.get("config")

            # Extract the iterable field from the inputs
            input_items = getattr(inputs, input_iterable_field)

            # Track which items need computation vs which are already cached
            cached_results = []  # tuple of (index, cached_result)
            uncached_indices = []  # Original indices
            uncached_input_items = []  # Input items that need computation

            # For each item in the input iterable
            for idx, item in enumerate(input_items):
                # Generate cache key
                cache_key = _generate_cache_key(
                    actual_tool_name, input_item=item, config=config
                )

                # Check cache
                cached_result = cache.get(actual_tool_name, cache_key)
                if cached_result is not None:
                    cached_results.append((idx, cached_result))
                else:
                    uncached_indices.append(idx)
                    uncached_input_items.append(item)

            # Log cache statistics
            num_cache_hits = len(cached_results)
            num_cache_misses = len(uncached_indices)
            total_items = len(input_items)
            logger.debug(
                f"[Iterable Cache Stats] {actual_tool_name}: {num_cache_hits} cache hits, {num_cache_misses} cache misses out of {total_items} items"
            )

            # Get the input and output pydantic model classes from the ToolRegistry
            tool_spec = ToolRegistry.get(actual_tool_name)
            input_model = tool_spec.input_model
            output_model = tool_spec.output_model

            # If everything is cached, return the cached results in a new OutputPydantic model
            if num_cache_misses == 0:
                logger.debug(f"[Iterable Cache] {actual_tool_name}: full cache hit, skipping computation")
                # Create new output model instance with cached results
                output_instance = output_model(
                    tool_id=actual_tool_name,
                    execution_time=0.0,
                    success=True,
                    warnings=[],
                    metadata={},
                    **{
                        output_iterable_field: [
                            cached_result for _, cached_result in cached_results
                        ]
                    },
                )
                return output_instance

            # If there are some items that need computation
            else:
                logger.debug(f"[Iterable Cache] {actual_tool_name}: computing {num_cache_misses} uncached items")
                # Create new input model instance with the uncached items
                input_data = inputs.model_dump()
                # Update only the iterable field
                input_data[input_iterable_field] = uncached_input_items
                input_model_instance = input_model(**input_data)

                # Run the function on the uncached items
                tool_output = func(inputs=input_model_instance, config=config)

                # Cache the results
                result_iterable = getattr(tool_output, output_iterable_field, [])
                for input_item, result_item in zip(
                    uncached_input_items, result_iterable
                ):
                    cache.set(
                        actual_tool_name,
                        _generate_cache_key(
                            actual_tool_name, input_item=input_item, config=config
                        ),
                        result_item,
                    )

                # Reconstruct the final ordering
                result_map = dict(cached_results)
                for orig_idx, result_item in zip(uncached_indices, result_iterable):
                    result_map[orig_idx] = result_item

                final_output_iterable = [result_map[i] for i in range(total_items)]

                # Create new output model instance with cached and uncached results in original order
                output_instance = output_model(
                    tool_id=tool_output.tool_id,
                    execution_time=tool_output.execution_time,
                    success=tool_output.success,
                    warnings=tool_output.warnings,
                    metadata=tool_output.metadata,
                    **{output_iterable_field: final_output_iterable},
                )
                return output_instance

        # Add utility methods to the wrapper
        wrapper.clear_cache = lambda: clear_tool_cache(actual_tool_name)
        wrapper.cache_info = lambda: get_cache_info(actual_tool_name)

        return wrapper

    return decorator


def clear_cache() -> None:
    """
    Clear all cached results from the program-scoped cache.

    Gets the cache from contextvar and clears it.
    """
    cache = _program_tool_cache.get()
    if cache:
        cache.clear()


def clear_tool_cache(tool_name: str) -> int:
    """
    Clear cache entries for a specific tool.

    Args:
        tool_name: Name of the tool to clear cache for

    Returns:
        Number of entries cleared
    """
    cache = _program_tool_cache.get()
    if cache:
        return cache.clear(tool_name)
    return 0


def get_cache_info(tool_name: str | None = None) -> dict[str, Any]:
    """
    Get information about the cache.

    Args:
        tool_name: Not currently used, but reserved for future filtering

    Returns:
        Dictionary with cache statistics
    """
    _ = tool_name  # Reserved for future use
    cache = _program_tool_cache.get()
    if cache:
        return cache.get_info()
    return {"total_entries": 0, "cache_size_bytes": 0}


def has_cached_entries(tool_name: str) -> bool:
    """
    Check if a specific tool has any cached entries.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if the tool has cached entries, False otherwise
    """
    cache = _program_tool_cache.get()
    if cache is None:
        return False

    tool_cache = cache._cache.get(tool_name)
    return tool_cache is not None and len(tool_cache) > 0
