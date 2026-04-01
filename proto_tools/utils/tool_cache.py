"""proto_tools/utils/tool_cache.py.

Tool cache utilities for caching expensive tool operations.

This module provides program-scoped caching capabilities for tools. Caching is
enabled by setting ``cacheable=True`` on the ``@tool()`` decorator, which
auto-selects the strategy:

- **Iterable tools** (have ``iterable_input_field``) → per-item cache
  (strip cached items, dispatch uncached only, stitch results back).
- **Non-iterable tools** → whole-output cache (hash full inputs + config).

Each Program instance maintains its own isolated cache using Python's
contextvars.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Context variable for program-scoped cache
_program_tool_cache: ContextVar[ToolCache | None] = ContextVar("_program_tool_cache", default=None)


def _get_obj_size(obj: Any, seen: set[int] | None = None) -> int:
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
        return obj.nbytes  # type: ignore[no-any-return]
    if hasattr(obj, "element_size") and hasattr(obj, "nelement"):  # PyTorch
        return obj.element_size() * obj.nelement()  # type: ignore[no-any-return]

    # Standard Python objects.
    try:
        size = sys.getsizeof(obj)
    except Exception:
        return 0

    if isinstance(obj, dict):
        size += sum(_get_obj_size(k, seen) + _get_obj_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(_get_obj_size(i, seen) for i in obj)

    return size


class ToolCache:
    """Program-scoped cache for tool results.

    Each Program/Optimizer instance creates its own ToolCache, ensuring
    isolation between different optimization runs.
    """

    def __init__(self) -> None:
        """Initialize empty cache storage."""
        self._cache: dict[str, dict[str, Any]] = {}
        self._current_size_bytes = 0

    @property
    def current_size(self) -> int:
        """Return the currently tracked size in bytes."""
        return self._current_size_bytes

    def get(self, tool_name: str, cache_key: str) -> Any | None:
        """Get cached result for a tool and cache key.

        Args:
            tool_name (str): Name of the tool
            cache_key (str): Cache key for the specific invocation

        Returns:
            Any | None: Cached result if available, None otherwise
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
        """Store result in cache for a tool and cache key.

        Args:
            tool_name (str): Name of the tool
            cache_key (str): Cache key for the specific invocation
            result (Any): Result to cache
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
        """Clear cache entries.

        Args:
            tool_name (str | None): If provided, clear only this tool's cache.
                       If None, clear all cache entries.

        Returns:
            int: Number of entries cleared
        """
        if tool_name:
            count = len(self._cache.get(tool_name, {}))
            popped = self._cache.pop(tool_name, None)
            if popped is not None:
                for cache_key in popped:
                    self._current_size_bytes -= _get_obj_size(popped[cache_key])
            return count
        count = sum(len(cache_dict) for cache_dict in self._cache.values())
        self._cache.clear()
        self._current_size_bytes = 0
        return count

    def prune(self, target_size: int = 0) -> None:
        """Removes items until the cache size is below target_size.

        Uses LRU eviction strategy (prioritizes least recently used entries).

        Args:
            target_size (int): The desired cache size in bytes.
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
        """Get cache statistics.

        Returns:
            dict[str, Any]: Dictionary with cache statistics including total entries and estimated size
        """
        total_entries = sum(len(cache_dict) for cache_dict in self._cache.values())
        return {
            "total_entries": total_entries,
            "cache_size_bytes": self._current_size_bytes,
        }


def _serialize_for_cache_key(obj: Any) -> str:
    """Convert any object to a string representation suitable for cache key generation.

    Handles Pydantic models, basic types, lists, dicts, etc.
    Fields marked with ``include_in_key=False`` on their ConfigField are excluded.

    Args:
        obj (Any): Object to serialize into a deterministic string.
    """
    if hasattr(obj, "cache_key"):
        return obj.cache_key()  # type: ignore[no-any-return]
    if isinstance(obj, BaseModel):
        model_dict = obj.model_dump(exclude_none=True)
        return json.dumps(model_dict, sort_keys=True, default=str)
    if isinstance(obj, (dict, list, tuple)):
        # For collections, use JSON with sorted keys
        return json.dumps(obj, sort_keys=True, default=str)
    # For basic types, convert to string
    return str(obj)


def _generate_cache_key(tool_name: str, *args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key for tool operations.

    Args:
        tool_name (str): Name of the tool.
        args: Positional arguments to the tool (via ``*args``).
        kwargs: Keyword arguments to the tool (via ``**kwargs``).

    Returns:
        str: A deterministic hash key for the cache
    """
    # Create a list of all parameters in a deterministic order
    key_parts = [tool_name]

    # Add positional arguments
    key_parts.extend(_serialize_for_cache_key(arg) for arg in args)

    # Add keyword arguments in sorted order
    key_parts.extend(f"{key}={_serialize_for_cache_key(kwargs[key])}" for key in sorted(kwargs.keys()))

    # Generate MD5 hash of the combined key parts
    combined = "|".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]  # noqa: S324 -- cache key, not security


@dataclass
class DeduplicatedItems:
    """Result of deduplicating a list of items by cache key.

    Attributes:
        unique_items (list[Any]): Deduplicated items in first-occurrence order.
        unique_keys (list[str]): Cache keys for each unique item (1:1 with unique_items).
        index_map (list[tuple[int, int]]): Maps every original position to its unique index.
            ``index_map[i]`` is ``(i, unique_idx)`` where ``unique_idx``
            indexes into ``unique_items``.
    """

    unique_items: list[Any]
    unique_keys: list[str]
    index_map: list[tuple[int, int]]


def deduplicate_items(
    items: list[Any],
    key_fn: Callable[[Any], str],
) -> DeduplicatedItems:
    """Deduplicate items by a serialization key function.

    Args:
        items (list[Any]): List of items to deduplicate.
        key_fn (Callable[[Any], str]): Callable that produces a deterministic string key for each item.

    Returns:
        DeduplicatedItems: A ``DeduplicatedItems`` with unique items, their keys, and a full
            index map from original positions to unique positions.
    """
    unique_items: list[Any] = []
    unique_keys: list[str] = []
    key_to_unique_idx: dict[str, int] = {}
    index_map: list[tuple[int, int]] = []

    for i, item in enumerate(items):
        k = key_fn(item)
        if k not in key_to_unique_idx:
            unique_idx = len(unique_items)
            key_to_unique_idx[k] = unique_idx
            unique_items.append(item)
            unique_keys.append(k)
        else:
            unique_idx = key_to_unique_idx[k]
        index_map.append((i, unique_idx))

    return DeduplicatedItems(
        unique_items=unique_items,
        unique_keys=unique_keys,
        index_map=index_map,
    )


# ============================================================================
# Per-item cache helpers (used by @tool wrapper for iterable cacheable tools)
# ============================================================================


@dataclass
class CacheStripResult:
    """Result of stripping cached items from an iterable input.

    Attributes:
        uncached_items (list[Any]): Items that were not found in cache.
        uncached_indices (list[int]): Original indices of uncached items (1:1 with uncached_items).
        cached_results (dict[int, Any]): Mapping from original index to cached result item.
        cache_keys (list[str]): Cache keys for uncached items (1:1 with uncached_items).
    """

    uncached_items: list[Any] = field(default_factory=list)
    uncached_indices: list[int] = field(default_factory=list)
    cached_results: dict[int, Any] = field(default_factory=dict)
    cache_keys: list[str] = field(default_factory=list)

    @property
    def all_cached(self) -> bool:
        """True when every item was found in cache."""
        return len(self.uncached_items) == 0


def cache_strip_items(
    tool_name: str,
    items: list[Any],
    config: Any,
) -> CacheStripResult | None:
    """Look up each item in the active cache, returning cached vs uncached split.

    Args:
        tool_name (str): Registry key of the tool.
        items (list[Any]): List of input items (already deduped).
        config (Any): Tool config (included in per-item cache key).

    Returns:
        CacheStripResult | None: A ``CacheStripResult``, or ``None`` if no active cache exists.
    """
    cache = _program_tool_cache.get()
    if cache is None:
        return None

    result = CacheStripResult()

    for idx, item in enumerate(items):
        cache_key = _generate_cache_key(tool_name, input_item=item, config=config)
        cached = cache.get(tool_name, cache_key)
        if cached is not None:
            result.cached_results[idx] = cached
        else:
            result.uncached_items.append(item)
            result.uncached_indices.append(idx)
            result.cache_keys.append(cache_key)

    num_hits = len(result.cached_results)
    total = len(items)
    logger.debug(
        "[Iterable Cache Stats] %s: %d cache hits, %d misses out of %d items",
        tool_name,
        num_hits,
        total - num_hits,
        total,
    )

    return result


def cache_store_items(
    tool_name: str,
    cache_keys: list[str],
    result_items: list[Any],
) -> None:
    """Write newly computed items into the active cache.

    Args:
        tool_name (str): Registry key of the tool.
        cache_keys (list[str]): Cache keys (1:1 with result_items).
        result_items (list[Any]): Computed result items to store.
    """
    cache = _program_tool_cache.get()
    if cache is None:
        return

    for key, item in zip(cache_keys, result_items, strict=False):
        cache.set(tool_name, key, item)


def cache_stitch_items(
    strip: CacheStripResult,
    computed_items: list[Any],
    total_count: int,
) -> list[Any]:
    """Merge cached and freshly computed items back into original order.

    Args:
        strip (CacheStripResult): The ``CacheStripResult`` from ``cache_strip_items``.
        computed_items (list[Any]): Newly computed items (1:1 with ``strip.uncached_indices``).
        total_count (int): Total number of items in the original (pre-strip) input.

    Returns:
        list[Any]: Merged list of length ``total_count`` with items in original order.
    """
    result_map = {
        **dict(strip.cached_results),
        **dict(zip(strip.uncached_indices, computed_items, strict=False)),
    }
    return [result_map[i] for i in range(total_count)]


# ============================================================================
# Module-level cache management functions
# ============================================================================


def clear_cache() -> None:
    """Clear all cached results from the program-scoped cache.

    Gets the cache from contextvar and clears it.
    """
    cache = _program_tool_cache.get()
    if cache:
        cache.clear()


def clear_tool_cache(tool_name: str) -> int:
    """Clear cache entries for a specific tool.

    Args:
        tool_name (str): Name of the tool to clear cache for

    Returns:
        int: Number of entries cleared
    """
    cache = _program_tool_cache.get()
    if cache:
        return cache.clear(tool_name)
    return 0


def get_cache_info() -> dict[str, Any]:
    """Get information about the cache.

    Returns:
        dict[str, Any]: Dictionary with cache statistics
    """
    cache = _program_tool_cache.get()
    if cache:
        return cache.get_info()
    return {"total_entries": 0, "cache_size_bytes": 0}


def has_cached_entries(tool_name: str) -> bool:
    """Check if a specific tool has any cached entries.

    Args:
        tool_name (str): Name of the tool to check

    Returns:
        bool: True if the tool has cached entries, False otherwise
    """
    cache = _program_tool_cache.get()
    if cache is None:
        return False

    tool_cache = cache._cache.get(tool_name)
    return tool_cache is not None and len(tool_cache) > 0
