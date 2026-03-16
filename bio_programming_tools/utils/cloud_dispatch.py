"""
cloud_dispatch.py — Pluggable cloud backend for ToolPool.

Contains retry logic, fan-out, and batch dispatch for cloud execution.
Imported by tool_pool.py when remote= is truthy. Can be replaced entirely
to swap cloud providers.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from bio_programming_tools.utils.tool_pool import PartialFailureError, WorkItem, _MAX_CLOUD_WORKERS

logger = logging.getLogger(__name__)

# ============================================================================
# Cloud dispatch config
# ============================================================================

# Retry config for transient cloud failures
_CLOUD_MAX_RETRIES = 4
_CLOUD_RETRY_BASE_DELAY = 2.0  # seconds (exponential backoff: 2s, 4s, 8s)
_CLOUD_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError)


# ============================================================================
# Cloud dispatch helpers
# ============================================================================

def _cloud_dispatch_with_retry(
    backend: Callable,
    tool_key: str,
    item_input: Any,
    config: Any,
    max_retries: int = _CLOUD_MAX_RETRIES,
) -> Any:
    """Dispatch a single item to the cloud backend with retry on transient failures.

    Args:
        backend: The registered execution backend callable.
        tool_key: Tool registry key.
        item_input: Input model for this single item (or partition).
        config: Tool config.
        max_retries: Maximum retry attempts for transient failures.

    Returns:
        The backend result.

    Raises:
        RuntimeError: If backend returns None (tool not available for remote execution).
        ConnectionError/TimeoutError: After all retries exhausted.
        Any other exception: Immediately (non-retryable).
    """
    for attempt in range(max_retries):
        try:
            result = backend(tool_key, item_input, config)
            if result is None:
                raise RuntimeError(
                    f"Cloud backend returned None for tool '{tool_key}'. "
                    f"The tool may not be registered for remote execution."
                )
            return result
        except _CLOUD_RETRYABLE_EXCEPTIONS:
            if attempt == max_retries - 1:
                raise
            delay = _CLOUD_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Cloud dispatch %s: transient failure on attempt %d/%d, "
                "retrying in %.1fs",
                tool_key, attempt + 1, max_retries, delay,
            )
            time.sleep(delay)


def _fan_out_single_calls(
    backend: Callable,
    tool_key: str,
    item_inputs: list[Any],
    config: Any,
    work_items: list[WorkItem],
    max_cloud_workers: int = _MAX_CLOUD_WORKERS,
) -> list[Any]:
    """Fan out single-item backend calls via ThreadPoolExecutor.

    Fallback for when no batch backend is registered. Each call is
    individually retried on transient failures.
    """
    n_workers = min(len(work_items), max_cloud_workers)
    results: list[Any] = [None] * len(work_items)

    failed: list[dict] = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {}
        for i, (item_input, wi) in enumerate(zip(item_inputs, work_items)):
            future = executor.submit(
                _cloud_dispatch_with_retry,
                backend, tool_key, item_input, config,
            )
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.error(
                    "Cloud dispatch failed for item index %d of tool '%s': %s",
                    work_items[idx].original_index, tool_key, e,
                )
                failed.append({
                    "device_id": "cloud",
                    "indices": [work_items[idx].original_index],
                    "exception": e,
                })

    if failed:
        succeeded: list[tuple[int, object]] = []
        for i, r in enumerate(results):
            if r is not None:
                succeeded.append((work_items[i].original_index, r))
        n_failed = sum(len(f["indices"]) for f in failed)
        raise PartialFailureError(
            f"Cloud fan-out: {len(failed)} item(s) failed "
            f"({n_failed} items lost, {len(succeeded)} succeeded) "
            f"for tool '{tool_key}'",
            succeeded=succeeded,
            failed=failed,
        )

    return results


# ============================================================================
# Cloud dispatch orchestrator
# ============================================================================

def dispatch_cloud_items(
    backend: Callable,
    tool_key: str,
    work_items: list[WorkItem],
    inputs: Any,
    config: Any,
    iterable_input_field: str,
    iterable_output_field: str,
    max_cloud_workers: int = _MAX_CLOUD_WORKERS,
) -> tuple[list[tuple[int, Any]], list[str], list[str], Any]:
    """Fan out work items to the cloud backend.

    Uses batch backend when available,
    falling back to N × single backend calls via ThreadPoolExecutor.

    Args:
        max_cloud_workers: Cap on concurrent cloud workers. Limits both
            the fan-out ThreadPoolExecutor and batch chunk size.

    Returns:
        (indexed_results, warnings, errors, last_result)
    """
    from bio_programming_tools.tools.tool_registry import ToolRegistry

    # Build single-item inputs for each work item
    item_inputs = [
        inputs.model_copy(update={iterable_input_field: [wi.item]})
        for wi in work_items
    ]

    batch_backend = ToolRegistry._execution_backend_batch

    if batch_backend is not None:
        # Batch dispatch — chunk into groups of max_cloud_workers to cap
        # concurrent cloud workers.
        logger.info(
            "Cloud batch dispatch: %d items for %s via batch backend "
            "(max_cloud_workers=%d)",
            len(work_items), tool_key, max_cloud_workers,
        )
        results = []
        for chunk_start in range(0, len(item_inputs), max_cloud_workers):
            chunk = item_inputs[chunk_start:chunk_start + max_cloud_workers]
            chunk_results = batch_backend(tool_key, chunk, config)
            if len(chunk_results) != len(chunk):
                raise RuntimeError(
                    f"Batch backend returned {len(chunk_results)} results "
                    f"but expected {len(chunk)} for tool '{tool_key}'"
                )
            results.extend(chunk_results)
    else:
        # Fallback: N × single backend calls with client-side retry
        logger.info(
            "Cloud fan-out dispatch: %d items for %s via %d workers",
            len(work_items), tool_key,
            min(len(work_items), max_cloud_workers),
        )
        results = _fan_out_single_calls(
            backend, tool_key, item_inputs, config, work_items,
            max_cloud_workers=max_cloud_workers,
        )

    # Collect indexed results from all responses
    all_indexed: list[tuple[int, Any]] = []
    all_warnings: list[str] = []
    all_errors: list[str] = []
    last_result = None

    for wi, result in zip(work_items, results):
        output_items = getattr(result, iterable_output_field, [])
        if len(output_items) != 1:
            raise RuntimeError(
                f"Cloud dispatch for {tool_key}: expected 1 output "
                f"item, got {len(output_items)}"
            )
        all_indexed.append((wi.original_index, output_items[0]))
        all_warnings.extend(getattr(result, "warnings", []))
        all_errors.extend(getattr(result, "errors", []))
        last_result = result

    return all_indexed, all_warnings, all_errors, last_result
