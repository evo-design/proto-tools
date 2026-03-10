"""
tool_pool.py

Parallel fan-out across multiple devices for list-input tools.

ToolPool intercepts @tool-decorated function calls, partitions input items
across persistent workers on different devices using cost-aware LPT scheduling,
runs them concurrently via ThreadPoolExecutor, and reassembles results in
original order.
"""
from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from bio_programming_tools.utils.device import (
    determine_visible_devices,
    number_of_available_gpus,
    parse_device_string,
)
from bio_programming_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)

# ============================================================================
# ContextVars for pool state
# ============================================================================

_active_pool: ContextVar[ToolPool | None] = ContextVar("_active_pool", default=None)
_pool_executing: ContextVar[bool] = ContextVar("_pool_executing", default=False)


def get_active_pool() -> ToolPool | None:
    """Return the active ToolPool, or None if no pool is active."""
    return _active_pool.get()


def is_pool_executing() -> bool:
    """Return True if we're inside a pool worker thread (prevents recursion)."""
    return _pool_executing.get()


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class DeviceCapability:
    """Describes a device (or device group) available for scheduling.

    Attributes:
        device_id: Device string, e.g. ``"cuda:0"`` or ``"cuda:0,cuda:1"``
            for multi-GPU worker slots.
        throughput_weight: Relative speed of this device compared to others.
            The scheduler divides a device's accumulated cost by its weight
            to estimate finish time, so a weight of 2.0 means "twice as fast"
            and the device will be assigned roughly twice the work. **Currently
            unused** — all devices default to 1.0 (uniform). Reserved for
            future heterogeneous GPU support (e.g., mixed H100/A100 pools).
        max_item_cost: Maximum item cost this device can handle, or None for
            no limit. Items whose ``item_cost()`` exceeds this cap are routed
            to other devices that can handle them (falls back to least-loaded
            if no device qualifies). **Currently unused** — all devices
            default to None. Reserved for future VRAM-aware scheduling
            (e.g., a 24 GB GPU cannot fold a 5000-residue protein).
    """
    device_id: str
    throughput_weight: float = 1.0
    max_item_cost: float | None = None


@dataclass
class WorkItem:
    """A single item to be scheduled, with its original position for reassembly."""
    original_index: int
    item: Any
    cost: float


@dataclass
class WorkerAssignment:
    """Items assigned to a specific device after scheduling."""
    device: DeviceCapability
    items: list[WorkItem] = field(default_factory=list)
    total_cost: float = 0.0


# ============================================================================
# LPT Scheduling
# ============================================================================

def lpt_schedule(
    items: list[WorkItem],
    devices: list[DeviceCapability],
) -> list[WorkerAssignment]:
    """Cost-aware Longest Processing Time (LPT) bin-packing.

    Sorts items by cost descending, then greedily assigns each to the device
    with the lowest estimated finish time (``total_cost / throughput_weight``).
    Gives a 4/3-approximation to optimal makespan.

    With the current defaults (uniform ``throughput_weight=1.0`` and no
    ``max_item_cost`` caps), this reduces to standard LPT, which itself
    degrades to round-robin when all item costs are equal (the common case
    for tools that don't override ``BaseToolInput.item_cost()``).

    Args:
        items: Work items with cost estimates (from ``item_cost()``).
        devices: Available devices. ``throughput_weight`` and ``max_item_cost``
            are supported by the algorithm but currently unused (all devices
            get weight 1.0 and no cap).

    Returns:
        List of WorkerAssignments, one per device (devices with no items
        are included but have empty item lists).
    """
    assignments = [WorkerAssignment(device=d) for d in devices]

    # Sort items by cost descending (LPT)
    sorted_items = sorted(items, key=lambda w: w.cost, reverse=True)

    for work_item in sorted_items:
        # Filter devices that can handle this item
        eligible = [
            a for a in assignments
            if a.device.max_item_cost is None or work_item.cost <= a.device.max_item_cost
        ]
        if not eligible:
            # No device can handle this item — assign to least-loaded anyway
            eligible = assignments

        # Pick device with lowest estimated finish time
        best = min(eligible, key=lambda a: a.total_cost / a.device.throughput_weight)
        best.items.append(work_item)
        best.total_cost += work_item.cost

    return assignments


# ============================================================================
# ToolPool
# ============================================================================

class ToolPool:
    """Parallel fan-out across devices for list-input tools.

    Usage::

        with ToolPool(devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]):
            result = run_esmfold(ESMFoldInput(complexes=all_100), ESMFoldConfig())
            # 100 complexes partitioned across 4 GPUs, run in parallel

        # Auto-detect all visible GPUs
        with ToolPool():
            result = run_boltz2(inputs, config)

    The pool intercepts ``@tool``-decorated function calls transparently.
    Only tools that declare ``iterable_input_field`` / ``iterable_output_field``
    on their ``@tool()`` decorator are parallelized; other tools pass through
    to normal single-worker execution (but still benefit from persistence).

    Multi-GPU tools override ``BaseConfig.devices_per_instance`` (a
    ``@property``, not a field) to tell the pool how many GPUs each worker
    needs.  The pool groups its device list into slots of that size — e.g.,
    4 GPUs with ``devices_per_instance == 2`` yields 2 workers on
    ``cuda:0,cuda:1`` and ``cuda:2,cuda:3``.  This is on Config (not
    ToolSpec) because the GPU requirement can depend on a runtime config
    value such as the active checkpoint size.
    """

    def __init__(
        self,
        devices: list[str] | str | None = None,
    ):
        """
        Args:
            devices: Device strings for the pool. Accepts a list
                (e.g. ``["cuda:0", "cuda:1"]``), a single string
                (e.g. ``"cuda:0"`` or ``"proto"``), or ``None`` to
                auto-detect all visible GPUs.
        """
        if isinstance(devices, str):
            devices = [devices]
        self._devices_arg = devices
        self._persist_ctx = None
        self._token = None

    def __enter__(self) -> ToolPool:
        if _active_pool.get() is not None:
            raise RuntimeError("ToolPool contexts cannot be nested")

        # Resolve devices
        if self._devices_arg is not None:
            self._devices = list(self._devices_arg)
            # Validate CUDA devices exist (raises ValueError if not)
            determine_visible_devices(self._devices)
        else:
            n = number_of_available_gpus()
            if n == 0:
                raise RuntimeError(
                    "ToolPool requires at least one GPU. "
                    "No GPUs detected (check CUDA_VISIBLE_DEVICES or nvidia-smi)."
                )
            self._devices = [f"cuda:{i}" for i in range(n)]

        logger.info("ToolPool entering with devices: %s", self._devices)

        # Always enter persistence — proto is a dispatch-level concern, not pool-level
        self._persist_ctx = ToolInstance.persist()
        self._persist_ctx.__enter__()

        # Set ourselves as the active pool
        self._token = _active_pool.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clear active pool
        _active_pool.reset(self._token)
        self._token = None

        # Exit persistence context
        if self._persist_ctx is not None:
            self._persist_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._persist_ctx = None

        logger.info("ToolPool exited")
        return False

    def _parallel_dispatch(
        self,
        tool_key: str,
        func: Callable,
        inputs: Any,
        config: Any,
    ) -> Any:
        """Partition items across devices, execute in parallel, reassemble.

        Called by the @tool wrapper when a pool is active. ``func`` is the
        raw tool function (caching is handled by the @tool wrapper, not
        within each partition).
        """
        from bio_programming_tools.tools.tool_registry import ToolRegistry

        # Proto check — any proto device in the pool triggers NotImplementedError
        proto_devices = [d for d in self._devices if parse_device_string(d).kind == "proto"]
        if proto_devices:
            raise NotImplementedError(
                f"Proto (cloud worker) execution is not yet implemented for '{tool_key}'. "
                f"Remove proto devices from ToolPool for local-only parallel execution."
            )

        # Look up ToolSpec
        spec = ToolRegistry.get(tool_key)
        iterable_input_field = spec.iterable_input_field
        iterable_output_field = spec.iterable_output_field

        # Extract items
        items = list(getattr(inputs, iterable_input_field))
        n_items = len(items)

        # Single-item optimization: skip pool overhead
        if n_items <= 1:
            token = _pool_executing.set(True)
            try:
                return func(inputs, config)
            finally:
                _pool_executing.reset(token)

        # Compute costs
        input_cls = type(inputs)
        work_items = [
            WorkItem(original_index=i, item=item, cost=input_cls.item_cost(item))
            for i, item in enumerate(items)
        ]

        # Group devices by devices_per_instance
        dpi = config.devices_per_instance
        device_groups = []
        for i in range(0, len(self._devices), dpi):
            group = self._devices[i:i + dpi]
            if len(group) == dpi:
                device_groups.append(",".join(group))
        leftover = len(self._devices) - len(device_groups) * dpi
        if leftover > 0:
            logger.warning(
                "ToolPool: %d device(s) unused (devices_per_instance=%d, total=%d)",
                leftover, dpi, len(self._devices),
            )
        # If no complete groups possible, use all devices as one group
        if not device_groups:
            device_groups = [",".join(self._devices)]

        # Build DeviceCapability list
        capabilities = [DeviceCapability(device_id=dg) for dg in device_groups]

        # Schedule
        assignments = lpt_schedule(work_items, capabilities)

        # Filter out empty assignments
        active_assignments = [a for a in assignments if a.items]

        logger.info(
            "ToolPool dispatching %s: %d items across %d workers (devices_per_instance=%d)",
            tool_key, n_items, len(active_assignments), dpi,
        )

        # Execute partitions in parallel
        output_model = spec.output_model

        def _run_partition(assignment: WorkerAssignment) -> tuple[list[tuple[int, Any]], Any]:
            """Run a single partition on its assigned device. Returns (indexed_results, raw_output)."""
            token = _pool_executing.set(True)
            try:
                device_id = assignment.device.device_id
                # Sanitize device_id for use as instance name
                worker_name = f"{tool_key}-pool-{device_id.replace(':', '_').replace(',', '_')}"

                # Build partition input
                partition_items = [wi.item for wi in assignment.items]
                partition_input = inputs.model_copy(update={iterable_input_field: partition_items})

                # Copy config with device overridden
                config_copy = config.model_copy(update={"device": device_id})

                # Call the cache-wrapped function with instance routing
                result = func(partition_input, config_copy, instance=worker_name)

                # Extract output items
                output_items = getattr(result, iterable_output_field, [])
                if len(output_items) != len(assignment.items):
                    raise RuntimeError(
                        f"ToolPool: {tool_key} returned {len(output_items)} "
                        f"items but expected {len(assignment.items)} for "
                        f"device {assignment.device.device_id}"
                    )
                indexed = [
                    (wi.original_index, item)
                    for wi, item in zip(assignment.items, output_items)
                ]
                return indexed, result
            finally:
                _pool_executing.reset(token)

        all_indexed: list[tuple[int, Any]] = []
        all_warnings: list[str] = []
        all_errors: list[str] = []
        last_result = None

        # ThreadPoolExecutor threads don't inherit ContextVars, so workers
        # wouldn't see _persist_mode=True and would fall back to one-shot.
        # Create a context copy per thread so each runs with persistence.
        with ThreadPoolExecutor(max_workers=len(active_assignments)) as executor:
            contexts = [contextvars.copy_context() for _ in active_assignments]
            futures = [
                executor.submit(ctx.run, _run_partition, a)
                for ctx, a in zip(contexts, active_assignments)
            ]
            for future, assignment in zip(futures, active_assignments):
                try:
                    indexed, result = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"ToolPool partition failed on device "
                        f"{assignment.device.device_id} "
                        f"({len(assignment.items)} items, indices "
                        f"{[wi.original_index for wi in assignment.items]})"
                    ) from e
                all_indexed.extend(indexed)
                all_warnings.extend(getattr(result, "warnings", []))
                all_errors.extend(getattr(result, "errors", []))
                last_result = result

        # Reassemble in original order
        all_indexed.sort(key=lambda x: x[0])
        merged_items = [item for _, item in all_indexed]

        # Construct merged output
        # NOTE: tool_id, success, execution_time are set by the @tool wrapper
        # after _parallel_dispatch returns — don't duplicate them here.
        merged_output = output_model.model_construct(
            **{
                iterable_output_field: merged_items,
                "warnings": all_warnings,
                "errors": all_errors,
                "metadata": last_result.metadata if last_result else {},
            }
        )

        return merged_output
