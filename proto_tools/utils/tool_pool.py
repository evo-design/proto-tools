"""
proto_tools/utils/tool_pool.py

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

from proto_tools.utils.device import (
    determine_visible_devices,
    number_of_available_gpus,
)
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)

# ============================================================================
# Error types
# ============================================================================

class PartialFailureError(RuntimeError):
    """Some worker partitions failed; successful results are preserved.

    Attributes:
        succeeded: List of ``(original_index, output_item)`` tuples from
            partitions that completed successfully.
        failed: List of dicts describing each failure, with keys
            ``device_id``, ``indices`` (list of original item indices),
            and ``exception`` (the original exception object).
    """

    def __init__(
        self,
        message: str,
        succeeded: list[tuple[int, Any]],
        failed: list[dict],
    ):
        super().__init__(message)
        self.succeeded = succeeded
        self.failed = failed

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
        device_id (str): Device string, e.g. ``"cuda:0"`` or ``"cuda:0,cuda:1"``
            for multi-GPU worker slots.
        throughput_weight (float): Relative speed of this device compared to others.
            The scheduler divides a device's accumulated cost by its weight
            to estimate finish time, so a weight of 2.0 means "twice as fast"
            and the device will be assigned roughly twice the work. **Currently
            unused**; all devices default to 1.0 (uniform). Reserved for
            future heterogeneous GPU support (e.g., mixed H100/A100 pools).
        max_item_cost (float | None): Maximum item cost this device can handle, or None for
            no limit. Items whose ``item_cost()`` exceeds this cap are routed
            to other devices that can handle them (falls back to least-loaded
            if no device qualifies). **Currently unused**; all devices
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
        items (list[WorkItem]): Work items with cost estimates (from ``item_cost()``).
        devices (list[DeviceCapability]): Available devices. ``throughput_weight`` and ``max_item_cost``
            are supported by the algorithm but currently unused (all devices
            get weight 1.0 and no cap).

    Returns:
        list[WorkerAssignment]: List of WorkerAssignments, one per device (devices with no items
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
            # No device can handle this item; assign to least-loaded anyway
            eligible = assignments

        # Pick device with lowest estimated finish time
        best = min(eligible, key=lambda a: a.total_cost / a.device.throughput_weight)
        best.items.append(work_item)
        best.total_cost += work_item.cost

    return assignments


def _build_dispatch_stats(
    total_items: int,
    local_items: int,
    local_devices: int,
) -> dict:
    """Build dispatch stats dict for output metadata."""
    return {
        "total_items": total_items,
        "local_items": local_items,
        "local_devices": local_devices,
    }


# ============================================================================
# ToolPool
# ============================================================================

class ToolPool:
    """Parallel fan-out across devices for list-input tools.

    Usage::

        # Auto-detect all visible GPUs (default)
        with ToolPool():
            result = run_boltz2(inputs, config)

        # Explicit local GPUs
        with ToolPool(devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]):
            result = run_esmfold(ESMFoldInput(complexes=all_100), ESMFoldConfig())

    The pool intercepts ``@tool``-decorated function calls transparently.
    Only tools that declare ``iterable_input_field`` / ``iterable_output_field``
    on their ``@tool()`` decorator are parallelized; other tools pass through
    to normal single-worker execution (but still benefit from persistence).

    Multi-GPU tools override ``BaseConfig.devices_per_instance`` (a
    ``@property``, not a field) to tell the pool how many GPUs each worker
    needs.  The pool groups its device list into slots of that size, e.g.
    4 GPUs with ``devices_per_instance == 2`` yields 2 workers on
    ``cuda:0,cuda:1`` and ``cuda:2,cuda:3``.
    """

    def __init__(
        self,
        devices: list[str] | str | None = None,
    ):
        """
        Args:
            devices: Device strings for the pool. Accepts a list
                (e.g. ``["cuda:0", "cuda:1"]``), a single string
                (e.g. ``"cuda:0"``), or ``None`` to auto-detect all
                visible GPUs.
        """
        if isinstance(devices, str):
            devices = [devices]

        self._devices_arg = devices
        self._persist_ctx = None
        self._token = None

    def __enter__(self) -> ToolPool:
        if _active_pool.get() is not None:
            raise RuntimeError("ToolPool contexts cannot be nested")

        # Resolve local devices
        if self._devices_arg is not None:
            self._devices: list[str] = list(self._devices_arg)
            if self._devices:
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

        # Enter persistence context for local workers
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

        Args:
            tool_key (str): Tool registry key.
            func (Callable): The raw tool function to execute.
            inputs (Any): Tool input model containing the iterable field.
            config (Any): Tool configuration.
        """
        from proto_tools.tools.tool_registry import ToolRegistry

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

        local_devices = list(self._devices)

        # Group local devices by devices_per_instance
        dpi = config.devices_per_instance
        device_groups: list[str] = []
        if local_devices:
            for i in range(0, len(local_devices), dpi):
                group = local_devices[i:i + dpi]
                if len(group) == dpi:
                    device_groups.append(",".join(group))
            leftover = len(local_devices) - len(device_groups) * dpi
            if leftover > 0:
                logger.warning(
                    "ToolPool: %d device(s) unused (devices_per_instance=%d, total=%d)",
                    leftover, dpi, len(local_devices),
                )
            # If no complete groups possible, use all local devices as one group
            if local_devices and not device_groups:
                device_groups = [",".join(local_devices)]

        # Build DeviceCapability list for local groups
        capabilities = [DeviceCapability(device_id=dg) for dg in device_groups]

        logger.info(
            "ToolPool dispatching %s: %d items (%d local, "
            "devices_per_instance=%d)",
            tool_key, n_items, len(work_items), dpi,
        )

        # Execute local partitions via LPT
        all_indexed: list[tuple[int, Any]] = []
        all_warnings: list[str] = []
        all_errors: list[str] = []
        last_result = None
        output_model = spec.output_model

        if work_items and capabilities:
            assignments = lpt_schedule(work_items, capabilities)
            active_assignments = [a for a in assignments if a.items]

            def _run_local_partition(assignment: WorkerAssignment) -> tuple[list[tuple[int, Any]], Any]:
                """Run a single partition on a local device."""
                token = _pool_executing.set(True)
                try:
                    device_id = assignment.device.device_id
                    partition_items = [wi.item for wi in assignment.items]
                    partition_input = inputs.model_copy(update={iterable_input_field: partition_items})
                    worker_name = f"{tool_key}-pool-{device_id.replace(':', '_').replace(',', '_')}"
                    config_copy = config.model_copy(update={"device": device_id})
                    result = func(partition_input, config_copy, instance=worker_name)

                    output_items = getattr(result, iterable_output_field, [])
                    if len(output_items) != len(assignment.items):
                        raise RuntimeError(
                            f"ToolPool: {tool_key} returned {len(output_items)} "
                            f"items but expected {len(assignment.items)} for "
                            f"device {device_id}"
                        )
                    indexed = [
                        (wi.original_index, item)
                        for wi, item in zip(assignment.items, output_items)
                    ]
                    return indexed, result
                finally:
                    _pool_executing.reset(token)

            failed: list[dict] = []

            with ThreadPoolExecutor(max_workers=len(active_assignments)) as executor:
                contexts = [contextvars.copy_context() for _ in active_assignments]
                futures = [
                    executor.submit(ctx.run, _run_local_partition, a)
                    for ctx, a in zip(contexts, active_assignments)
                ]
                for future, assignment in zip(futures, active_assignments):
                    try:
                        indexed, result = future.result()
                    except Exception as e:
                        indices = [wi.original_index for wi in assignment.items]
                        logger.error(
                            "ToolPool partition failed on device %s "
                            "(%d items, indices %s): %s",
                            assignment.device.device_id,
                            len(assignment.items), indices, e,
                        )
                        failed.append({
                            "device_id": assignment.device.device_id,
                            "indices": indices,
                            "exception": e,
                        })
                        continue
                    all_indexed.extend(indexed)
                    all_warnings.extend(getattr(result, "warnings", []))
                    all_errors.extend(getattr(result, "errors", []))
                    last_result = result

            if failed:
                n_failed = sum(len(f["indices"]) for f in failed)
                n_ok = len(all_indexed)
                raise PartialFailureError(
                    f"ToolPool: {len(failed)} partition(s) failed "
                    f"({n_failed} items lost, {n_ok} succeeded)",
                    succeeded=list(all_indexed),
                    failed=failed,
                )

        # Reassemble in original order
        all_indexed.sort(key=lambda x: x[0])
        merged_items = [item for _, item in all_indexed]

        metadata = dict(last_result.metadata) if last_result else {}
        metadata["dispatch_stats"] = _build_dispatch_stats(
            n_items, len(work_items), len(capabilities),
        )

        merged_output = output_model.model_construct(
            **{
                iterable_output_field: merged_items,
                "warnings": all_warnings,
                "errors": all_errors,
                "metadata": metadata,
            }
        )

        return merged_output
