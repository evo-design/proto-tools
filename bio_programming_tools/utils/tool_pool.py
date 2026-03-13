"""
tool_pool.py

Parallel fan-out across multiple devices for list-input tools.

ToolPool intercepts @tool-decorated function calls, partitions input items
across persistent workers on different devices using cost-aware LPT scheduling,
runs them concurrently via ThreadPoolExecutor, and reassembles results in
original order.

Cloud dispatch uses ``ToolRegistry._execution_backend`` — the backend is
registered by bio-programming at startup. ToolPool never imports
deployment code directly, keeping bio-programming-tools standalone-safe.
"""
from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from bio_programming_tools.utils.device import (
    determine_visible_devices,
    number_of_available_gpus,
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


def _build_dispatch_stats(
    total_items: int,
    local_items: int,
    cloud_items: int,
    local_devices: int,
    batch_dispatch: bool = False,
) -> dict:
    """Build dispatch stats dict for output metadata."""
    return {
        "total_items": total_items,
        "local_items": local_items,
        "cloud_items": cloud_items,
        "local_devices": local_devices,
        "batch_dispatch": batch_dispatch,
    }


# ============================================================================
# ToolPool
# ============================================================================

class ToolPool:
    """Parallel fan-out across devices for list-input tools.

    Usage::

        # Auto-detect all visible GPUs (default, local only)
        with ToolPool():
            result = run_boltz2(inputs, config)

        # Explicit local GPUs
        with ToolPool(devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]):
            result = run_esmfold(ESMFoldInput(complexes=all_100), ESMFoldConfig())

        # Hybrid: local GPUs first, overflow to cloud
        with ToolPool(remote=True):
            result = run_esmfold(inputs, config)

        # Cloud only — skip local GPUs entirely
        with ToolPool(remote="cloud"):
            result = run_esmfold(inputs, config)

    The pool intercepts ``@tool``-decorated function calls transparently.
    Only tools that declare ``iterable_input_field`` / ``iterable_output_field``
    on their ``@tool()`` decorator are parallelized; other tools pass through
    to normal single-worker execution (but still benefit from persistence).

    **Remote modes:**

    - ``remote=False`` (default) — local GPUs only. Free compute.
    - ``remote=True`` — hybrid: fill local GPU slots first via LPT,
      overflow items dispatch to cloud. Local is preferred (free GPUs).
    - ``remote="cloud"`` — cloud only: skip local GPUs, fan out all items
      via the registered cloud backend. The backend autoscales workers.

    Cloud dispatch uses ``ToolRegistry._execution_backend``, which is
    registered by bio-programming at startup. No cross-repo imports needed.

    Multi-GPU tools override ``BaseConfig.devices_per_instance`` (a
    ``@property``, not a field) to tell the pool how many GPUs each worker
    needs.  The pool groups its device list into slots of that size — e.g.,
    4 GPUs with ``devices_per_instance == 2`` yields 2 workers on
    ``cuda:0,cuda:1`` and ``cuda:2,cuda:3``.
    """

    def __init__(
        self,
        devices: list[str] | str | None = None,
        remote: bool | Literal["cloud"] = False,
    ):
        """
        Args:
            devices: Device strings for the pool. Accepts a list
                (e.g. ``["cuda:0", "cuda:1"]``), a single string
                (e.g. ``"cuda:0"``), or ``None`` to auto-detect all
                visible GPUs. Ignored when ``remote="cloud"``.
            remote: Cloud dispatch mode.
                - ``False`` (default): local only.
                - ``True``: hybrid — local GPUs + cloud overflow.
                - ``"cloud"``: cloud only, skip local GPUs.
        """
        if isinstance(devices, str):
            devices = [devices]

        if remote not in (False, True, "cloud"):
            raise ValueError(
                f"remote must be False, True, or 'cloud', got {remote!r}"
            )

        self._devices_arg = devices
        self._remote = remote
        self._persist_ctx = None
        self._token = None

    def __enter__(self) -> ToolPool:
        if _active_pool.get() is not None:
            raise RuntimeError("ToolPool contexts cannot be nested")

        from bio_programming_tools.tools.tool_registry import ToolRegistry

        # Validate backend if remote is truthy
        if self._remote and ToolRegistry._execution_backend is None:
            raise RuntimeError(
                "remote=True requires a cloud backend. "
                "Register a cloud backend via ToolRegistry.set_execution_backend() first."
            )

        # Resolve local devices
        if self._remote == "cloud":
            # Cloud-only: no local devices needed
            self._devices: list[str] = []
        elif self._devices_arg is not None:
            self._devices = list(self._devices_arg)
            if self._devices:
                determine_visible_devices(self._devices)
        else:
            n = number_of_available_gpus()
            if n == 0 and not self._remote:
                raise RuntimeError(
                    "ToolPool requires at least one GPU. "
                    "No GPUs detected (check CUDA_VISIBLE_DEVICES or nvidia-smi)."
                )
            self._devices = [f"cuda:{i}" for i in range(n)]

        if self._remote is True and not self._devices:
            logger.warning(
                "ToolPool(remote=True): no local GPUs detected — all items "
                "will dispatch to cloud. Consider using remote='cloud' explicitly."
            )

        logger.info(
            "ToolPool entering with devices: %s, remote=%s",
            self._devices, self._remote,
        )

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
        """
        from bio_programming_tools.tools.tool_registry import ToolRegistry

        # Look up ToolSpec
        spec = ToolRegistry.get(tool_key)
        iterable_input_field = spec.iterable_input_field
        iterable_output_field = spec.iterable_output_field

        # Extract items
        items = list(getattr(inputs, iterable_input_field))
        n_items = len(items)

        # Single-item optimization: skip pool overhead (local only)
        if n_items <= 1 and self._remote != "cloud":
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

        # Determine local vs cloud split
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

        # Split items between local and cloud
        if self._remote == "cloud":
            # Cloud only — all items go to cloud
            local_work_items: list[WorkItem] = []
            cloud_work_items = work_items
        elif self._remote is True:
            # Hybrid — fill local slots, overflow to cloud
            n_local_slots = len(capabilities)
            if n_local_slots == 0:
                # No local GPUs available, all to cloud
                local_work_items = []
                cloud_work_items = work_items
            elif n_items <= n_local_slots:
                # Fewer items than local slots — all local
                local_work_items = work_items
                cloud_work_items = []
            else:
                # Fill local, overflow to cloud
                local_work_items = work_items[:n_local_slots]
                cloud_work_items = work_items[n_local_slots:]
        else:
            # Local only
            local_work_items = work_items
            cloud_work_items = []

        logger.info(
            "ToolPool dispatching %s: %d items (%d local, %d cloud, "
            "devices_per_instance=%d)",
            tool_key, n_items, len(local_work_items), len(cloud_work_items), dpi,
        )

        # Execute local partitions via LPT
        all_indexed: list[tuple[int, Any]] = []
        all_warnings: list[str] = []
        all_errors: list[str] = []
        last_result = None
        output_model = spec.output_model

        if local_work_items and capabilities:
            assignments = lpt_schedule(local_work_items, capabilities)
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

        # Execute cloud items via fan-out (no LPT — cloud backend autoscales)
        if cloud_work_items:
            backend = ToolRegistry._execution_backend
            if backend is None:
                raise RuntimeError(
                    "remote dispatch requested but no cloud backend registered"
                )

            from bio_programming_tools.utils.cloud_dispatch import dispatch_cloud_items

            cloud_indexed, cloud_warnings, cloud_errors, cloud_last = (
                dispatch_cloud_items(
                    backend, tool_key, cloud_work_items,
                    inputs, config,
                    iterable_input_field, iterable_output_field,
                )
            )
            all_indexed.extend(cloud_indexed)
            all_warnings.extend(cloud_warnings)
            all_errors.extend(cloud_errors)
            if cloud_last is not None:
                last_result = cloud_last

        # Reassemble in original order
        all_indexed.sort(key=lambda x: x[0])
        merged_items = [item for _, item in all_indexed]

        metadata = dict(last_result.metadata) if last_result else {}
        metadata["dispatch_stats"] = _build_dispatch_stats(
            n_items, len(local_work_items), len(cloud_work_items), len(capabilities),
            batch_dispatch=ToolRegistry._execution_backend_batch is not None,
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
