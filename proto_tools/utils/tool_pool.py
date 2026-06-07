"""proto_tools/utils/tool_pool.py.

Parallel fan-out across multiple devices for list-input tools.

ToolPool intercepts @tool-decorated function calls, partitions input items
across persistent workers on different devices using cost-aware LPT scheduling,
runs them concurrently via ThreadPoolExecutor, and reassembles results in
original order.
"""

import contextlib
import contextvars
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from proto_tools.utils.device import (
    determine_visible_devices,
    number_of_available_gpus,
)
from proto_tools.utils.tool_instance import ToolInstance, _persist_mode

# Type alias for the ``gpus`` argument on ToolPool.
GpuSpec = int | list[str] | Literal["all"]

# Conservative default; pass ``ToolPool(cpus=N)`` to override.
_DEFAULT_CPU_CAP = 4


def _detect_cpus() -> int:
    """Return the number of CPUs this process can use (always ``>= 1``).

    Prefers ``os.sched_getaffinity(0)`` (Linux/cgroup-aware: returns the
    Slurm/Kubernetes-allocated cores), falling back to ``os.cpu_count()`` on
    platforms where ``sched_getaffinity`` is unavailable (macOS, Windows).
    Both branches floor at ``1`` for symmetry — the kernel won't permit a
    running process to have an empty affinity set, but downstream callers
    (ToolPool budget math, the ``uses_cpu(n)`` test gate) rely on a positive
    value, so the floor keeps the contract honest.
    """
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


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
        failed: list[dict[str, Any]],
    ):
        """Initialize PartialFailureError."""
        super().__init__(message)
        self.succeeded = succeeded
        self.failed = failed


# ============================================================================
# ContextVars for pool state
# ============================================================================

_active_pool: ContextVar["ToolPool | None"] = ContextVar("_active_pool", default=None)
_pool_executing: ContextVar[bool] = ContextVar("_pool_executing", default=False)


def get_active_pool() -> "ToolPool | None":
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


@dataclass
class WorkerSlot:
    """Layout for one parallel worker invocation in a dispatch.

    Captures everything that varies per worker — what to call its cache key,
    what to set ``config.device`` to inside its partition, and any per-worker
    env vars to inject into its subprocess. Both GPU and CPU fan-out paths
    produce a list of WorkerSlots so the rest of ``_parallel_dispatch`` can
    treat them uniformly.

    Attributes:
        device_id (str): Human-readable label used for logging and partial-
            failure messages. ``"cuda:0,cuda:1"`` for GPU groups,
            ``"cpu#0"`` / ``"cpu#1"`` / ... for CPU workers.
        worker_name (str): ``ToolInstance`` cache key. Stable across
            dispatches within the pool so workers stay warm.
        device_override (str): Value to set on ``config.device`` for this
            worker's partition (e.g. ``"cuda:0,cuda:1"`` or ``"cpu"``).
        env_overrides (dict[str, str] | None): Per-worker subprocess env vars
            (e.g. ``OMP_NUM_THREADS`` for CPU workers). Forwarded to
            ``ToolInstance.get(env_overrides=...)`` before the partition runs.
    """

    device_id: str
    worker_name: str
    device_override: str
    env_overrides: dict[str, str] | None = None


def _compute_worker_layout(
    tool_key: str,
    config: Any,
    gpu_devices: list[str],
    cpus_budget: int,
) -> list[WorkerSlot]:
    """Compute the worker layout for one dispatch.

    Returns the list of worker slots to fan out across:

    - **GPU mode** (``gpus_per_instance > 0``): groups ``gpu_devices`` into
      slots of ``gpus_per_instance`` (e.g. 4 GPUs with gpi=2 → 2 slots).
      Trailing devices that don't form a complete group are logged and
      dropped, except when no complete group fits at all (1 GPU with gpi=2)
      in which case all remaining devices form a single slot.
    - **CPU fan-out mode** (``gpus_per_instance == 0`` and the tool has
      explicitly opted in via ``cpus_per_instance == N`` for some positive
      int): produces ``max(1, cpus_budget // cpus_per_instance)`` slots,
      each pinned to ``device="cpu"`` with
      ``OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS`` set to ``cpus_per_instance``
      to prevent oversubscription. PyRosetta is the canonical opt-in.
    - **Short-circuit** (``gpus_per_instance == 0`` and
      ``cpus_per_instance is None`` — the default for any ``BaseConfig``
      subclass that doesn't override): returns an empty list. The caller
      runs a single direct call; the tool stays off the pool's CPU
      scheduler (the default for every CPU tool unless it explicitly opts in).

    Args:
        tool_key (str): Tool registry key, used to derive worker names.
        config (Any): Tool config — read for ``gpus_per_instance`` and
            ``cpus_per_instance``.
        gpu_devices (list[str]): Resolved GPU device strings from the pool.
        cpus_budget (int): Total CPU budget the pool has to spend.

    Returns:
        list[WorkerSlot]: One slot per planned worker; empty list signals
            short-circuit to a single direct call.
    """
    gpi = config.gpus_per_instance

    if gpi > 0:
        slots: list[WorkerSlot] = []
        for i in range(0, len(gpu_devices), gpi):
            group = gpu_devices[i : i + gpi]
            if len(group) == gpi:
                device_str = ",".join(group)
                slots.append(
                    WorkerSlot(
                        device_id=device_str,
                        worker_name=f"{tool_key}-pool-{device_str.replace(':', '_').replace(',', '_')}",
                        device_override=device_str,
                    )
                )
        leftover = len(gpu_devices) - len(slots) * gpi
        if leftover > 0:
            logger.warning(
                "ToolPool: %d device(s) unused (gpus_per_instance=%d, total=%d)",
                leftover,
                gpi,
                len(gpu_devices),
            )
        # Edge case: not enough devices for even one full group → use what we have
        if gpu_devices and not slots:
            device_str = ",".join(gpu_devices)
            slots.append(
                WorkerSlot(
                    device_id=device_str,
                    worker_name=f"{tool_key}-pool-{device_str.replace(':', '_').replace(',', '_')}",
                    device_override=device_str,
                )
            )
        return slots

    # gpi == 0: CPU mode
    cpi = config.cpus_per_instance
    if cpi is None:
        # Short-circuit — tool wants single direct call.
        return []

    n_workers = max(1, cpus_budget // cpi)
    cpu_thread_env = {
        "OMP_NUM_THREADS": str(cpi),
        "MKL_NUM_THREADS": str(cpi),
        "OPENBLAS_NUM_THREADS": str(cpi),
        "NUMEXPR_NUM_THREADS": str(cpi),
    }
    return [
        WorkerSlot(
            device_id=f"cpu#{i}",
            worker_name=f"{tool_key}-pool-cpu-{i}",
            device_override="cpu",
            env_overrides=cpu_thread_env,
        )
        for i in range(n_workers)
    ]


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
            a for a in assignments if a.device.max_item_cost is None or work_item.cost <= a.device.max_item_cost
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
) -> dict[str, Any]:
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

        # Default: all visible GPUs, modest CPU cap (min(_detect_cpus(), 4))
        with ToolPool():
            result = run_boltz2(inputs, config)

        # CPU-only fan-out (e.g. PyRosetta on a CPU node)
        with ToolPool(gpus=0, cpus=16):
            result = run_pyrosetta_relax(inputs, config)

        # Specific GPUs by name
        with ToolPool(gpus=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]):
            result = run_esmfold(ESMFoldInput(complexes=all_100), ESMFoldConfig())

        # Take the first N visible GPUs
        with ToolPool(gpus=2):
            result = run_boltz2(inputs, config)

    The pool intercepts ``@tool``-decorated function calls transparently.
    Only tools that declare ``iterable_input_fields`` / ``iterable_output_field``
    on their ``@tool()`` decorator are parallelized; other tools pass through
    to normal single-worker execution (but still benefit from persistence).

    Multi-GPU tools override ``BaseConfig.gpus_per_instance`` (a
    ``@property``, not a field) to tell the pool how many GPUs each worker
    needs.  The pool groups its device list into slots of that size, e.g.
    4 GPUs with ``gpus_per_instance == 2`` yields 2 workers on
    ``cuda:0,cuda:1`` and ``cuda:2,cuda:3``.

    CPU-bound tools default to ``cpus_per_instance == None`` — the pool
    dispatches a single direct call and ``pool.cpus`` is ignored. Tools
    where per-call work is heavy enough to amortize subprocess startup
    (PyRosetta is the canonical case) opt in by overriding to a positive
    integer; the pool then spawns ``max(1, pool.cpus // cpus_per_instance)``
    worker subprocesses for them.
    """

    def __init__(
        self,
        gpus: GpuSpec = "all",
        cpus: int | None = None,
    ):
        """Initialize a ToolPool with the given GPU and CPU resource budget.

        Args:
            gpus (GpuSpec): GPUs available to the pool. Accepts:

                - ``"all"`` (default): every visible GPU.
                - ``int N``: the first N visible GPUs (errors if N exceeds visible).
                - ``list[str]``: explicit device strings (e.g.
                  ``["cuda:0", "cuda:2"]``); validated against the visible set.
                - ``0``: explicit CPU-only mode; no GPU detection runs.
            cpus (int | None): Total CPU budget for CPU fan-out. ``None`` (default)
                resolves to ``min(_detect_cpus(), 4)`` — a conservative cap that
                won't blow up memory on many-core nodes. Pass an explicit integer
                for full control. Must be ``>= 1``.
        """
        if cpus is not None and cpus < 1:
            raise ValueError(f"cpus must be >= 1, got {cpus}")
        self._gpus_arg: GpuSpec = gpus
        self.cpus: int = cpus if cpus is not None else min(_detect_cpus(), _DEFAULT_CPU_CAP)
        self._persist_ctx: contextlib.AbstractContextManager[None] | None = None
        self._token: contextvars.Token[ToolPool | None] | None = None

    def _resolve_gpus(self) -> list[str]:
        """Resolve ``gpus_arg`` to a concrete list of device strings."""
        gpus = self._gpus_arg
        if isinstance(gpus, list):
            devices = list(gpus)
            if devices:
                determine_visible_devices(devices)  # type: ignore[arg-type]
            return devices
        if gpus == "all":
            n = number_of_available_gpus()
            return [f"cuda:{i}" for i in range(n)]
        if isinstance(gpus, int):
            if gpus < 0:
                raise ValueError(f"gpus must be >= 0, got {gpus}")
            if gpus == 0:
                return []
            visible = number_of_available_gpus()
            if gpus > visible:
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
                raise RuntimeError(
                    f"ToolPool requested gpus={gpus} but only {visible} are visible "
                    f"(CUDA_VISIBLE_DEVICES={cuda_visible})"
                )
            return [f"cuda:{i}" for i in range(gpus)]
        raise TypeError(f"gpus must be 'all', a non-negative int, or a list of device strings; got {gpus!r}")

    def __enter__(self) -> "ToolPool":
        if _active_pool.get() is not None:
            raise RuntimeError("ToolPool contexts cannot be nested")

        # Resolve gpus
        self._gpu_devices: list[str] = self._resolve_gpus()

        if not self._gpu_devices and self.cpus == 1:
            logger.warning(
                "ToolPool entering with no GPUs and cpus=1 — pool will short-circuit "
                "every dispatch to a single direct call (no parallelism). Set gpus= "
                "or cpus= explicitly to enable fan-out."
            )

        logger.info(
            "ToolPool entering: gpus=%s cpus=%d",
            self._gpu_devices,
            self.cpus,
        )

        # Enter persistence context for local workers
        self._persist_ctx = ToolInstance.persist()
        self._persist_ctx.__enter__()

        # Set ourselves as the active pool
        self._token = _active_pool.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        # Clear active pool
        assert self._token is not None
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
        func: Callable[..., Any],
        inputs: Any,
        config: Any,
    ) -> Any:
        """Partition items across devices, execute in parallel, reassemble.

        Called by the @tool wrapper when a pool is active. ``func`` is the
        raw tool function (caching is handled by the @tool wrapper, not
        within each partition).

        Args:
            tool_key (str): Tool registry key.
            func (Callable[..., Any]): The raw tool function to execute.
            inputs (Any): Tool input model containing the iterable field.
            config (Any): Tool configuration.
        """
        from proto_tools.tools.tool_registry import ToolRegistry

        # Look up ToolSpec
        spec = ToolRegistry.get(tool_key)
        # Caller (@tool wrapper) only routes here when both fields are set.
        assert spec.iterable_input_fields is not None
        assert spec.iterable_output_field is not None
        primary_input_field: str = spec.iterable_input_fields[0]
        iterable_output_field: str = spec.iterable_output_field

        # Parallel iterable group (primary + aligned siblings, e.g. complexes + msas):
        # slice every present field together so siblings stay aligned per partition.
        active_iter_fields: list[str] = [f for f in spec.iterable_input_fields if getattr(inputs, f, None) is not None]

        # Extract items (primary field defines the per-item count / cost)
        items = list(getattr(inputs, primary_input_field))
        n_items = len(items)

        # Single-item / empty: no collision risk, skip pool overhead.
        if n_items <= 1:
            token = _pool_executing.set(True)
            try:
                return func(inputs, config)
            finally:
                _pool_executing.reset(token)

        # Build the worker layout. Empty list = tool didn't opt in to fan-out
        # (gpus_per_instance==0 and cpus_per_instance is None) → single direct call.
        slots = _compute_worker_layout(tool_key, config, list(self._gpu_devices), self.cpus)

        if not slots:
            token = _pool_executing.set(True)
            try:
                return func(inputs, config)
            finally:
                _pool_executing.reset(token)

        # Compute costs
        input_cls = type(inputs)
        work_items = [
            WorkItem(original_index=i, item=item, cost=input_cls.item_cost(item)) for i, item in enumerate(items)
        ]

        # Build DeviceCapability list (one per slot — labels carry across to LPT)
        capabilities = [DeviceCapability(device_id=slot.device_id) for slot in slots]
        slot_by_device_id = {slot.device_id: slot for slot in slots}

        logger.info(
            "ToolPool dispatching %s: %d items across %d worker slot(s) (gpus_per_instance=%d, cpus_per_instance=%s)",
            tool_key,
            n_items,
            len(slots),
            config.gpus_per_instance,
            config.cpus_per_instance,
        )

        # Execute local partitions via LPT
        all_indexed: list[tuple[int, Any]] = []
        all_warnings: list[str] = []
        all_errors: list[str] = []
        last_result = None
        output_model = spec.output_model

        if work_items:
            assignments = lpt_schedule(work_items, capabilities)
            active_assignments = [a for a in assignments if a.items]

            def _run_local_partition(assignment: WorkerAssignment) -> tuple[list[tuple[int, Any]], Any]:
                """Run a single partition on a local worker."""
                pool_token = _pool_executing.set(True)
                # A ToolPool is inherently a persist context: partition workers are
                # named per-slot and must be reusable across dispatch calls. Force
                # _persist_mode on for this partition's execution so the
                # named-instance kwarg passed to ``func`` auto-creates under its
                # key (idempotent if the caller already entered ``with pool:``).
                persist_token = _persist_mode.set(True)
                try:
                    slot = slot_by_device_id[assignment.device.device_id]
                    # Pre-create the ToolInstance with this slot's env_overrides so
                    # the worker subprocess inherits OMP/MKL pinning before its
                    # first request. Idempotent: if the slot already created it
                    # earlier, the cached instance is returned and env stays put.
                    if slot.env_overrides is not None:
                        ToolInstance.get(
                            tool_key,
                            instance_name=slot.worker_name,
                            env_overrides=slot.env_overrides,
                        )

                    partition_indices = [wi.original_index for wi in assignment.items]
                    partition_input = inputs.model_copy(
                        update={f: [getattr(inputs, f)[idx] for idx in partition_indices] for f in active_iter_fields}
                    )

                    config_copy = config.model_copy(update={"device": slot.device_override})
                    result = func(partition_input, config_copy, instance=slot.worker_name)

                    output_items = getattr(result, iterable_output_field, [])
                    if len(output_items) != len(assignment.items):
                        input_indices = [wi.original_index for wi in assignment.items]
                        raise RuntimeError(
                            f"ToolPool: {tool_key} on {slot.device_id} returned {len(output_items)} "
                            f"{iterable_output_field} but expected {len(assignment.items)} "
                            f"(input indices {input_indices})"
                        )
                    indexed = [
                        (wi.original_index, item) for wi, item in zip(assignment.items, output_items, strict=False)
                    ]
                    return indexed, result
                finally:
                    _persist_mode.reset(persist_token)
                    _pool_executing.reset(pool_token)

            failed: list[dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=len(active_assignments)) as executor:
                contexts = [contextvars.copy_context() for _ in active_assignments]
                futures = [
                    executor.submit(ctx.run, _run_local_partition, a)
                    for ctx, a in zip(contexts, active_assignments, strict=False)
                ]
                for future, assignment in zip(futures, active_assignments, strict=False):
                    try:
                        indexed, result = future.result()
                    except Exception as e:
                        indices = [wi.original_index for wi in assignment.items]
                        logger.error(
                            "ToolPool partition failed on device %s (%d items, indices %s): %s",
                            assignment.device.device_id,
                            len(assignment.items),
                            indices,
                            e,
                        )
                        failed.append(
                            {
                                "device_id": assignment.device.device_id,
                                "indices": indices,
                                "exception": e,
                            }
                        )
                        continue
                    all_indexed.extend(indexed)
                    all_warnings.extend(getattr(result, "warnings", []))
                    all_errors.extend(getattr(result, "errors", []))
                    last_result = result

            if failed:
                n_failed = sum(len(f["indices"]) for f in failed)
                n_ok = len(all_indexed)
                first_exc = failed[0]["exception"]
                first_msg = str(first_exc).splitlines()[0] if str(first_exc) else "<no message>"
                if len(first_msg) > 200:
                    first_msg = first_msg[:200] + "..."
                raise PartialFailureError(
                    f"ToolPool: {len(failed)}/{len(active_assignments)} partition(s) failed "
                    f"({n_failed} items lost, {n_ok} succeeded); "
                    f"first failure: {type(first_exc).__name__}: {first_msg}",
                    succeeded=list(all_indexed),
                    failed=failed,
                )

        # Reassemble in original order
        all_indexed.sort(key=lambda x: x[0])
        merged_items = [item for _, item in all_indexed]

        metadata = dict(last_result.metadata) if last_result else {}
        metadata["dispatch_stats"] = _build_dispatch_stats(
            n_items,
            len(work_items),
            len(capabilities),
        )

        return output_model.model_construct(
            **{  # type: ignore[arg-type]
                iterable_output_field: merged_items,
                "warnings": all_warnings,
                "errors": all_errors,
                "metadata": metadata,
            }
        )
