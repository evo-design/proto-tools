"""
bio_programming_tools/utils/device_manager.py

Automatically tracks and manages GPU allocation across persistent workers,
with LRU eviction, CPU offloading, and configurable strategies.

Usage::

    # Automatic (default) - DeviceManager handles everything
    with ToolInstance.persist():
        run_esmfold(inputs, config)  # Auto-allocated to cuda:0
        run_evo2_sample(inputs2, config2)  # Auto-allocated to cuda:1 or offloads esmfold

    # Manual device control
    tool = ToolInstance.get("esmfold")
    tool.to("cuda:1")  # Explicitly move to cuda:1

    # Check status
    from bio_programming_tools.utils import DeviceManager
    status = DeviceManager.get_instance().get_device_status()

Configuration
-------------
DeviceManager can be configured via environment variables or programmatically.

**Environment Variables:**

- ``BIO_TOOLS_MANAGED_DEVICES``: Comma-separated list of devices for DeviceManager to allocate.
  Accepts two formats for consistency with CUDA_VISIBLE_DEVICES:

  - Numbers only: ``"0,1,2"`` (same format as CUDA_VISIBLE_DEVICES)
  - With cuda prefix: ``"cuda:0,cuda:1,cuda:2"``

  Both specify **logical** device IDs (after CUDA_VISIBLE_DEVICES filtering).
  Default: auto-detect all visible GPUs.

- ``BIO_TOOLS_OFFLOAD_STRATEGY``: Eviction strategy when GPUs are full (``"cpu"`` or ``"restart"``).
  Default: ``"cpu"`` (keeps models warm in RAM).

- ``BIO_TOOLS_ALLOW_MULTI_DEVICE``: Allow multiple tool instances per GPU (``"true"`` or ``"false"``).
  Default: ``"false"`` (one model per GPU).

**CUDA_VISIBLE_DEVICES vs BIO_TOOLS_MANAGED_DEVICES:**

These environment variables work at different levels:

- ``CUDA_VISIBLE_DEVICES``: OS/CUDA runtime level - controls which **physical** GPUs are visible
  to CUDA and how they're renumbered. Set by user/cluster scheduler before process starts.
  Example: ``CUDA_VISIBLE_DEVICES="2,3,5"`` makes physical GPUs 2,3,5 visible as cuda:0,1,2.

- ``BIO_TOOLS_MANAGED_DEVICES``: Application level - controls which **logical** GPUs (already filtered
  by CUDA_VISIBLE_DEVICES) DeviceManager will use for auto-allocation. Use this to reserve some GPUs
  for manual allocation while letting DeviceManager manage others.

Example: On an 8-GPU machine, reserve GPUs for bio-tools::

    export CUDA_VISIBLE_DEVICES="2,3,4,5"  # Make physical GPUs 2-5 visible as cuda:0-3
    export BIO_TOOLS_MANAGED_DEVICES="0,1"  # DeviceManager only auto-allocates cuda:0,1
    # Physical GPUs 4,5 (logical cuda:2,3) remain visible but won't be auto-allocated

    # Or equivalently with cuda: prefix:
    export BIO_TOOLS_MANAGED_DEVICES="cuda:0,cuda:1"

If ``BIO_TOOLS_MANAGED_DEVICES`` is not set, DeviceManager auto-allocates from all GPUs
visible via ``CUDA_VISIBLE_DEVICES`` (or all system GPUs if CUDA_VISIBLE_DEVICES is unset).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator
from uuid import uuid4

from .device import is_exclusive_process_mode, number_of_visible_gpus

logger = logging.getLogger(__name__)

SUPPORTED_DEVICE_PREFIXES = ("cuda", "cpu")

# ============================================================================
# Configuration
# ============================================================================


class OffloadStrategy(Enum):
    """Strategy for handling device conflicts when all GPUs are occupied."""

    CPU = "cpu"  # Move models to CPU (keeps them warm, uses RAM)
    RESTART = "restart"  # Kill and restart worker (frees all memory)


class AllocationType(Enum):
    """Type of device allocation."""

    PERSISTENT = "persistent"  # Long-lived, evictable via callback
    TRANSIENT = "transient"  # One-shot lease, non-evictable, auto-released


@dataclass
class DeviceAllocation:
    """Tracks device allocation for a tool instance (single or multi-GPU)."""

    tool_name: str
    instance_name: str
    device_ids: list[str]  # Changed from device_id: str for multi-GPU support
    allocated_at: float
    last_used: float
    eviction_callback: Callable[[str], None]  # Called with "cpu" or "shutdown" during eviction
    allocation_type: AllocationType = AllocationType.PERSISTENT

    def age_seconds(self) -> float:
        """Return seconds since last use."""
        return time.time() - self.last_used



# ============================================================================
# DeviceManager Singleton
# ============================================================================


class DeviceManager:
    """Singleton that manages GPU allocation for persistent tool instances.

    Automatically allocates devices when persistent workers are created,
    tracks usage, and evicts least-recently-used tools when devices are full.

    Thread-safe - all public methods are protected by a lock.
    """

    _instance: DeviceManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Private constructor - use get_instance() instead."""
        self._allocations: dict[str, DeviceAllocation] = {}
        self._instance_lock = (
            threading.RLock()
        )  # Reentrant lock to allow eviction callbacks to release devices
        self._device_available = threading.Condition(self._instance_lock)

        # Configuration (defaults, can be overridden via configure() or env vars)
        self._managed_devices: list[str] | None = None  # None = auto-detect all
        self._allow_multiple_per_device: bool = False
        self._offload_strategy: OffloadStrategy = OffloadStrategy.RESTART

        # Apply environment variable configuration
        self._apply_env_overrides()

    @classmethod
    def get_instance(cls) -> DeviceManager:
        """Return the singleton DeviceManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing only)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_device_id(device_str: str) -> str:
        """Normalize device string to cuda:N format.

        Accepts both formats:
        - "0" or "1" (same as CUDA_VISIBLE_DEVICES) -> "cuda:0", "cuda:1"
        - "cuda:0" or "cuda:1" (our format) -> "cuda:0", "cuda:1"

        Args:
            device_str (str): Device string in either format

        Returns:
            str: Normalized device string in "cuda:N" format
        """
        device_str = device_str.strip()

        # Already in cuda:N format
        if device_str.startswith("cuda:"):
            return device_str

        # Plain number format (same as CUDA_VISIBLE_DEVICES) - interpret as logical index
        try:
            idx = int(device_str)
            return f"cuda:{idx}"
        except ValueError:
            # Invalid format - return as-is and let validation catch it
            return device_str

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # BIO_TOOLS_MANAGED_DEVICES
        if "BIO_TOOLS_MANAGED_DEVICES" in os.environ:
            devices_str = os.environ["BIO_TOOLS_MANAGED_DEVICES"]
            # Parse and normalize device IDs (supports both "0,1" and "cuda:0,cuda:1" formats)
            raw_devices = [d.strip() for d in devices_str.split(",")]
            self._managed_devices = [self._normalize_device_id(d) for d in raw_devices]
            logger.debug(
                "DeviceManager: Using managed devices from env: %s",
                self._managed_devices,
            )

        # BIO_TOOLS_OFFLOAD_STRATEGY
        if "BIO_TOOLS_OFFLOAD_STRATEGY" in os.environ:
            strategy_str = os.environ["BIO_TOOLS_OFFLOAD_STRATEGY"].lower()
            try:
                self._offload_strategy = OffloadStrategy(strategy_str)
                logger.debug(
                    "DeviceManager: Using offload strategy from env: %s",
                    self._offload_strategy.value,
                )
            except ValueError:
                logger.warning(
                    "Invalid BIO_TOOLS_OFFLOAD_STRATEGY: %r, using default: %s",
                    strategy_str,
                    self._offload_strategy.value,
                )

        # BIO_TOOLS_ALLOW_MULTI_DEVICE
        if "BIO_TOOLS_ALLOW_MULTI_DEVICE" in os.environ:
            val = os.environ["BIO_TOOLS_ALLOW_MULTI_DEVICE"].lower()
            self._allow_multiple_per_device = val in ("true", "1", "yes")
            logger.debug(
                "DeviceManager: Allow multiple per device from env: %s",
                self._allow_multiple_per_device,
            )

        # Auto-escalate CPU -> RESTART under Exclusive_Process GPU mode
        self._escalate_for_exclusive_process()

    def _escalate_for_exclusive_process(self) -> None:
        """Switch CPU offload to RESTART if any GPU is in Exclusive_Process mode.

        Under Exclusive_Process, an evicted subprocess retains its CUDA context
        and blocks other processes from using the GPU even after offloading to CPU.
        """
        if self._offload_strategy == OffloadStrategy.CPU:
            if is_exclusive_process_mode():
                logger.warning(
                    "GPU compute mode is Exclusive_Process — CPU offload strategy "
                    "is incompatible (evicted subprocess retains CUDA context). "
                    "Auto-switching to RESTART strategy."
                )
                self._offload_strategy = OffloadStrategy.RESTART

    def configure(
        self,
        *,
        managed_devices: list[str] | None = None,
        allow_multiple_per_device: bool | None = None,
        offload_strategy: OffloadStrategy | str | None = None,
    ) -> None:
        """Configure DeviceManager behavior.

        Args:
            managed_devices (list[str] | None): List of logical device IDs for
                DeviceManager to allocate from (e.g., ``["cuda:0", "cuda:1"]``).
                These are the logical device IDs **after** CUDA_VISIBLE_DEVICES
                filtering. None means auto-detect all visible GPUs.
            allow_multiple_per_device (bool | None): Whether to allow multiple tool
                instances on the same device. Default: False (one model per GPU).
            offload_strategy (OffloadStrategy | str | None): Strategy for handling
                device conflicts when all GPUs are full: ``"cpu"`` moves the
                least-recently-used model to CPU, ``"restart"`` shuts down its worker.
        """
        with self._instance_lock:
            if managed_devices is not None:
                self._managed_devices = managed_devices
                # Shut down allocations on GPU devices no longer in the pool
                new_pool = set(self._get_available_devices())
                orphaned = [
                    alloc for alloc in list(self._allocations.values())
                    if any(
                        self._is_gpu(d) and d not in new_pool
                        for d in alloc.device_ids
                    )
                ]
                for alloc in orphaned:
                    logger.info(
                        "DeviceManager: Shutting down %s on %s "
                        "(device removed from pool)",
                        alloc.tool_name,
                        self._device_str(alloc.device_ids),
                    )
                    try:
                        alloc.eviction_callback("shutdown")
                    except Exception as e:
                        logger.warning(
                            "DeviceManager: Error shutting down %s: %s",
                            alloc.instance_name, e,
                        )
                    self._allocations.pop(alloc.instance_name, None)
            if allow_multiple_per_device is not None:
                self._allow_multiple_per_device = allow_multiple_per_device
            if offload_strategy is not None:
                if isinstance(offload_strategy, str):
                    offload_strategy = OffloadStrategy(offload_strategy.lower())
                self._offload_strategy = offload_strategy

            # Auto-escalate CPU -> RESTART under Exclusive_Process GPU mode
            self._escalate_for_exclusive_process()

            logger.info(
                "DeviceManager configured: managed_devices=%s, "
                "allow_multiple_per_device=%s, offload_strategy=%s",
                self._managed_devices or "auto",
                self._allow_multiple_per_device,
                self._offload_strategy.value,
            )

    # ------------------------------------------------------------------
    # Device Discovery
    # ------------------------------------------------------------------

    def _get_available_devices(self) -> list[str]:
        """Return list of available device IDs for DeviceManager allocation.

        Resolution order:
        1. If managed_devices is explicitly configured (via configure() or
           BIO_TOOLS_MANAGED_DEVICES env var), use those logical device IDs.
        2. Otherwise, auto-detect all GPUs visible to CUDA.
        3. If CUDA_VISIBLE_DEVICES is set, the auto-detected devices reflect
           the logical GPU numbering after CUDA's physical-to-logical mapping.

        Returns
        -------
        list[str]
            List of logical device IDs (e.g., ["cuda:0", "cuda:1"]) that
            DeviceManager will allocate from.

        Raises
        ------
        ValueError
            If managed_devices specifies device IDs that don't exist in the system.
        """
        # If explicitly configured, validate and use that
        if self._managed_devices is not None:
            # Validate that managed devices actually exist
            num_gpus = number_of_visible_gpus()
            if num_gpus == 0 and self._managed_devices:
                raise ValueError(
                    f"BIO_TOOLS_MANAGED_DEVICES specifies {len(self._managed_devices)} devices "
                    f"({', '.join(self._managed_devices)}), but no GPUs are available"
                )

            # Extract device indices from managed_devices (e.g., "cuda:2" -> 2)
            invalid_devices = []
            for device_id in self._managed_devices:
                if device_id.startswith("cuda:"):
                    try:
                        idx = int(device_id.split(":")[1])
                        if idx >= num_gpus:
                            invalid_devices.append(device_id)
                    except (IndexError, ValueError):
                        invalid_devices.append(device_id)

            if invalid_devices:
                raise ValueError(
                    f"BIO_TOOLS_MANAGED_DEVICES specifies invalid device(s): "
                    f"{', '.join(invalid_devices)}. Only {num_gpus} GPU(s) available "
                    f"(cuda:0 through cuda:{num_gpus-1})"
                )

            return list(self._managed_devices)

        # Auto-detect GPUs
        num_gpus = number_of_visible_gpus()
        if num_gpus == 0:
            return []

        # Return sequential logical indices based on detected GPU count
        return [f"cuda:{i}" for i in range(num_gpus)]

    # ------------------------------------------------------------------
    # Internal Helpers (deduplication)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_gpu(device_id: str) -> bool:
        """Return True if device_id refers to a CUDA GPU."""
        return device_id.startswith("cuda")

    @staticmethod
    def _validate_device(device: str) -> None:
        """Raise ValueError if device string has an unsupported prefix."""
        if not any(device.startswith(p) for p in SUPPORTED_DEVICE_PREFIXES):
            raise ValueError(
                f"Unsupported device '{device}'. "
                f"Supported prefixes: {', '.join(SUPPORTED_DEVICE_PREFIXES)}"
            )

    @staticmethod
    def _device_str(device_ids: list[str]) -> str:
        """Join device IDs into a comma-separated string."""
        return ",".join(device_ids)

    def _check_existing_allocation(
        self,
        instance_name: str,
        device: str,
    ) -> str | None:
        """Return compatible existing allocation, or None.

        If the instance already has an allocation that is compatible with
        *device*, updates ``last_used`` and returns the device string.
        If the allocation is incompatible, releases it and returns None
        so the caller can proceed with a fresh allocation.

        Args:
            instance_name (str): Unique instance identifier.
            device (str): Requested device string (e.g., ``"cuda"``, ``"cuda:0"``).

        Compatibility rules:

        - ``"cpu"`` — existing must be ``cpu``.
        - ``"cuda"`` (auto single) — any single CUDA device.
        - ``"cudaxN"`` (auto multi) — any N CUDA devices.
        - ``"cuda:0"`` (specific) — exact device match.
        - ``"cuda:0,1"`` / ``"cuda:0,cuda:1"`` (specific multi) — exact
          device list match.
        """
        if instance_name not in self._allocations:
            return None

        existing = self._allocations[instance_name]
        existing_str = self._device_str(existing.device_ids)

        from bio_programming_tools.utils.device import parse_device_string
        spec = parse_device_string(device)

        if spec.devices is not None:
            # Specific device(s) requested — must match exactly.
            compatible = existing.device_ids == spec.devices
        else:
            # General CUDA requested — any GPU allocation with the
            # right device count is compatible.
            compatible = (
                existing_str != "cpu"
                and len(existing.device_ids) == spec.count
            )

        if not compatible:
            logger.info(
                "DeviceManager: Existing allocation for %s (%s) is "
                "incompatible with requested device %s, re-allocating",
                instance_name, existing_str, device,
            )
            # Release without taking the lock (caller already holds it).
            self._allocations.pop(instance_name, None)
            return None

        existing.last_used = time.time()
        return existing_str

    def _create_allocation(
        self,
        tool_name: str,
        instance_name: str,
        device_ids: list[str],
        eviction_callback: Callable[[str], None],
        allocation_type: AllocationType = AllocationType.PERSISTENT,
    ) -> str:
        """Create a DeviceAllocation record and log it.

        Args:
            tool_name (str): Name of the tool being allocated.
            instance_name (str): Unique instance identifier.
            device_ids (list[str]): List of device IDs to allocate (e.g., ``["cuda:0"]``).
            eviction_callback (Callable[[str], None]): Callback invoked when evicting this allocation.
            allocation_type (AllocationType): Type of allocation (persistent or lease).

        Returns the device string (comma-joined device IDs).
        """
        now = time.time()
        self._allocations[instance_name] = DeviceAllocation(
            tool_name=tool_name,
            instance_name=instance_name,
            device_ids=device_ids,
            allocated_at=now,
            last_used=now,
            eviction_callback=eviction_callback,
            allocation_type=allocation_type,
        )

        device_str = self._device_str(device_ids)
        logger.info(
            "DeviceManager: Allocated %s [%s] for %s (instance: %s, type: %s)",
            "device" if len(device_ids) == 1 else f"{len(device_ids)} device(s)",
            device_str, tool_name, instance_name, allocation_type.value,
        )
        return device_str

    # ------------------------------------------------------------------
    # Device Query Helpers
    # ------------------------------------------------------------------

    def _get_device_allocations(self, device_id: str) -> list[DeviceAllocation]:
        """Return all allocations using a specific device.

        Searches through all current allocations and returns those that include
        the specified device ID in their device_ids list. This handles both
        single-GPU allocations (where device_ids has one element) and multi-GPU
        allocations (where device_ids has multiple elements).

        Args:
            device_id (str): Device ID to search for (e.g., ``"cuda:0"``, ``"cpu"``).

        Returns:
            list[DeviceAllocation]: List of allocations currently using this device.

        Note:
            When allow_multiple_per_device=False (default), the returned list will
            have at most one element per GPU device.
        """
        return [
            alloc
            for alloc in self._allocations.values()
            if device_id in alloc.device_ids
        ]

    def _get_all_allocated_devices(self) -> set[str]:
        """Return set of all currently allocated device IDs.

        Returns
        -------
        set[str]
            Set of device IDs currently allocated (e.g., {"cuda:0", "cuda:1"})
        """
        allocated = set()
        for alloc in self._allocations.values():
            allocated.update(alloc.device_ids)
        return allocated

    def _find_n_free_devices(self, n: int) -> list[str]:
        """Find N free GPU devices.

        When allow_multiple_per_device=True, prefers free devices but allows
        reusing allocated devices if needed. When False, only returns unallocated devices.

        Args:
            n (int): Number of devices needed.

        Returns:
            list[str]: List of free device IDs (up to N), sorted by device index.
        """
        available = self._get_available_devices()
        all_allocated = self._get_all_allocated_devices()

        # Find free devices first
        free_devices = [
            dev for dev in available
            if self._is_gpu(dev) and dev not in all_allocated
        ]

        if self._allow_multiple_per_device and len(free_devices) < n:
            # Reuse allocated devices, prioritizing fewest allocations first
            allocated_devices = [
                dev for dev in available
                if self._is_gpu(dev) and dev in all_allocated
            ]
            allocated_devices.sort(key=lambda dev: len(self._get_device_allocations(dev)))
            result = free_devices + allocated_devices
            return result[:n]

        # Either we have enough free devices, or multiple-per-device is disabled
        return sorted(free_devices[:n])

    # ------------------------------------------------------------------
    # Device Eviction (LRU)
    # ------------------------------------------------------------------

    def _ensure_n_free_devices(
        self, n: int, exclude_instance: str | None = None
    ) -> list[str]:
        """Evict LRU allocations to free N devices.

        Strategy:
        1. Start with currently free devices
        2. Sort all allocations by last_used (LRU first)
        3. Evict allocations until we have N free GPU slots
        4. Raise if impossible to free N devices

        Args:
            n (int): Number of devices needed.
            exclude_instance (str | None): Instance to exclude from eviction (for moves).

        Returns:
            list[str]: List of N device IDs (sorted).

        Raises:
            RuntimeError: If cannot free N devices after eviction.
        """
        freed_devices = set(self._find_n_free_devices(n))

        if len(freed_devices) >= n:
            return sorted(list(freed_devices))[:n]

        # Need to evict - sort PERSISTENT allocations by LRU
        # TRANSIENT allocations (one-shot leases) are never evicted
        evictable_allocs = [
            alloc for alloc in self._allocations.values()
            if alloc.allocation_type == AllocationType.PERSISTENT
            and any(self._is_gpu(d) for d in alloc.device_ids)
            and (exclude_instance is None or alloc.instance_name != exclude_instance)
        ]

        sorted_allocs = sorted(evictable_allocs, key=lambda a: a.last_used)

        for alloc in sorted_allocs:
            if len(freed_devices) >= n:
                break

            # Add GPU devices from this allocation to freed pool
            for dev_id in alloc.device_ids:
                if self._is_gpu(dev_id):
                    freed_devices.add(dev_id)

            self._evict_allocation(alloc)

        if len(freed_devices) < n:
            raise RuntimeError(
                f"Cannot allocate {n} GPUs: only {len(freed_devices)} available "
                f"after LRU eviction. Available: {sorted(freed_devices)}"
            )

        return sorted(list(freed_devices))[:n]

    def _evict_allocation(self, allocation: DeviceAllocation) -> None:
        """Evict a specific allocation using the configured strategy.

        Calls the allocation's eviction_callback to actually move the model
        or shutdown the worker, then updates bookkeeping.

        Args:
            allocation (DeviceAllocation): The allocation to evict.

        Raises:
            Exception: If CPU strategy callback fails (fatal - GPU remains occupied).
        """
        device_str = self._device_str(allocation.device_ids)
        if self._offload_strategy == OffloadStrategy.CPU:
            logger.info(
                "DeviceManager: Moving %s from %s to CPU (LRU eviction)",
                allocation.tool_name,
                device_str,
            )
            # Call eviction callback to move model to CPU
            # If this fails, we want to know - the GPU is still occupied
            allocation.eviction_callback("cpu")

            # Update allocation record - model is now on CPU
            allocation.device_ids = ["cpu"]
            # DON'T update last_used - preserve LRU ordering for next eviction
        else:  # RESTART
            logger.info(
                "DeviceManager: Restarting %s on %s (LRU eviction, strategy: RESTART)",
                allocation.tool_name,
                device_str,
            )
            # Call eviction callback to shutdown worker
            try:
                allocation.eviction_callback("shutdown")
            except Exception as e:
                # Log but don't raise - worker might already be dead, removal is idempotent
                logger.warning(
                    "DeviceManager: Error calling shutdown callback for %s: %s",
                    allocation.instance_name,
                    e,
                )

            # Remove allocation regardless of callback success
            self._allocations.pop(allocation.instance_name, None)

    def _resolve_device_conflicts(
        self,
        requested_devices: list[str],
        exclude_instance: str | None = None,
    ) -> None:
        """Evict conflicting allocations on requested devices if needed.

        No-op when allow_multiple_per_device is True or there are no conflicts.

        Args:
            requested_devices (list[str]): Device IDs to claim (e.g., ``["cuda:0", "cuda:1"]``).
            exclude_instance (str | None): Instance to skip during conflict detection (for moves).

        Raises:
            RuntimeError: If conflicting devices are held by unevictable allocations.
        """
        if self._allow_multiple_per_device:
            return

        all_allocated = self._get_all_allocated_devices()
        if exclude_instance:
            excluded_devs = set()
            alloc = self._allocations.get(exclude_instance)
            if alloc:
                excluded_devs = set(alloc.device_ids)
            all_allocated -= excluded_devs

        conflicts = [dev for dev in requested_devices if dev in all_allocated]
        if not conflicts:
            return

        evictable_conflicts = [
            alloc
            for alloc in self._allocations.values()
            if alloc.instance_name != exclude_instance
            and any(dev in alloc.device_ids for dev in conflicts)
            and alloc.allocation_type == AllocationType.PERSISTENT
        ]

        evictable_devices = {
            dev for alloc in evictable_conflicts for dev in alloc.device_ids
        }
        unevictable = [dev for dev in conflicts if dev not in evictable_devices]
        if unevictable:
            raise RuntimeError(
                f"Devices {unevictable} held by transient allocations"
            )

        # Warn if free alternatives exist
        available = self._get_available_devices()
        free_alternatives = [
            d for d in available
            if self._is_gpu(d) and d not in all_allocated
        ]
        if free_alternatives:
            names = [a.instance_name for a in evictable_conflicts]
            logger.warning(
                "Requested devices %s are busy with %s, but %s are free. "
                "Honoring explicit device request and evicting LRU.",
                requested_devices, names, free_alternatives,
            )

        for alloc in sorted(evictable_conflicts, key=lambda a: a.last_used):
            self._evict_allocation(alloc)

    # ------------------------------------------------------------------
    # Device Allocation (Internal)
    # ------------------------------------------------------------------

    def _allocate_n_devices(
        self,
        tool_name: str,
        instance_name: str,
        n: int,
        eviction_callback: Callable[[str], None],
        allocation_type: AllocationType = AllocationType.PERSISTENT,
    ) -> str:
        """Allocate N GPU devices with LRU eviction if needed.

        Args:
            tool_name (str): Tool requesting devices.
            instance_name (str): Unique instance identifier.
            n (int): Number of devices to allocate.
            eviction_callback (Callable[[str], None]): Callback for eviction
                (receives ``"cpu"`` or ``"shutdown"``).
            allocation_type (AllocationType): Type of allocation (PERSISTENT or TRANSIENT).

        Returns:
            str: Device string (e.g., ``"cuda:0,cuda:1"``).
        """
        # Try to find N free devices
        available = self._find_n_free_devices(n)

        if len(available) < n:
            logger.info(
                "Insufficient free GPUs (%d available, %d needed). Evicting LRU...",
                len(available), n
            )
            available = self._ensure_n_free_devices(n)

        return self._create_allocation(
            tool_name, instance_name, available, eviction_callback, allocation_type,
        )

    def _allocate_specific_devices(
        self,
        tool_name: str,
        instance_name: str,
        requested_devices: list[str],
        eviction_callback: Callable[[str], None],
        allocation_type: AllocationType = AllocationType.PERSISTENT,
    ) -> str:
        """Allocate specific requested devices.

        Args:
            tool_name (str): Tool requesting devices.
            instance_name (str): Unique instance identifier.
            requested_devices (list[str]): List of device IDs (e.g., ``["cuda:0", "cuda:1"]``).
            eviction_callback (Callable[[str], None]): Callback for eviction
                (receives ``"cpu"`` or ``"shutdown"``).
            allocation_type (AllocationType): Type of allocation (PERSISTENT or TRANSIENT).

        Returns:
            str: Device string (e.g., ``"cuda:0,cuda:1"``).

        Raises:
            RuntimeError: If devices unavailable and multiple-per-device not allowed.
        """
        # Validate devices are in managed pool
        available = self._get_available_devices()
        invalid = [dev for dev in requested_devices if dev not in available]
        if invalid:
            raise RuntimeError(
                f"Devices {invalid} not in managed pool {available}"
            )

        self._resolve_device_conflicts(requested_devices)

        return self._create_allocation(
            tool_name, instance_name, requested_devices, eviction_callback,
            allocation_type,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_device(
        self,
        tool_name: str,
        instance_name: str,
        device: str,
        eviction_callback: Callable[[str], None] | None = None,
    ) -> str:
        """Request device(s) for a tool instance.

        Supports single and multi-GPU allocation via device string parsing.
        Allocates devices according to the configured strategy. If all devices
        are occupied, evicts the least recently used tool (LRU) using the
        configured offload strategy.

        Args:
            tool_name (str): Tool name (e.g., ``"esmfold"``, ``"boltz2"``).
            instance_name (str): Unique instance identifier (cache key).
            device (str): Device string (e.g., ``"cpu"``, ``"cuda"``, ``"cudax2"``, ``"cuda:0,1"``).
            eviction_callback (Callable[[str], None] | None): Callback invoked when this
                allocation is evicted. Receives ``"cpu"`` or ``"shutdown"``. Required for
                GPU allocations.

        Returns:
            str: Allocated device string (e.g., ``"cuda:0"``).

        Raises:
            ValueError: If eviction_callback is None for a GPU allocation.
        """
        with self._instance_lock:
            existing = self._check_existing_allocation(
                instance_name, device=device,
            )
            if existing is not None:
                return existing
            self._validate_device(device)

            # Parse device string
            from bio_programming_tools.utils.device import parse_device_string
            spec = parse_device_string(device)

            # Validate callback — always required since a CPU allocation
            # may later be moved to GPU and need eviction support.
            if eviction_callback is None:
                raise ValueError(
                    f"eviction_callback is required for device allocation "
                    f"(tool: {tool_name}, instance: {instance_name}, device: {device})"
                )

            # Handle CPU allocation
            if spec.devices and spec.devices[0] == "cpu":
                return self._create_allocation(
                    tool_name, instance_name, ["cpu"], eviction_callback,
                )

            # Check if any GPUs are available in the system
            all_devices = self._get_available_devices()
            gpu_devices = [d for d in all_devices if self._is_gpu(d)]
            if not gpu_devices:
                raise RuntimeError(
                    f"No GPUs available for {tool_name} (requested device: {device}). "
                    f"Set device='cpu' in the tool config to run on CPU."
                )

            # Explicit device(s) requested
            if spec.devices:
                return self._allocate_specific_devices(
                    tool_name, instance_name, spec.devices, eviction_callback
                )

            # Auto-allocate N GPUs
            return self._allocate_n_devices(
                tool_name, instance_name, spec.count, eviction_callback
            )

    def release_device(self, instance_name: str) -> None:
        """Release device(s) allocated to a tool instance.

        Args:
            instance_name (str): Unique instance identifier (cache key).
        """
        with self._instance_lock:
            allocation = self._allocations.pop(instance_name, None)
            if allocation is not None:
                logger.info(
                    "DeviceManager: Released %s from %s (instance: %s)",
                    self._device_str(allocation.device_ids),
                    allocation.tool_name,
                    instance_name,
                )
                self._device_available.notify_all()

    # ------------------------------------------------------------------
    # Lease API (transient one-shot allocations)
    # ------------------------------------------------------------------

    @contextmanager
    def lease(
        self,
        tool_name: str,
        device: str,
        timeout: float = 300.0,
    ) -> Generator[str, None, None]:
        """Acquire a transient device lease for a one-shot tool call.

        Context manager that allocates a GPU for the duration of the block,
        then automatically releases it on exit (including exceptions).

        Transient leases differ from persistent allocations:
        - They are **never evicted** — persistent allocations are evicted first
        - They **wait** when all GPUs are held by other transient leases
        - They are **auto-released** on context exit

        Non-GPU devices (cpu, etc.) yield immediately with no tracking.

        Args:
            tool_name (str): Tool name (for logging and allocation tracking).
            device (str): Device string (e.g., ``"cpu"``, ``"cuda"``, ``"cuda:0"``).
            timeout (float): Maximum seconds to wait for a GPU to become available.

        Yields
        ------
        str
            Resolved device string (e.g., "cuda:0").

        Raises
        ------
        TimeoutError
            If no GPU becomes available within timeout.
        RuntimeError
            If no GPUs exist in the system.
        """
        self._validate_device(device)

        # Non-GPU devices: passthrough, no tracking
        if not self._is_gpu(device):
            yield device
            return

        lease_id = f"_lease_{tool_name}_{uuid4().hex[:8]}"
        allocated_device = self._acquire_lease(
            tool_name, lease_id, device, timeout
        )
        try:
            yield allocated_device
        finally:
            self.release_device(lease_id)

    def _acquire_lease(
        self,
        tool_name: str,
        lease_id: str,
        device: str,
        timeout: float,
    ) -> str:
        """Acquire a GPU lease, waiting if all GPUs are held by transient leases.

        Reuses the existing allocation infrastructure (_allocate_n_devices /
        _allocate_specific_devices) which already handles LRU eviction of
        PERSISTENT allocations. Only waits when all remaining GPU allocations
        are TRANSIENT (non-evictable).

        Args:
            tool_name (str): Tool name for allocation record.
            lease_id (str): Unique lease identifier.
            device (str): Device string (e.g., ``"cuda"``, ``"cuda:0"``).
            timeout (float): Maximum seconds to wait.

        Returns:
            str: Resolved device string (e.g., ``"cuda:0"``).
        """
        from bio_programming_tools.utils.device import parse_device_string
        spec = parse_device_string(device)

        # Fail fast if no GPUs exist at all
        all_devices = self._get_available_devices()
        gpu_devices = [d for d in all_devices if self._is_gpu(d)]
        if not gpu_devices:
            raise RuntimeError(
                f"No GPUs available for {tool_name} (requested device: {device}). "
                f"Set device='cpu' in the tool config to run on CPU."
            )

        noop_callback = lambda action: None
        deadline = time.monotonic() + timeout

        with self._device_available:
            while True:
                try:
                    if spec.devices:
                        result = self._allocate_specific_devices(
                            tool_name, lease_id, spec.devices, noop_callback,
                            AllocationType.TRANSIENT,
                        )
                    else:
                        result = self._allocate_n_devices(
                            tool_name, lease_id, spec.count, noop_callback,
                            AllocationType.TRANSIENT,
                        )
                    return result

                except RuntimeError:
                    # All GPUs occupied by transient leases — wait for release
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out waiting for GPU for {tool_name} "
                            f"(waited {timeout:.1f}s, device={device})"
                        ) from None

                    logger.debug(
                        "DeviceManager: Lease %s waiting for GPU (%.1fs remaining)",
                        lease_id, remaining,
                    )
                    self._device_available.wait(timeout=remaining)

    def update_last_used(self, instance_name: str) -> None:
        """Update the last-used timestamp for an allocation.

        Called on every tool dispatch to track recency for LRU eviction.

        Args:
            instance_name (str): Unique instance identifier (cache key).
        """
        with self._instance_lock:
            allocation = self._allocations.get(instance_name)
            if allocation is not None:
                allocation.last_used = time.time()

    def get_device_status(self) -> dict[str, Any]:
        """Return current device allocation status with GPU memory information.

        Returns
        -------
        dict
            Status dictionary with keys:
            - available_devices: List of all managed device IDs
            - allocations: Dict mapping instance_name -> allocation info
            - strategy: Current offload strategy
            - allow_multiple: Whether multiple instances per device is allowed
            - gpu_memory: List of dicts with memory info for each GPU (empty if no GPUs)
        """
        with self._instance_lock:
            allocations_info = {}
            for name, alloc in self._allocations.items():
                allocations_info[name] = {
                    "tool_name": alloc.tool_name,
                    "device_id": self._device_str(alloc.device_ids),  # String for backward compat
                    "device_ids": alloc.device_ids,  # List for multi-GPU
                    "allocated_at": alloc.allocated_at,
                    "last_used": alloc.last_used,
                    "age_seconds": alloc.age_seconds(),
                    "allocation_type": alloc.allocation_type.value,
                }

            # Get GPU memory information
            from bio_programming_tools.utils.device import get_gpu_memory_info
            raw_gpu_info = get_gpu_memory_info()

            # Convert to GB and add device_id
            gpu_memory = []
            for gpu in raw_gpu_info:
                gpu_memory.append({
                    "device_id": f"cuda:{gpu['index']}",
                    "name": gpu['name'],
                    "total_gb": round(gpu['total_bytes'] / 1e9, 1),
                    "used_gb": round(gpu['used_bytes'] / 1e9, 1),
                    "free_gb": round(gpu['free_bytes'] / 1e9, 1),
                    "utilization_percent": round(
                        (gpu['used_bytes'] / gpu['total_bytes'] * 100) if gpu['total_bytes'] > 0 else 0,
                        1
                    )
                })

            return {
                "available_devices": self._get_available_devices(),
                "allocations": allocations_info,
                "offload_strategy": self._offload_strategy.value,
                "allow_multiple_per_device": self._allow_multiple_per_device,
                "gpu_memory": gpu_memory,
            }

    def get_gpu_memory_used(self, device: str) -> int:
        """Get TOTAL GPU memory used in bytes for a logical device (all processes).

        IMPORTANT: This returns the total memory used on the GPU across ALL processes
        and contexts, not just memory for a specific instance. Multiple tool instances
        or external processes may be sharing the GPU.

        This method handles logical device IDs (e.g., "cuda:0") and maps them to
        physical GPU indices, respecting CUDA_VISIBLE_DEVICES settings.

        Args:
            device (str): Logical device ID (e.g., ``"cuda:0"``). Only CUDA devices supported.

        Returns:
            int: Total memory used in bytes across all processes on this GPU, or 0 if
                device is not a CUDA device or query fails.

        Examples:
        >>> # With CUDA_VISIBLE_DEVICES="3,5,7"
        >>> dm = DeviceManager.get_instance()
        >>> mem = dm.get_gpu_memory_used("cuda:0")  # Queries physical GPU 3
        >>> print(f"cuda:0 TOTAL memory: {mem / 1e9:.2f} GB")
        cuda:0 TOTAL memory: 1.23 GB  # Includes all processes

        >>> # If multiple instances share cuda:0, this returns combined memory
        >>> dm.request_device("esm2", "inst1", eviction_callback=lambda x: None)
        >>> dm.request_device("esm2", "inst2", eviction_callback=lambda x: None)
        >>> mem = dm.get_gpu_memory_used("cuda:0")
        >>> # mem includes memory from both inst1 AND inst2 (and any other processes)

        Notes
        -----
        - Returns TOTAL GPU memory, not per-instance
        - Cannot differentiate memory between processes without tracking PIDs
        - Returns 0 for CPU devices
        - Uses nvidia-smi via get_gpu_memory_used_physical() from device.py
        - Automatically handles CUDA_VISIBLE_DEVICES mapping
        """
        from bio_programming_tools.utils.device import get_gpu_memory_used_physical
        import os

        # Only support CUDA devices
        if not self._is_gpu(device):
            return 0

        # Extract logical device index
        try:
            logical_idx = int(device.split(":")[1])
        except (IndexError, ValueError):
            return 0

        # Map logical to physical device ID
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            # CUDA_VISIBLE_DEVICES is set - map logical to physical
            visible_devices = [d.strip() for d in cuda_visible.split(",")]
            if logical_idx >= len(visible_devices):
                return 0
            try:
                physical_idx = int(visible_devices[logical_idx])
            except ValueError:
                # CUDA_VISIBLE_DEVICES might contain UUIDs
                return 0
        else:
            # No CUDA_VISIBLE_DEVICES - logical == physical
            physical_idx = logical_idx

        return get_gpu_memory_used_physical(physical_idx)

    def get_instance_memory_stats(self, instance_name: str) -> dict[str, Any]:
        """Get per-instance memory statistics from a tool's worker process.

        Queries the ToolInstance for memory usage reported by the framework
        (PyTorch or JAX) running inside the worker subprocess. This provides
        **per-instance** memory usage, unlike ``get_gpu_memory_used()`` which
        reports total GPU memory across all processes.

        Args:
            instance_name (str): Unique instance identifier (cache key). Must be a
                currently allocated instance tracked by DeviceManager.

        Returns:
            dict[str, Any]: Memory statistics dictionary from the worker.

        Examples:
        >>> # Single instance memory
        >>> dm = DeviceManager.get_instance()
        >>> with ToolInstance.persist_tool("esm2", instance_name="esm2_1"):
        ...     result = run_esm2_sample(inputs, config, instance="esm2_1")
        ...     stats = dm.get_instance_memory_stats("esm2_1")
        ...     if stats["available"]:
        ...         print(f"Instance using {stats['allocated_bytes'] / 1e9:.2f} GB")

        >>> # Compare instance vs total GPU memory
        >>> instance_mem = dm.get_instance_memory_stats("esm2_1")
        >>> total_mem = dm.get_gpu_memory_used("cuda:0")
        >>> print(f"Instance: {instance_mem['allocated_bytes'] / 1e9:.2f} GB")
        >>> print(f"Total GPU: {total_mem / 1e9:.2f} GB")

        Notes
        -----
        - Only works with persistent workers (``ToolInstance.persist_tool()``)
        - Calls through to ``ToolInstance.get_memory_stats()``
        - Per-instance memory ≤ total GPU memory (multiple processes may share GPU)
        """
        from bio_programming_tools.utils.tool_instance import ToolInstance

        with self._instance_lock:
            allocation = self._allocations.get(instance_name)
            if allocation is None:
                return {
                    "available": False,
                    "error": f"Instance '{instance_name}' not found in allocations",
                }
            tool_name = allocation.tool_name

        # Query memory stats outside the lock (may be slow)
        try:
            tool_instance = ToolInstance.get(tool_name, instance_name=instance_name)
            return tool_instance.get_memory_stats()
        except Exception as e:
            logger.error("Failed to get memory stats for %s: %s", instance_name, e)
            return {"available": False, "error": str(e)}

    def move_to_device(
        self,
        instance_name: str,
        target_device: str,
        worker_callback=None,
    ) -> str | None:
        """Move a tool instance to a different device.

        Called by ``ToolInstance._to(device)`` for recovery/user-initiated
        moves. Updates allocation record and invokes worker_callback to send
        the ``to_device`` command to the subprocess.

        This is NOT used for eviction moves — eviction is handled entirely
        by ``_evict_allocation()`` (bookkeeping) + the eviction callback
        (worker command), which avoids lock ordering issues.

        Args:
            instance_name (str): Unique instance identifier (cache key).
            target_device (str): Target device ID (e.g., ``"cuda:1"``, ``"cpu"``) or
                generic (``"cuda"``).

        Returns:
            str | None: The resolved device string, or None if the instance is not allocated.
        """
        self._validate_device(target_device)

        with self._instance_lock:
            allocation = self._allocations.get(instance_name)
            if allocation is None:
                logger.warning(
                    "DeviceManager: Cannot move %s to %s - not allocated",
                    instance_name,
                    target_device,
                )
                return None

            # Resolve generic device requests (e.g., "cuda" → "cuda:0")
            if target_device == "cuda":
                try:
                    resolved_devices = self._ensure_n_free_devices(
                        1, exclude_instance=instance_name,
                    )
                except RuntimeError:
                    resolved_devices = ["cpu"]
            elif target_device == "cpu":
                resolved_devices = ["cpu"]
            else:
                # Parse multi-GPU targets (e.g., "cuda:2,3" → ["cuda:2", "cuda:3"])
                from bio_programming_tools.utils.device import parse_device_string
                spec = parse_device_string(target_device)
                resolved_devices = spec.devices if spec.devices else [target_device]
                self._resolve_device_conflicts(
                    resolved_devices, exclude_instance=instance_name,
                )

            resolved_device = self._device_str(resolved_devices)
            old_devices = allocation.device_ids
            old_device_str = self._device_str(old_devices)

            if old_devices == resolved_devices:
                logger.debug(
                    "DeviceManager: %s already on %s, skipping move",
                    instance_name,
                    resolved_device,
                )
                return resolved_device

            # Update allocation
            allocation.device_ids = resolved_devices
            allocation.last_used = time.time()

            logger.info(
                "Moving %s from %s to %s",
                instance_name,
                old_device_str,
                resolved_device,
            )

            # Invoke worker callback to actually move the model
            if worker_callback is not None:
                worker_callback(resolved_device)

            return resolved_device
