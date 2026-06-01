"""proto_tools/utils/device.py.

GPU detection and device visibility.

GPU detection uses nvidia-smi rather than torch.cuda so that the
orchestrator package works with a CPU-only PyTorch install (or no
PyTorch at all).  Actual GPU workloads run inside isolated venvs
that have their own CUDA-enabled PyTorch.
"""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _run_nvidia_smi_query(*args: str) -> str | None:
    """Run nvidia-smi query, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *list(args)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@dataclass(frozen=True)
class DeviceSpec:
    """Structured result from :func:`parse_device_string`.

    Attributes:
        kind (str): ``"cpu"``, ``"cuda"``, or ``"cloud"``.
        devices (list[str] | None): Explicit device IDs when provided (e.g. ``["cuda:0"]``),
            ``None`` for auto-allocate CUDA.
        count (int): Number of CUDA devices requested (always 1 for cpu and cloud).
    """

    kind: str
    devices: list[str] | None
    count: int


def get_gpu_compute_modes() -> list[str]:
    """Return the compute mode for each GPU via nvidia-smi.

    Returns:
        list[str]: List of mode strings per GPU (e.g. ``["Exclusive_Process", "Default"]``).
            Empty list if nvidia-smi is unavailable.
    """
    out = _run_nvidia_smi_query("--query-gpu=compute_mode", "--format=csv,noheader")
    if out is None:
        return []
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def is_exclusive_process_mode() -> bool:
    """Return True if any GPU reports ``Exclusive_Process`` compute mode."""
    return any(m == "Exclusive_Process" for m in get_gpu_compute_modes())


def number_of_physical_gpus() -> int:
    """Returns the number of physical NVIDIA GPUs via nvidia-smi.

    This function always queries nvidia-smi and returns the total count of
    physical GPUs in the system, regardless of CUDA_VISIBLE_DEVICES settings.

    Returns:
        int: Number of physical GPUs detected by nvidia-smi, or 0 if nvidia-smi
             is not available or fails.

    Example:
        >>> number_of_physical_gpus()  # On 8-GPU machine
        8
        >>> # CUDA_VISIBLE_DEVICES doesn't affect this function
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
        >>> number_of_physical_gpus()  # Still returns all physical GPUs
        8
    """
    out = _run_nvidia_smi_query("--query-gpu=count", "--format=csv,noheader")
    if out is not None:
        # nvidia-smi returns one line per GPU, each containing total count;
        # the number of lines equals the number of GPUs.
        return len(out.strip().splitlines())
    return 0


def number_of_visible_gpus() -> int:
    """Returns the number of GPUs visible to CUDA runtime.

    This function respects the CUDA_VISIBLE_DEVICES environment variable.
    If CUDA_VISIBLE_DEVICES is set, it returns the count of devices in that list.
    Otherwise, it returns the total count of physical GPUs (same as number_of_physical_gpus()).

    Also validates that CUDA_VISIBLE_DEVICES doesn't reference non-existent GPUs
    and logs a warning if it does.

    Returns:
        int: Number of GPUs visible to CUDA runtime after CUDA_VISIBLE_DEVICES filtering.

    Example:
        >>> # Without CUDA_VISIBLE_DEVICES
        >>> number_of_visible_gpus()  # Returns all physical GPUs
        8

        >>> # With CUDA_VISIBLE_DEVICES set
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,5"
        >>> number_of_visible_gpus()  # Returns count of visible devices
        3

        >>> # Empty CUDA_VISIBLE_DEVICES means no GPUs visible
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
        >>> number_of_visible_gpus()
        0
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    # If CUDA_VISIBLE_DEVICES not set, all physical GPUs are visible
    if not cuda_visible:
        # Check if empty string was explicitly set (means no GPUs visible)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            return 0
        # Not set at all - all physical GPUs are visible
        return number_of_physical_gpus()

    # Parse CUDA_VISIBLE_DEVICES - it's a comma-separated list of device indices
    device_indices_str = [d.strip() for d in cuda_visible.split(",") if d.strip()]

    # Validate that indices don't exceed physical GPU count
    num_physical = number_of_physical_gpus()
    if num_physical > 0:
        try:
            device_indices = [int(idx) for idx in device_indices_str]
            invalid_indices = [idx for idx in device_indices if idx >= num_physical]
            if invalid_indices:
                logger.warning(
                    f"CUDA_VISIBLE_DEVICES references non-existent GPU(s): {invalid_indices}. "
                    f"Only {num_physical} physical GPU(s) available (indices 0-{num_physical - 1}). "
                    f"These invalid indices will be ignored by CUDA."
                )
        except ValueError:
            # CUDA_VISIBLE_DEVICES can also contain UUIDs, not just indices
            # In that case, skip validation
            pass

    return len(device_indices_str)


# Backward compatibility alias - prefer number_of_visible_gpus() in new code
def number_of_available_gpus() -> int:
    """Deprecated: Use number_of_visible_gpus() instead.

    This function is kept for backward compatibility and calls number_of_visible_gpus().
    """
    return number_of_visible_gpus()


def _is_local_gpu_available() -> bool:
    """Check if a local NVIDIA GPU is available via nvidia-smi."""
    return shutil.which("nvidia-smi") is not None and number_of_visible_gpus() > 0


def get_gpu_memory_used_physical(physical_device_id: int) -> int:
    """Get GPU memory used in bytes via nvidia-smi using physical device ID.

    IMPORTANT: This function uses PHYSICAL device IDs (0-indexed), which are the
    actual hardware GPU indices reported by nvidia-smi. These are NOT affected by
    CUDA_VISIBLE_DEVICES. For example, if CUDA_VISIBLE_DEVICES="3,5,7", physical
    device ID 3 refers to the actual GPU 3, not logical cuda:0.

    For logical device ID handling (e.g., "cuda:0"), use the DeviceManager wrapper
    method get_gpu_memory_used() which handles the mapping.

    Args:
        physical_device_id (int): Physical GPU index (0-based hardware index)

    Returns:
        int: Memory used in bytes, or 0 if query fails

    Example:
        >>> # Direct physical GPU query
        >>> mem = get_gpu_memory_used_physical(3)  # Queries actual hardware GPU 3
        >>> print(f"Physical GPU 3 using {mem / 1e9:.2f} GB")
        Physical GPU 3 using 1.23 GB

    Note:
        - Queries memory.used from nvidia-smi (memory allocated by active contexts)
        - Returns bytes for consistency with torch.cuda.memory_allocated()
        - Works across nvidia-smi versions using standard --query-gpu interface
    """
    out = _run_nvidia_smi_query(
        f"--id={physical_device_id}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    )
    if out is not None:
        try:
            # nvidia-smi returns memory in MiB, convert to bytes
            mib = int(out.strip())
            return mib * 1024 * 1024
        except ValueError:
            return 0
    return 0


def _parse_count_suffix(device: str, prefix: str) -> int:
    """Extract and validate a positive integer count after *prefix*.

    *prefix* must include the separator character (e.g. ``"cudax"``),
    so the remainder is purely numeric.

    Args:
        device (str): Full device string (e.g., ``"cudax2"``).
        prefix (str): Prefix including separator (e.g., ``"cudax"``).

    Raises:
        ValueError: If the suffix is missing, non-numeric, zero, or negative.
    """
    suffix = device[len(prefix) :]
    if not suffix:
        raise ValueError(f"Invalid device string: '{device}' (missing count)")
    try:
        num = int(suffix)
    except ValueError:
        raise ValueError(f"Invalid device string: '{device}' (count must be integer)") from None
    if num < 1:
        raise ValueError(f"Invalid count in '{device}': must be >= 1")
    return num


def _validate_cuda_index(idx_str: str, device: str) -> int:
    """Parse a CUDA device index, rejecting empty, non-integer, or negative values.

    Negatives are rejected because they would otherwise wrap to the last GPU via list indexing.
    """
    if not idx_str:
        raise ValueError(f"Invalid device: '{device}' (missing index)")
    try:
        idx = int(idx_str)
    except ValueError:
        raise ValueError(f"Invalid device: '{device}' (index must be integer)") from None
    if idx < 0:
        raise ValueError(f"Invalid device: '{device}' (index must be non-negative, got {idx})")
    return idx


def parse_device_string(device: str) -> DeviceSpec:
    """Parse a device string into a structured :class:`DeviceSpec`.

    Supports CPU, CUDA (single/multi, auto/explicit), and ``"cloud"`` device
    strings. ``"cloud"`` runs the tool on Proto's remote execution service
    (see :mod:`proto_tools.cloud`); its ``count`` is always 1.

    Args:
        device (str): Device string to parse.

    Returns:
        DeviceSpec: class:`DeviceSpec` describing the parsed device.

    Examples:
        >>> parse_device_string("cpu")
        DeviceSpec(kind='cpu', devices=['cpu'], count=1)
        >>> parse_device_string("cloud")
        DeviceSpec(kind='cloud', devices=['cloud'], count=1)
        >>> parse_device_string("cuda")
        DeviceSpec(kind='cuda', devices=None, count=1)
        >>> parse_device_string("cudax2")
        DeviceSpec(kind='cuda', devices=None, count=2)
        >>> parse_device_string("cuda:0")
        DeviceSpec(kind='cuda', devices=['cuda:0'], count=1)
        >>> parse_device_string("cuda:0,1")
        DeviceSpec(kind='cuda', devices=['cuda:0', 'cuda:1'], count=2)

    Raises:
        ValueError: If device string format is invalid.
    """
    device = device.strip()

    # CPU
    if device == "cpu":
        return DeviceSpec(kind="cpu", devices=["cpu"], count=1)

    # Cloud (dispatch delegated to proto_tools.cloud when enabled)
    if device == "cloud":
        return DeviceSpec(kind="cloud", devices=["cloud"], count=1)

    # Auto-allocate N GPUs: "cudax2", "cudax3", etc.
    if device.startswith("cudax"):
        num = _parse_count_suffix(device, "cudax")
        return DeviceSpec(kind="cuda", devices=None, count=num)

    # Single auto-allocate GPU
    if device == "cuda":
        return DeviceSpec(kind="cuda", devices=None, count=1)

    # Explicit device(s)
    if "," not in device:
        # Single explicit device: "cuda:0"
        if not device.startswith("cuda:"):
            raise ValueError(f"Invalid device: '{device}'")
        idx = _validate_cuda_index(device.split(":", 1)[1], device)
        return DeviceSpec(kind="cuda", devices=[f"cuda:{idx}"], count=1)

    # Multiple explicit devices: "cuda:0,1" or "cuda:0,cuda:1"
    parts = [p.strip() for p in device.split(",")]
    devices = []
    prefix = None

    for i, part in enumerate(parts):
        if ":" in part:
            # Explicit "cuda:N"
            if not part.startswith("cuda:"):
                raise ValueError(f"Invalid device: '{part}'")
            idx = _validate_cuda_index(part.split(":", 1)[1], device)
            if i == 0:
                prefix = "cuda"
        elif prefix:
            # Shorthand "N" after "cuda:N"
            idx = _validate_cuda_index(part, device)
        else:
            raise ValueError(f"Invalid device string '{device}': shorthand '{part}' without prefix")
        devices.append(f"cuda:{idx}")

    return DeviceSpec(kind="cuda", devices=devices, count=len(devices))


def _get_parent_device_list() -> list[str] | None:
    """Return parsed CUDA_VISIBLE_DEVICES from the parent environment, or None."""
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_visible and parent_visible.strip():
        return [d.strip() for d in parent_visible.split(",")]
    return None


def _validate_and_map_cuda_indices(
    cuda_indices: list[int],
    parent_device_list: list[str] | None,
) -> list[str]:
    """Validate CUDA indices against available GPUs and map to physical indices.

    Args:
        cuda_indices (list[int]): Logical CUDA device indices to validate.
        parent_device_list (list[str] | None): Parsed CUDA_VISIBLE_DEVICES, or None.

    Returns:
        list[str]: Physical device index strings, deduplicated and in input order.

    Raises:
        ValueError: If no GPUs available or an index exceeds visible GPUs.
    """
    if not cuda_indices:
        return []

    if min(cuda_indices) < 0:
        raise ValueError(f"CUDA indices must be non-negative; got {cuda_indices}")

    num_gpus = len(parent_device_list) if parent_device_list else number_of_available_gpus()
    max_idx = max(cuda_indices)

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    if num_gpus == 0:
        raise ValueError(
            f"Requested CUDA indices {cuda_indices} but no GPUs detected (CUDA_VISIBLE_DEVICES={cvd}; check nvidia-smi)"
        )
    if max_idx >= num_gpus:
        if parent_device_list:
            raise ValueError(
                f"CUDA index {max_idx} out of range for parent CUDA_VISIBLE_DEVICES={cvd} "
                f"(length {len(parent_device_list)}; requested {cuda_indices})"
            )
        raise ValueError(
            f"CUDA index {max_idx} out of range; only {num_gpus} GPU(s) available "
            f"(CUDA_VISIBLE_DEVICES={cvd}; requested {cuda_indices})"
        )

    seen: set[str] = set()
    physical: list[str] = []
    for idx in cuda_indices:
        mapped = parent_device_list[idx] if parent_device_list else str(idx)
        if mapped not in seen:
            seen.add(mapped)
            physical.append(mapped)
    return physical


def determine_visible_devices(device: int | str | list[int | str]) -> str:
    """Returns a string corresponding to the CUDA_VISIBLE_DEVICES environment variable.

    for a given device or devices.

    Supports single and multi-GPU device strings and correctly handles the case where
    the parent process has CUDA_VISIBLE_DEVICES set, mapping logical indices to
    physical device indices.

    When given a list, collects all CUDA indices across entries, validates them
    in a single pass (one nvidia-smi call), and returns the deduplicated physical
    indices. Non-CUDA entries (cpu) are skipped.

    Args:
        device (int | str | list[int | str]): Device specification. Accepts an int, single device string, or list of
            ints/strings (e.g. ``["cuda:0", "cuda:1"]``).

    Returns:
        str: CUDA_VISIBLE_DEVICES value (comma-separated device indices)

    Examples:
        >>> determine_visible_devices("cpu")
        ""
        >>> determine_visible_devices("cuda")
        "0"
        >>> determine_visible_devices("cuda:0")
        "0"
        >>> determine_visible_devices("cuda:0,1")
        "0,1"
        >>> determine_visible_devices("cuda:0,cuda:1")
        "0,1"
        >>> determine_visible_devices(["cuda:0", "cuda:1"])
        "0,1"

        With parent CUDA_VISIBLE_DEVICES=3,5,7:
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "3,5,7"
        >>> determine_visible_devices("cuda:0")  # Maps to physical GPU 3
        "3"
        >>> determine_visible_devices("cuda:1")  # Maps to physical GPU 5
        "5"

    Raises:
        ValueError: If device indices exceed available GPUs
    """
    parent_device_list = _get_parent_device_list()

    # List input: collect all CUDA indices, validate once, return physical indices
    if isinstance(device, list):
        all_cuda_indices: list[int] = []
        for d in device:
            if isinstance(d, int):
                all_cuda_indices.append(d)
                continue
            spec = parse_device_string(d)
            if spec.kind != "cuda" or spec.devices is None:
                continue
            all_cuda_indices.extend(int(dev.split(":")[1]) for dev in spec.devices)
        physical = _validate_and_map_cuda_indices(all_cuda_indices, parent_device_list)
        return ",".join(physical)

    # If we are using the CPU, set no devices to be visible
    if device == "cpu":
        return ""

    # Handle integer device index (legacy support)
    if isinstance(device, int):
        device_int = device
        num_gpus = number_of_available_gpus()
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
        if device_int >= num_gpus:
            raise ValueError(f"Device index {device_int} >= visible GPU count ({num_gpus}); CUDA_VISIBLE_DEVICES={cvd}")
        if parent_device_list:
            if device_int >= len(parent_device_list):
                raise ValueError(
                    f"Device index {device_int} exceeds parent CUDA_VISIBLE_DEVICES length ({len(parent_device_list)}); "
                    f"CUDA_VISIBLE_DEVICES={cvd}"
                )
            return parent_device_list[device_int]
        return str(device)

    # Parse device string
    device_str = str(device)
    spec = parse_device_string(device_str)

    # If auto-allocate (spec.devices is None), fall back to sequential allocation
    # Note: DeviceManager should resolve "cudax2" to specific devices before calling this
    if spec.devices is None:
        if parent_device_list:
            # Map sequential indices to parent's visible devices
            return ",".join(parent_device_list[i] for i in range(min(spec.count, len(parent_device_list))))
        return ",".join(str(i) for i in range(spec.count))

    # CPU case
    if len(spec.devices) == 1 and spec.devices[0] == "cpu":
        return ""

    # Extract CUDA indices and delegate validation + mapping
    cuda_indices = [int(dev.split(":", 1)[1]) for dev in spec.devices]
    physical = _validate_and_map_cuda_indices(cuda_indices, parent_device_list)
    return ",".join(physical)


def parse_device_count_requirement(spec: str) -> dict[str, int | None]:
    """Parse device count requirement into min/max range.

    Args:
        spec (str): Device count specification string

    Returns:
        dict[str, int | None]: Dict with "min" and "max" keys (None means unbounded)

    Examples:
        >>> parse_device_count_requirement("1")
        {"min": 1, "max": 1}
        >>> parse_device_count_requirement("1-2")
        {"min": 1, "max": 2}
        >>> parse_device_count_requirement(">=1")
        {"min": 1, "max": None}
        >>> parse_device_count_requirement("<=2")
        {"min": None, "max": 2}
        >>> parse_device_count_requirement(">=1,<=4")
        {"min": 1, "max": 4}

    Raises:
        ValueError: If format is invalid
    """
    spec = spec.strip()

    # Check for combined spec: ">=1,<=4"
    if "," in spec:
        parts = [p.strip() for p in spec.split(",")]
        result: dict[str, int | None] = {"min": None, "max": None}

        for part in parts:
            if part.startswith(">="):
                try:
                    result["min"] = int(part[2:])
                except ValueError:
                    raise ValueError(f"Invalid device count specification: '{spec}'") from None
            elif part.startswith("<="):
                try:
                    result["max"] = int(part[2:])
                except ValueError:
                    raise ValueError(f"Invalid device count specification: '{spec}'") from None
            else:
                raise ValueError(
                    f"Invalid device count specification: '{spec}'. Combined specs must use '>=' or '<=' operators."
                )

        # Validate min <= max if both present
        if result["min"] is not None and result["max"] is not None and result["min"] > result["max"]:
            raise ValueError(
                f"Invalid device count specification: '{spec}'. "
                f"Minimum ({result['min']}) cannot exceed maximum ({result['max']})."
            )

        return result

    # Open-ended specs: ">=1", "<=2"
    if spec.startswith(">="):
        try:
            min_count = int(spec[2:])
            return {"min": min_count, "max": None}
        except ValueError:
            raise ValueError(f"Invalid device count specification: '{spec}'") from None

    if spec.startswith("<="):
        try:
            max_count = int(spec[2:])
            return {"min": None, "max": max_count}
        except ValueError:
            raise ValueError(f"Invalid device count specification: '{spec}'") from None

    # Range spec: "1-2"
    if "-" in spec:
        parts = spec.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid device count specification: '{spec}'")

        try:
            min_count = int(parts[0])
            max_count = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid device count specification: '{spec}'") from None

        if min_count > max_count:
            raise ValueError(
                f"Invalid device count specification: '{spec}'. "
                f"Minimum ({min_count}) cannot exceed maximum ({max_count})."
            )

        if min_count < 1:
            raise ValueError(f"Invalid device count specification: '{spec}'. Device count must be >= 1.")

        return {"min": min_count, "max": max_count}

    # Exact count: "1", "2"
    try:
        count = int(spec)
    except ValueError:
        raise ValueError(f"Invalid device count specification: '{spec}'") from None

    if count < 1:
        raise ValueError(f"Invalid device count specification: '{spec}'. Device count must be >= 1.")
    return {"min": count, "max": count}


def validate_device_allocation(requested_count: int, requirement: str, tool_key: str) -> None:
    """Validate requested device count against requirement.

    Args:
        requested_count (int): Number of devices being allocated
        requirement (str): Device count requirement string (e.g., "1", "1-2", ">=1")
        tool_key (str): Tool identifier for error messages

    Behavior:
        - requested < min: Raise ValueError (tool won't work)
        - requested > max: Log WARNING (wastes resources)
        - within range: No action (valid allocation)

    Examples:
        Tool requires "1", user requests 2 → WARNING (over-allocation)
        Tool requires "2", user requests 1 → ERROR (under-allocation)
        Tool requires "1-2", user requests 2 → OK

    Raises:
        ValueError: If under-allocated (requested < minimum)
    """
    range_spec = parse_device_count_requirement(requirement)
    min_required = range_spec["min"]
    max_required = range_spec["max"]

    # Check under-allocation (ERROR)
    if min_required is not None and requested_count < min_required:
        raise ValueError(
            f"Tool '{tool_key}' requires at least {min_required} device(s), "
            f"but only {requested_count} requested. "
            f"Requirement: {requirement}"
        )

    # Check over-allocation (WARNING)
    if max_required is not None and requested_count > max_required:
        logger.warning(
            "Tool '%s' requires at most %d device(s), but %d requested. This may waste GPU resources. Requirement: %s",
            tool_key,
            max_required,
            requested_count,
            requirement,
        )


# ============================================================================
# GPU Memory Information
# ============================================================================


def get_gpu_memory_info() -> list[dict[str, int | str]]:
    """Get memory information for all GPUs.

    Returns detailed memory statistics for each GPU including total capacity,
    used memory, and free memory. All values are in bytes for consistency
    with other memory utilities.

    Returns:
        list[dict[str, int | str]]: List of dicts with keys:
            - index: GPU index (int)
            - name: GPU name (str)
            - total_bytes: Total memory capacity in bytes (int)
            - used_bytes: Used memory in bytes (int)
            - free_bytes: Free memory in bytes (int)
            Empty list if nvidia-smi is not available or fails.

    Example:
        >>> mem_info = get_gpu_memory_info()
        >>> for gpu in mem_info:
        ...     print(f"GPU {gpu['index']}: {gpu['used_bytes'] / 1e9:.1f} GB / {gpu['total_bytes'] / 1e9:.1f} GB")
        GPU 0: 1.2 GB / 80.0 GB
        GPU 1: 15.3 GB / 80.0 GB
    """
    out = _run_nvidia_smi_query(
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    )
    if out is None:
        return []

    try:
        gpus: list[dict[str, int | str]] = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "total_bytes": int(parts[2]) * 1024 * 1024,  # MiB to bytes
                        "used_bytes": int(parts[3]) * 1024 * 1024,
                        "free_bytes": int(parts[4]) * 1024 * 1024,
                    }
                )
        return gpus
    except ValueError:
        return []


def get_gpu_process_memory() -> list[dict[str, int | str]]:
    """Get per-process GPU memory usage across all GPUs.

    Queries nvidia-smi for all processes currently using GPU resources and
    their memory consumption. Useful for understanding which processes are
    occupying GPU memory.

    Returns:
        list[dict[str, int | str]]: List of dicts with keys:
            - gpu_index: GPU index (int) - extracted from PCI bus ID
            - pid: Process ID (int)
            - process_name: Process name (str)
            - used_bytes: Memory used by this process in bytes (int)
            Empty list if nvidia-smi is not available, fails, or no processes
            are using GPUs.

    Example:
        >>> processes = get_gpu_process_memory()
        >>> for proc in processes:
        ...     print(
        ...         f"PID {proc['pid']} ({proc['process_name']}): {proc['used_bytes'] / 1e9:.1f} GB on GPU {proc['gpu_index']}"
        ...     )
        PID 12345 (python): 2.3 GB on GPU 0
        PID 67890 (python): 15.1 GB on GPU 1

    Note:
        GPU index is inferred from PCI bus ID by querying nvidia-smi for the mapping.
        If the mapping fails, gpu_index will be -1.
    """
    out = _run_nvidia_smi_query(
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    if out is None:
        return []

    try:
        # Build PCI bus ID to GPU index mapping
        bus_to_index = _get_pci_bus_to_index_mapping()

        processes: list[dict[str, int | str]] = []
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_bus_id = parts[0]
                gpu_index = bus_to_index.get(gpu_bus_id, -1)

                processes.append(
                    {
                        "gpu_index": gpu_index,
                        "pid": int(parts[1]),
                        "process_name": parts[2],
                        "used_bytes": int(parts[3]) * 1024 * 1024,  # MiB to bytes
                    }
                )
        return processes
    except ValueError:
        return []


def _get_pci_bus_to_index_mapping() -> dict[str, int]:
    """Build a mapping from PCI bus ID to GPU index.

    Returns:
        dict[str, int]: Dict mapping PCI bus ID (str) to GPU index (int).
            Empty dict if query fails.
    """
    out = _run_nvidia_smi_query(
        "--query-gpu=index,pci.bus_id",
        "--format=csv,noheader,nounits",
    )
    if out is None:
        return {}

    try:
        mapping = {}
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                mapping[parts[1]] = int(parts[0])
        return mapping
    except ValueError:
        return {}


def display_gpu_memory_usage(
    bar_width: int = 40,
    name_width: int = 30,
    show_processes: bool = False,
    verbose: bool = False,
    devices: list[int] | None = None,
) -> None:
    """Display GPU memory usage with horizontal bar chart visualization.

    Compact, fixed-width display of GPU memory usage with optional process details.
    Gracefully handles systems without GPUs by returning silently.

    Args:
        bar_width (int): Width of the progress bar in characters (default: 40)
        name_width (int): Maximum width for GPU name (default: 30, will be truncated)
        show_processes (bool): Show individual processes using each GPU (default: False)
        verbose (bool): If True, show full details (name, percentage). If False, show
            a compact one-line-per-GPU view. (default: False)
        devices (list[int] | None): List of GPU indices to display (default: None, show all).
            Example: ``[0, 1]`` shows only GPU 0 and GPU 1.

    Example output (verbose=True, show_processes=False):
        GPU 0 | NVIDIA H100 80GB HBM3      | [████░░░░░░░░░░░░░░░░░░░░] | 12.5 / 80.0 GB (15.6%)
        GPU 1 | NVIDIA H100 80GB HBM3      | [████████████████████░░░░] | 64.1 / 80.0 GB (80.1%)

    Example output (verbose=True, show_processes=True):
        GPU 0 | NVIDIA H100 80GB HBM3      | [████░░░░░░░░░░░░░░░░░░░░] | 12.5 / 80.0 GB (15.6%)
          → PID 12345 (python): 10.2 GB
          → PID 67890 (pytorch): 2.3 GB

    Example output (verbose=False):
        GPU 0: 12.5 / 80.0 GB  ████░░░░░░░░░░
        GPU 1: 64.1 / 80.0 GB  ████████████░░

    Note:
        Prints to stdout. Returns None.
        Silently returns if no GPUs are available (not an error condition).
    """
    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        # Silently return - no GPU is not an error condition
        return

    # Filter to requested devices
    if devices is not None:
        device_set = set(devices)
        gpu_info = [g for g in gpu_info if g["index"] in device_set]

    # Compact mode: one short line per GPU
    if not verbose:
        compact_bar_width = 14
        for gpu in gpu_info:
            idx = gpu["index"]
            total_gb = gpu["total_bytes"] / 1e9  # type: ignore[operator]
            used_gb = gpu["used_bytes"] / 1e9  # type: ignore[operator]
            filled = int((used_gb / total_gb) * compact_bar_width) if total_gb > 0 else 0
            filled = max(0, min(filled, compact_bar_width))
            bar = "█" * filled + "░" * (compact_bar_width - filled)
            print(f"GPU {idx}: {used_gb:5.1f} / {total_gb:5.1f} GB  {bar}")
        return

    process_info = get_gpu_process_memory() if show_processes else []

    # Group processes by GPU
    processes_by_gpu: dict[int, list[dict[str, Any]]] = {}
    if show_processes:
        for proc in process_info:
            gpu_idx = proc["gpu_index"]
            if gpu_idx not in processes_by_gpu:
                processes_by_gpu[gpu_idx] = []  # type: ignore[index]
            processes_by_gpu[gpu_idx].append(proc)  # type: ignore[index]

    # Display each GPU
    for gpu in gpu_info:
        idx = gpu["index"]
        name = gpu["name"]
        # Truncate or pad GPU name to fixed width
        name_display = name[:name_width].ljust(name_width)  # type: ignore[index]

        total_bytes = gpu["total_bytes"]
        used_bytes = gpu["used_bytes"]
        total_gb = total_bytes / 1e9  # type: ignore[operator]
        used_gb = used_bytes / 1e9  # type: ignore[operator]
        utilization = (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0  # type: ignore[operator]

        # Build progress bar (guaranteed to be exactly bar_width characters)
        filled = int((used_gb / total_gb) * bar_width) if total_gb > 0 else 0
        filled = max(0, min(filled, bar_width))  # Clamp to [0, bar_width]
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print compact single-line format with fixed widths
        print(f"GPU {idx:2d} | {name_display} | [{bar}] | {used_gb:5.1f} / {total_gb:5.1f} GB ({utilization:5.1f}%)")

        # Print processes if requested
        if show_processes:
            gpu_processes = processes_by_gpu.get(idx, [])  # type: ignore[arg-type]
            for proc in gpu_processes:
                proc_gb = proc["used_bytes"] / 1e9  # type: ignore[operator]
                print(f"  → PID {proc['pid']:6d} ({proc['process_name']:12s}): {proc_gb:5.1f} GB")
