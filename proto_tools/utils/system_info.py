"""proto_tools/utils/system_info.py.

Collects platform, GPU, and environment information without torch dependency.
"""

import functools
import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================
@dataclass
class PlatformInfo:
    """Platform/OS information."""

    os: str
    os_version: str
    architecture: str
    hostname: str
    python_version: str
    ram_gb: float


@dataclass
class GPUDevice:
    """Single GPU device information."""

    index: int
    name: str
    compute_capability: str
    vram_gb: float


@dataclass
class GPUInfo:
    """GPU availability and device information."""

    available: bool
    count: int
    driver_version: str | None
    cuda_version: str | None
    devices: list[GPUDevice] = field(default_factory=list)


@dataclass
class ParentEnv:
    """Parent process environment (venv, conda, mamba, or none)."""

    type: str  # "venv", "conda", "mamba", "none"
    name: str | None
    prefix: str | None


# ============================================================================
# Platform Info
# ============================================================================
def get_platform_info() -> PlatformInfo:
    """Collect platform/OS information.

    Returns:
        PlatformInfo: OS, architecture, hostname, Python version, and RAM.
    """
    uname = platform.uname()

    # Get RAM in GB
    ram_gb = _get_ram_gb()

    return PlatformInfo(
        os=uname.system,
        os_version=f"{uname.system} {uname.release}",
        architecture=uname.machine,
        hostname=uname.node,
        python_version=platform.python_version(),
        ram_gb=ram_gb,
    )


def _get_ram_gb() -> float:
    """Get total RAM in GB."""
    try:
        if platform.system() == "Darwin":
            # macOS: use sysctl
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024**3)
        else:
            # Linux: read from /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb / (1024**2)
    except Exception as e:
        logger.debug(f"Failed to get RAM info: {e}")
    return 0.0


# Values at or above this (or the literal "max") mean the cgroup is effectively unbounded.
_CGROUP_UNLIMITED = 1 << 62


def _read_cgroup_value(path: Path) -> int | None:
    """Read a single-integer cgroup file; return None for missing, ``"max"``, or unparseable."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text or text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _cgroup_memory_limit_bytes() -> int | None:
    """Effective cgroup memory limit for this process in bytes, or None if unbounded/unavailable.

    Handles both cgroup hierarchies portably:

    - v2 (unified): reads ``memory.max`` from the process's cgroup up to the
      root, keeping the smallest concrete limit (a parent — e.g. a Slurm job
      cgroup — is often the one that actually caps us).
    - v1: reads ``memory.limit_in_bytes`` for the ``memory`` controller.

    Returns None when no controller file is present or every limit is "max"
    (unbounded), so callers fall back to physical RAM.
    """
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None

    v2_path: str | None = None
    v1_path: str | None = None
    for line in cgroup_lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers, cgroup_path = parts
        if hierarchy_id == "0" and controllers == "":
            v2_path = cgroup_path  # unified (v2) hierarchy
        elif "memory" in controllers.split(","):
            v1_path = cgroup_path

    root = Path("/sys/fs/cgroup")

    # cgroup v2: walk from the leaf cgroup to the root; keep the smallest concrete cap.
    if v2_path is not None:
        limits: list[int] = []
        current = root / v2_path.lstrip("/")
        while True:
            value = _read_cgroup_value(current / "memory.max")
            if value is not None and value < _CGROUP_UNLIMITED:
                limits.append(value)
            if current == root:
                break
            current = current.parent
        if limits:
            return min(limits)

    # cgroup v1: a single limit file under the memory controller mount.
    if v1_path is not None:
        value = _read_cgroup_value(root / "memory" / v1_path.lstrip("/") / "memory.limit_in_bytes")
        if value is not None and value < _CGROUP_UNLIMITED:
            return value

    return None


def available_memory_bytes() -> int:
    """Best-effort memory budget for this process in bytes, respecting cgroup caps.

    Returns the smaller of physical RAM (``/proc/meminfo`` / ``sysctl``) and the
    effective cgroup memory limit, so Slurm / container / k8s caps are honored
    rather than the physical host size. Falls back to physical RAM when no cgroup
    limit applies, and to 0 only when nothing is readable.

    Pure-stdlib and portable: reads ``/proc`` and ``/sys/fs/cgroup`` directly
    (cgroup v1 and v2), degrading gracefully on systems without them.
    """
    physical = int(_get_ram_gb() * 1024**3)
    cgroup_limit = _cgroup_memory_limit_bytes()
    candidates = [v for v in (physical, cgroup_limit) if v and v > 0]
    return min(candidates) if candidates else 0


def resolve_num_threads(configured: int | None) -> int:
    """Return ``configured`` if set, else the CPU count this process may use (always >= 1).

    Auto-detection prefers ``os.sched_getaffinity(0)`` (cgroup/Slurm-aware on
    Linux: the allocated cores), falling back to ``os.cpu_count()`` on platforms
    where it is unavailable (macOS, Windows).
    """
    if configured is not None:
        return configured
    try:
        return len(os.sched_getaffinity(0)) or 1
    except AttributeError:
        return os.cpu_count() or 1


# ============================================================================
# GPU Info
# ============================================================================
@functools.lru_cache(maxsize=1)
def get_gpu_info() -> GPUInfo:
    """Collect GPU information using nvidia-smi.

    Cached for the lifetime of the process

    Returns:
        GPUInfo: GPU availability, count, driver/CUDA versions, and device details.
    """
    # Check if CUDA is explicitly disabled
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return GPUInfo(
            available=False,
            count=0,
            driver_version=None,
            cuda_version=None,
            devices=[],
        )

    # Check if nvidia-smi exists
    if not shutil.which("nvidia-smi"):
        return GPUInfo(
            available=False,
            count=0,
            driver_version=None,
            cuda_version=None,
            devices=[],
        )

    try:
        # Query GPU info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return GPUInfo(
                available=False,
                count=0,
                driver_version=None,
                cuda_version=None,
                devices=[],
            )

        devices = []
        driver_version = None

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                index = int(parts[0])
                name = parts[1]
                compute_cap = parts[2]
                try:
                    vram_gb = round(float(parts[3]) / 1024, 1)
                except (ValueError, TypeError):
                    # Unified memory GPUs (e.g., GB10) report [N/A] for
                    # dedicated VRAM; fall back to total system RAM since
                    # it's all GPU-addressable.
                    vram_gb = round(_get_ram_gb(), 1)
                driver_version = parts[4]

                devices.append(
                    GPUDevice(
                        index=index,
                        name=name,
                        compute_capability=compute_cap,
                        vram_gb=vram_gb,
                    )
                )

        # Get CUDA version separately
        cuda_version = _get_cuda_version()

        return GPUInfo(
            available=len(devices) > 0,
            count=len(devices),
            driver_version=driver_version,
            cuda_version=cuda_version,
            devices=devices,
        )

    except Exception as e:
        logger.debug(f"Failed to get GPU info: {e}")
        return GPUInfo(
            available=False,
            count=0,
            driver_version=None,
            cuda_version=None,
            devices=[],
        )


def _get_cuda_version() -> str | None:
    """Get CUDA toolkit version from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse CUDA Version from output like "CUDA Version: 12.9"
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except Exception:  # noqa: S110 -- best-effort CUDA version detection
        pass
    return None


# ============================================================================
# Parent Process Environment
# ============================================================================
def get_parent_process_env() -> ParentEnv:
    """Detect the active virtual environment (venv, conda, or mamba).

    Returns:
        ParentEnv: Environment type, name, and prefix path.
    """
    # Check for mamba (takes precedence over conda)
    if os.environ.get("MAMBA_EXE"):
        prefix = os.environ.get("CONDA_PREFIX")
        name = os.environ.get("CONDA_DEFAULT_ENV")
        return ParentEnv(type="mamba", name=name, prefix=prefix)

    # Check for conda
    if os.environ.get("CONDA_PREFIX"):
        prefix = os.environ.get("CONDA_PREFIX")
        name = os.environ.get("CONDA_DEFAULT_ENV")
        return ParentEnv(type="conda", name=name, prefix=prefix)

    # Check for standard venv
    if os.environ.get("VIRTUAL_ENV"):
        prefix = os.environ.get("VIRTUAL_ENV")
        name = Path(prefix).name if prefix else None
        return ParentEnv(type="venv", name=name, prefix=prefix)

    return ParentEnv(type="none", name=None, prefix=None)


# ============================================================================
# Environment Variables
# ============================================================================

# Patterns that indicate sensitive environment variables (case-insensitive substring match)
_SENSITIVE_PATTERNS = [
    # Credentials & secrets
    "KEY",
    "TOKEN",
    "SECRET",
    "PASS",
    "AUTH",
    "CRED",
    "PRIVATE",
    # API & access
    "BEARER",
    "OAUTH",
    "API",
    "ACCESS",
    "WEBHOOK",
    # Encryption & signing
    "ENCRYPT",
    "SIGNING",
    "SALT",
    "CERT",
    "PEM",
    "JWT",
    # Session & identity
    "SESSION",
    "COOKIE",
    "SID",
    # Connection strings
    "DSN",
    # SSH & remote access
    "SSH",
]


def _is_sensitive_env_var(key: str, value: str) -> bool:
    """Check if an environment variable should be redacted from reports.

    Args:
        key (str): Environment variable name.
        value (str): Environment variable value.

    Returns:
        bool: True if the variable should be excluded from reports.
    """
    key_upper = key.upper()

    # Check if key contains any sensitive pattern
    for pattern in _SENSITIVE_PATTERNS:
        if pattern in key_upper:
            return True

    # Check for connection strings with embedded credentials (user:pass@host pattern)
    # Common in DATABASE_URL, REDIS_URL, MONGO_URI, etc.
    return bool(key_upper.endswith(("_URL", "_URI")) and "://" in value and "@" in value)


def _sanitize_env_dict(env: dict[str, str] | None) -> dict[str, str] | None:
    """Remove sensitive environment variables from a dict.

    Args:
        env (dict[str, str] | None): Environment variable dictionary to sanitize.

    Returns:
        dict[str, str] | None: Sanitized copy with sensitive variables removed.
    """
    if env is None:
        return None
    return {k: v for k, v in env.items() if not _is_sensitive_env_var(k, v)}


# Module-level storage for captured environment variables
_captured_parent_env: dict[str, str] | None = None
_captured_subprocess_env: dict[str, str] | None = None


def capture_parent_env() -> dict[str, str]:
    """Capture and store the current process environment variables.

    Call this once at the start of a test session to record the parent
    process environment before any subprocesses are spawned.

    Returns:
        dict[str, str]: Copy of os.environ at capture time.
    """
    global _captured_parent_env
    _captured_parent_env = dict(os.environ)
    logger.debug("Captured parent process environment (%d vars)", len(_captured_parent_env))
    return _captured_parent_env


def capture_subprocess_env(env: dict[str, str]) -> None:
    """Record the environment variables passed to a subprocess.

    Call this from subprocess execution code (e.g., _build_subprocess_env or ToolInstance)
    to record what env vars are actually passed to tool subprocesses.

    Args:
        env (dict[str, str]): The environment dict being passed to subprocess.
    """
    global _captured_subprocess_env
    _captured_subprocess_env = dict(env)
    logger.debug("Captured subprocess environment (%d vars)", len(_captured_subprocess_env))


def get_captured_env() -> dict[str, Any]:
    """Retrieve captured environment variables for reporting.

    Sensitive variables (containing KEY, TOKEN, SECRET, PASS, AUTH, etc.)
    are automatically excluded from the returned data.

    Returns:
        dict[str, Any]: parent_env and subprocess_env (each a dict or None).
    """
    return {
        "parent_env": _sanitize_env_dict(_captured_parent_env),
        "subprocess_env": _sanitize_env_dict(_captured_subprocess_env),
    }


def clear_captured_env() -> None:
    """Clear captured environment variables."""
    global _captured_parent_env, _captured_subprocess_env
    _captured_parent_env = None
    _captured_subprocess_env = None


# ============================================================================
# Platform ID
# ============================================================================
def _get_username() -> str | None:
    """Get the current OS username."""
    try:
        return os.getlogin()
    except OSError:
        # os.getlogin() can fail in non-interactive contexts (e.g., cron, SSH)
        return os.environ.get("USER") or os.environ.get("USERNAME")


# Hostname regex -> friendly name. Checked in order; first match wins.
# Add cluster-specific patterns here as (compiled_pattern, friendly_name) tuples.
_HOSTNAME_ALIASES: list[tuple[re.Pattern[str], str]] = []


def _resolve_hostname_alias(hostname: str) -> str | None:
    """Return a friendly name if the hostname matches a known pattern."""
    for pattern, alias in _HOSTNAME_ALIASES:
        if pattern.search(hostname):
            return alias
    return None


def _sanitize_hostname(hostname: str) -> str:
    """Make a hostname safe for use in filenames.

    Args:
        hostname (str): Raw hostname string to sanitize.

    - Strips the DNS domain suffix (everything after the first dot)
    - Keeps only alphanumeric characters and hyphens
    - Collapses consecutive separators and strips leading/trailing ones
    - Truncates to 40 characters
    """
    # Strip domain suffix: "sh04-15n01.int" -> "sh04-15n01"
    hostname = hostname.split(".")[0]
    # Keep only filename-safe characters
    hostname = re.sub(r"[^a-zA-Z0-9\-]", "-", hostname)
    # Collapse consecutive hyphens and strip edges
    hostname = re.sub(r"-+", "-", hostname).strip("-")
    # Truncate
    return hostname[:40]


def get_platform_id(
    *,
    include_user: bool = True,
    include_date: bool = True,
    include_commit: bool = True,
) -> str:
    """Generate a unique platform identifier string.

    Format: `[{user}_]{cluster_or_os}[_{hostname}]_{arch}_{gpu_or_cpu}[_{YYYYMMDD}][_{commit}]`

    The hostname is included for generic OS names (e.g. ``linux``) to
    disambiguate machines that would otherwise produce the same ID.
    Named platforms (dgx_spark, registered hostname aliases) and macOS
    already have unique OS parts, so the hostname is omitted for brevity.

    Examples:
    --------
    - Mac: `alice_macosDarwin_arm64_cpu_20260216_bcf5907`
    - DGX Spark: `alice_dgx_spark_arm64_gb10_20260216_bcf5907`
    - Unknown Linux: `alice_linux_myhost_x86_64_a100_20260216_bcf5907`

    Args:
        include_user (bool): Include username prefix (default True).
        include_date (bool): Include date suffix (default True).
        include_commit (bool): Include git commit hash suffix (default True).

    Returns:
        str: Platform identifier string.
    """
    platform_info = get_platform_info()
    gpu_info = get_gpu_info()

    parts: list[str] = []

    # Username prefix
    if include_user:
        username = _get_username()
        if username:
            parts.append(username)

    # Determine cluster/OS prefix
    # Named platforms and macOS are already unique; generic OS names
    # (e.g. "linux") get the hostname appended to disambiguate machines.
    # Hostname patterns registered in _HOSTNAME_ALIASES resolve to
    # friendly names and are treated as named platforms (no raw hostname).
    include_hostname = False
    hostname_alias = _resolve_hostname_alias(platform_info.hostname)
    if "dgx" in platform_info.hostname.lower() or "spark" in platform_info.hostname.lower():
        os_part = "dgx_spark"
    elif hostname_alias:
        os_part = hostname_alias
    elif platform_info.os.lower() == "darwin":
        os_part = "macosDarwin"
    else:
        os_part = platform_info.os.lower()
        include_hostname = True

    # Normalize architecture
    arch = platform_info.architecture
    if arch == "aarch64":
        arch = "arm64"

    # GPU or CPU suffix
    if gpu_info.available and gpu_info.devices:
        # Use first GPU name, normalized
        gpu_name = gpu_info.devices[0].name.lower()
        # Extract key identifier (h100, gb10, a100, etc.)
        if "h100" in gpu_name:
            gpu_part = "h100"
        elif "gb10" in gpu_name:
            gpu_part = "gb10"
        elif "a100" in gpu_name:
            gpu_part = "a100"
        elif "v100" in gpu_name:
            gpu_part = "v100"
        else:
            # Fallback: use compute capability
            gpu_part = f"sm{gpu_info.devices[0].compute_capability.replace('.', '')}"
    else:
        gpu_part = "cpu"

    parts.append(os_part)
    if include_hostname:
        hostname = _sanitize_hostname(platform_info.hostname)
        if hostname:
            parts.append(hostname)
    parts.extend([arch, gpu_part])

    if include_date:
        parts.append(datetime.now().strftime("%Y%m%d"))

    if include_commit:
        commit = _get_git_commit_short()
        if commit:
            parts.append(commit)

    return "_".join(parts)


def _get_git_commit_short(length: int = 7) -> str | None:
    """Get short git commit hash of HEAD, or package version as fallback."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", f"--short={length}", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: S110 -- best-effort git version detection
        pass
    # Not in a git repo (e.g., non-editable pip install); use package version
    try:
        from proto_tools import __version__

        return f"v{__version__}"
    except ImportError:
        return None


def get_git_info() -> dict[str, Any]:
    """Get git repository information.

    Returns:
        dict[str, Any]: commit (12-char short hash), branch name, dirty status,
            and package version (always present as fallback).
    """
    repo_root = Path(__file__).parent.parent.parent

    info: dict[str, Any] = {
        "commit": None,
        "branch": None,
        "dirty": False,
        "version": None,
    }

    # Always include package version
    try:
        from proto_tools import __version__

        info["version"] = __version__
    except ImportError:
        pass

    try:
        # Get commit hash (12 chars)
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if dirty (staged or unstaged changes)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())

    except Exception as e:
        logger.debug(f"Failed to get git info: {e}")

    return info


# ============================================================================
# Aggregate Collection
# ============================================================================
def collect_system_info() -> dict[str, Any]:
    """Collect all system information as a JSON-serializable dict.

    Returns:
        dict[str, Any]: All platform, GPU, and environment information.
    """
    platform_info = get_platform_info()
    gpu_info = get_gpu_info()
    parent_env = get_parent_process_env()
    git_info = get_git_info()

    return {
        "git_info": git_info,
        "platform": {
            "platform_id": get_platform_id(),
            "os": platform_info.os,
            "os_version": platform_info.os_version,
            "architecture": platform_info.architecture,
            "hostname": platform_info.hostname,
            "python_version": platform_info.python_version,
            "ram_gb": round(platform_info.ram_gb, 1),
        },
        "gpu": {
            "available": gpu_info.available,
            "count": gpu_info.count,
            "driver_version": gpu_info.driver_version,
            "cuda_version": gpu_info.cuda_version,
            "devices": [
                {
                    "index": d.index,
                    "name": d.name,
                    "compute_capability": d.compute_capability,
                    "vram_gb": d.vram_gb,
                }
                for d in gpu_info.devices
            ],
        },
        "parent_process_env": {
            "type": parent_env.type,
            "name": parent_env.name,
            "prefix": parent_env.prefix,
        },
    }
