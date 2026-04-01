"""
proto_tools/utils/system_info.py

Collects platform, GPU, and environment information without torch dependency.
"""

from __future__ import annotations

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

    Returns
    -------
    PlatformInfo
        OS, architecture, hostname, Python version, and RAM.
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


# ============================================================================
# GPU Info
# ============================================================================
@functools.lru_cache(maxsize=1)
def get_gpu_info() -> GPUInfo:
    """Collect GPU information using nvidia-smi.

    Cached for the lifetime of the process

    Returns
    -------
    GPUInfo
        GPU availability, count, driver/CUDA versions, and device details.
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
    except Exception:
        pass
    return None


# ============================================================================
# Parent Process Environment
# ============================================================================
def get_parent_process_env() -> ParentEnv:
    """Detect the active virtual environment (venv, conda, or mamba).

    Returns
    -------
    ParentEnv
        Environment type, name, and prefix path.
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
    if key_upper.endswith(("_URL", "_URI")):
        if "://" in value and "@" in value:
            return True

    return False


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

    Returns
    -------
    dict
        Copy of os.environ at capture time.
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

    Returns
    -------
    dict
        parent_env: Environment of the parent process (or None if not captured).
        subprocess_env: Environment passed to subprocesses (or None if not captured).
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


def _get_slurm_cluster_name() -> str | None:
    """Get SLURM cluster name if running on a SLURM cluster.

    Returns
    -------
    str or None
        Cluster name (e.g., "arc-slurm") or None if not on SLURM.
    """
    # First check environment variable (set by some SLURM configs)
    cluster_name = os.environ.get("SLURM_CLUSTER_NAME")
    if cluster_name:
        return cluster_name

    # If SLURM_JOB_ID is set, we're on SLURM - query scontrol for cluster name
    if os.environ.get("SLURM_JOB_ID"):
        try:
            result = subprocess.run(
                ["scontrol", "show", "config"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse "ClusterName = arc-slurm" from output
                match = re.search(r"ClusterName\s*=\s*(\S+)", result.stdout)
                if match:
                    return match.group(1)
        except Exception as e:
            logger.debug(f"Failed to query SLURM cluster name: {e}")

    return None


# Hostname regex -> friendly name. Checked in order; first match wins.
# Add new clusters here as (compiled_pattern, friendly_name) tuples.
_HOSTNAME_ALIASES: list[tuple[re.Pattern[str], str]] = [
    # Sherlock (Stanford): login nodes "sh03-lnNN", compute nodes "sh04-NNnNN"
    (re.compile(r"^sh\d+-", re.IGNORECASE), "sherlock"),
]


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
    Named clusters (chimera, dgx_spark) and macOS already have unique
    OS parts, so the hostname is omitted for brevity.

    Examples
    --------
    - Mac: `alice_macosDarwin_arm64_cpu_20260216_bcf5907`
    - Chimera: `bob_chimera_x86_64_h100_20260216_bcf5907`
    - DGX Spark: `alice_dgx_spark_arm64_gb10_20260216_bcf5907`
    - Sherlock: `viggiano_sherlock_x86_64_h100_20260216_bcf5907`
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
    # Named clusters and macOS are already unique; generic OS names
    # (e.g. "linux") get the hostname appended to disambiguate machines.
    # Known hostname patterns (see _HOSTNAME_ALIASES) are resolved to
    # friendly names and treated as named clusters (no raw hostname).
    include_hostname = False
    cluster_name = _get_slurm_cluster_name()
    hostname_alias = _resolve_hostname_alias(platform_info.hostname)
    if cluster_name == "arc-slurm":
        os_part = "chimera"
    elif "dgx" in platform_info.hostname.lower() or "spark" in platform_info.hostname.lower():
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
    except Exception:
        pass
    # Not in a git repo (e.g., non-editable pip install); use package version
    try:
        from proto_tools import __version__
        return f"v{__version__}"
    except ImportError:
        return None


def get_git_info() -> dict[str, Any]:
    """Get git repository information.

    Returns
    -------
    dict
        commit (12-char short hash), branch name, dirty status, and
        package version (always present as fallback).
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

    Returns
    -------
    dict
        All platform, GPU, and environment information.
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
