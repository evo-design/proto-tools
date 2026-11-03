"""Foundation env host-probe helpers.

The foundation env (a shared micromamba env with git, curl, make, cmake,
pkg-config, binutils, gcc, gxx) is provisioned by
``ToolInstance._ensure_foundation_env`` only when the host can't satisfy those
dependencies on its own. This module owns the tool list, the probe logic, and the
minimum-version constant that decides "satisfies".
"""

import functools
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum gcc/g++ major version accepted on the host before the foundation
# env is skipped. Picked to match common modern Linux defaults (gcc 11 ships
# with ubuntu 22.04) while still rejecting ancient hosts (e.g. CentOS 7's
# 4.8.5) whose compilers fail to build modern C++ tools.
MIN_FOUNDATION_GCC = 11

# Binaries setup scripts may call, and the conda-forge package providing each. gcc/g++
# are probed separately for their version. ``ar`` is here because CMake needs it to
# link static libraries (AlphaFold3's build) and the gcc package no longer brings it.
FOUNDATION_TOOLS: dict[str, str] = {
    "git": "git",
    "curl": "curl",
    "make": "make",
    "cmake": "cmake",
    "pkg-config": "pkg-config",
    "ar": "binutils",
}


def host_has_foundation_tools() -> bool:
    """Return True if the host provides every ``FOUNDATION_TOOLS`` binary and gcc/g++ >= MIN_FOUNDATION_GCC.

    The foundation env is skipped only when nothing is missing. Probes
    ``shutil.which`` for each binary and parses the major version
    from ``gcc --version`` / ``g++ --version``. Returns False (and logs the
    reason) on any of: missing binary, gcc/g++ too old, or unparseable
    version output.

    Returns:
        bool: True iff the host satisfies the foundation env contract.
    """
    for tool in FOUNDATION_TOOLS:
        if shutil.which(tool) is None:
            logger.debug("Foundation env probe: %s not on PATH", tool)
            return False

    for compiler in ("gcc", "g++"):
        path = shutil.which(compiler)
        if path is None:
            logger.debug("Foundation env probe: %s not on PATH", compiler)
            return False
        try:
            proc = subprocess.run([path, "--version"], check=True, capture_output=True, text=True, timeout=10)
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug("Foundation env probe: %s --version failed: %s", compiler, e)
            return False
        match = re.search(r"\b(\d+)\.\d+(?:\.\d+)?\b", proc.stdout)
        if not match:
            logger.debug("Foundation env probe: cannot parse %s version from %r", compiler, proc.stdout)
            return False
        major = int(match.group(1))
        if major < MIN_FOUNDATION_GCC:
            logger.debug(
                "Foundation env probe: %s major version %d < %d minimum",
                compiler,
                major,
                MIN_FOUNDATION_GCC,
            )
            return False

    return True


def missing_foundation_packages(foundation_path: Path) -> list[str]:
    """Return the packages whose ``FOUNDATION_TOOLS`` binary is absent from a built foundation env.

    An env created before a tool joined ``FOUNDATION_TOOLS`` (``ar`` was added later)
    lacks it; ``ToolInstance._ensure_foundation_env`` installs these into the env.

    Args:
        foundation_path (Path): Root of the foundation env.

    Returns:
        list[str]: conda-forge package names to install, empty when the env is complete.
    """
    bin_dir = foundation_path / "bin"
    return [package for tool, package in FOUNDATION_TOOLS.items() if not (bin_dir / tool).exists()]


def foundation_env_path() -> Path:
    """Return the foundation env root, ``PROTO_HOME/.foundation_env/``.

    Returns:
        Path: Foundation env root (it may not exist yet).
    """
    from proto_tools.utils.proto_home import get_proto_home

    return get_proto_home() / ".foundation_env"


@functools.cache
def _host_has_foundation_tools_cached() -> bool:
    """Return ``host_has_foundation_tools()``, probed once per process."""
    return host_has_foundation_tools()


def active_foundation_bin() -> Path | None:
    """Return the foundation env's ``bin/`` when tool subprocesses should use it, else ``None``.

    Tool workers need the foundation tools at run time as well as during setup:
    Triton compiles a small C launcher with ``gcc`` when it initializes, and
    ``torch.compile`` JIT-compiles C++. This mirrors the decision
    ``ToolInstance._ensure_foundation_env`` makes, without creating anything: the
    env must already be built, and the host must not satisfy the contract itself
    (unless ``PROTO_USE_FOUNDATION_ENV=1``). ``PROTO_USE_FOUNDATION_ENV=0`` disables it.

    Returns:
        Path | None: ``PROTO_HOME/.foundation_env/bin`` when in use, else ``None``.
    """
    override = os.environ.get("PROTO_USE_FOUNDATION_ENV")
    if override == "0":
        return None
    root = foundation_env_path()
    if not (root / ".ready").exists():
        return None
    if override != "1" and _host_has_foundation_tools_cached():
        return None
    return root / "bin"
