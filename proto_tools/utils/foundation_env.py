"""Foundation env host-probe helpers.

The foundation env (a shared micromamba env with git, curl, make, cmake,
pkg-config, gcc, gxx) is provisioned by ``ToolInstance._ensure_foundation_env``
only when the host can't satisfy those dependencies on its own. This module
owns the probe logic and the minimum-version constant that decides "satisfies".
"""

import logging
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

# Minimum gcc/g++ major version accepted on the host before the foundation
# env is skipped. Picked to match common modern Linux defaults (gcc 11 ships
# with ubuntu 22.04) while still rejecting ancient hosts (e.g. CentOS 7's
# 4.8.5) whose compilers fail to build modern C++ tools.
MIN_FOUNDATION_GCC = 11


def host_has_foundation_tools() -> bool:
    """Return True if the host already provides git, curl, make, cmake, pkg-config, and gcc/g++ >= MIN_FOUNDATION_GCC.

    Probes ``shutil.which`` for each binary and parses the major version
    from ``gcc --version`` / ``g++ --version``. Returns False (and logs the
    reason) on any of: missing binary, gcc/g++ too old, or unparseable
    version output.

    Returns:
        bool: True iff the host satisfies the foundation env contract.
    """
    for tool in ("git", "curl", "make", "cmake", "pkg-config"):
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
