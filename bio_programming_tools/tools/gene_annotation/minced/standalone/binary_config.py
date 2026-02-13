"""
MinCED binary download and extraction configuration.

MinCED is a Java tool (platform-independent JAR), so a single URL
serves all platforms. The extract function copies the JAR and generates
a shell wrapper script that invokes it via `java -jar`.

Used by the shared install_binary.py utility during venv setup.
"""

from __future__ import annotations

import stat
from pathlib import Path

_VERSION = "0.4.2"
_BASE_URL = f"https://github.com/ctSkennerton/minced/releases/download/{_VERSION}"

# Platform-independent — same JAR for all platforms (requires Java).
URLS = {
    ("Darwin", "arm64"): f"{_BASE_URL}/minced.jar",
    ("Darwin", "x86_64"): f"{_BASE_URL}/minced.jar",
    ("Linux", "x86_64"): f"{_BASE_URL}/minced.jar",
    ("Linux", "arm64"): f"{_BASE_URL}/minced.jar",
}

# Shell wrapper that invokes the JAR via java.
_WRAPPER_SCRIPT = """\
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
exec java -jar "$DIR/minced.jar" "$@"
"""


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Install MinCED JAR and shell wrapper into bin_dir.

    The downloaded file is the JAR itself (not an archive). We copy it
    and generate a minimal shell wrapper alongside it.
    """
    import shutil

    # Copy the JAR
    jar_dest = bin_dir / "minced.jar"
    shutil.copy2(archive_path, jar_dest)
    print(f"  Installed: minced.jar")

    # Generate the shell wrapper
    wrapper_dest = bin_dir / "minced"
    wrapper_dest.write_text(_WRAPPER_SCRIPT)
    wrapper_dest.chmod(wrapper_dest.stat().st_mode | stat.S_IEXEC)
    print(f"  Installed: minced (wrapper)")
