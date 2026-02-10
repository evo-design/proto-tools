"""
MMseqs2 binary download and extraction configuration.

Provides platform-specific download URLs and extraction logic for MMseqs2.
Used by the shared install_binary.py utility during venv setup.

Archive structure: mmseqs/bin/mmseqs (single static binary)
"""

from __future__ import annotations

import stat
import tarfile
from pathlib import Path

URLS = {
    ("Darwin", "arm64"): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-osx-universal.tar.gz",
    ("Darwin", "x86_64"): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-osx-universal.tar.gz",
    ("Linux", "x86_64"): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-avx2.tar.gz",
    ("Linux", "arm64"): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-arm64.tar.gz",
}


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract MMseqs2 binary from the release tarball into bin_dir."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Match the binary: mmseqs/bin/mmseqs
            if len(parts) == 3 and parts[1] == "bin" and member.isfile():
                binary_name = parts[2]
                member.name = binary_name  # flatten to just the filename
                tar.extract(member, path=bin_dir)
                dest = bin_dir / binary_name
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                print(f"  Installed: {binary_name}")
