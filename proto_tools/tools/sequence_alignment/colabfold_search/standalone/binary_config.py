"""MMseqs2 GPU binary download and extraction configuration for ColabFold search.

Uses the GPU-capable MMseqs2 binary which supports both GPU-accelerated and CPU-only
searches. The GPU binary is a superset of the CPU binary (148 MB vs 17 MB).
"""

import stat
import tarfile
from pathlib import Path

URLS = {
    ("Darwin", "arm64"): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-osx-universal.tar.gz",
    (
        "Darwin",
        "x86_64",
    ): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-osx-universal.tar.gz",
    ("Linux", "x86_64"): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz",
    (
        "Linux",
        "arm64",
    ): "https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-gpu-arm64.tar.gz",
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
