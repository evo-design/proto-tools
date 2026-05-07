"""FoldMason binary download and extraction configuration.

Provides platform-specific download URLs and extraction logic for the FoldMason
CLI. Used by the shared install_binary.py utility during venv setup.

Archive structure: foldmason/bin/foldmason (single static binary).
"""

import stat
import tarfile
from pathlib import Path

URLS = {
    ("Darwin", "arm64"): "https://mmseqs.com/foldmason/foldmason-osx-universal.tar.gz",
    ("Darwin", "x86_64"): "https://mmseqs.com/foldmason/foldmason-osx-universal.tar.gz",
    ("Linux", "x86_64"): "https://mmseqs.com/foldmason/foldmason-linux-avx2.tar.gz",
    ("Linux", "arm64"): "https://mmseqs.com/foldmason/foldmason-linux-arm64.tar.gz",
}


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract the FoldMason binary from the release tarball into bin_dir."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Match the binary: foldmason/bin/foldmason
            if len(parts) == 3 and parts[1] == "bin" and member.isfile():
                binary_name = parts[2]
                member.name = binary_name
                tar.extract(member, path=bin_dir)
                dest = bin_dir / binary_name
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                print(f"  Installed: {binary_name}")
