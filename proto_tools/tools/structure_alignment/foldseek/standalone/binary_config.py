"""Foldseek binary download and extraction configuration.

Provides platform-specific download URLs and extraction logic for the Foldseek
CLI. Used by the shared install_binary.py utility during venv setup.

Archive structure: foldseek/bin/foldseek (single static binary).
"""

import stat
import tarfile
from pathlib import Path

# Pinned to the v10 tagged GitHub release (Jan 2025). The mmseqs.com tarballs
# are continuous master-HEAD builds and currently ship a regression that heap-
# corrupts inside easy-multimercluster on mixed-chain-count input sets; v10
# handles the same inputs cleanly. Tracked upstream:
# https://github.com/steineggerlab/foldseek/issues/584
_FOLDSEEK_RELEASE_TAG = "10-941cd33"
_RELEASE_URL = f"https://github.com/steineggerlab/foldseek/releases/download/{_FOLDSEEK_RELEASE_TAG}"

URLS = {
    ("Darwin", "arm64"): f"{_RELEASE_URL}/foldseek-osx-universal.tar.gz",
    ("Darwin", "x86_64"): f"{_RELEASE_URL}/foldseek-osx-universal.tar.gz",
    ("Linux", "x86_64"): f"{_RELEASE_URL}/foldseek-linux-avx2.tar.gz",
    ("Linux", "arm64"): f"{_RELEASE_URL}/foldseek-linux-arm64.tar.gz",
}


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract the Foldseek binary from the release tarball into bin_dir."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Match the binary: foldseek/bin/foldseek
            if len(parts) == 3 and parts[1] == "bin" and member.isfile():
                binary_name = parts[2]
                member.name = binary_name
                tar.extract(member, path=bin_dir)
                dest = bin_dir / binary_name
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                print(f"  Installed: {binary_name}")
