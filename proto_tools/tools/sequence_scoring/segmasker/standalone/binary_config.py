"""
Segmasker binary download and extraction configuration.

Segmasker is distributed as part of the NCBI BLAST+ suite. This config
downloads the full BLAST+ package and extracts only the segmasker binary.

Used by the shared install_binary.py utility during venv setup.

Archive structure: ncbi-blast-2.17.0+/bin/{segmasker,...}
"""

from __future__ import annotations

import stat
import tarfile
from pathlib import Path

URLS = {
    ("Darwin", "arm64"): "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-aarch64-macosx.tar.gz",
    ("Darwin", "x86_64"): "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-macosx.tar.gz",
    ("Linux", "x86_64"): "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz",
}

# Only extract segmasker from the BLAST+ archive
_BINARIES_TO_EXTRACT = {"segmasker"}


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract segmasker binary from the NCBI BLAST+ tarball into bin_dir."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Match files in the bin/ subdirectory
            if (
                len(parts) == 3
                and parts[1] == "bin"
                and member.isfile()
                and parts[2] in _BINARIES_TO_EXTRACT
            ):
                binary_name = parts[2]
                member.name = binary_name  # flatten to just the filename
                tar.extract(member, path=bin_dir)
                dest = bin_dir / binary_name
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                print(f"  Installed: {binary_name}")
