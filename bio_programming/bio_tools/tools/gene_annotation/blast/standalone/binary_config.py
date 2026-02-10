"""
BLAST+ binary download and extraction configuration.

Provides platform-specific download URLs and extraction logic for BLAST+.
Used by the shared install_binary.py utility during venv setup.

Archive structure: ncbi-blast-2.17.0+/bin/{blastn,blastp,blastx,tblastn,tblastx,makeblastdb,...}
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


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract BLAST+ binaries from the NCBI tarball into bin_dir."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # Match files in the bin/ subdirectory (e.g., ncbi-blast-2.17.0+/bin/blastn)
            if len(parts) == 3 and parts[1] == "bin" and member.isfile():
                binary_name = parts[2]
                member.name = binary_name  # flatten to just the filename
                tar.extract(member, path=bin_dir)
                dest = bin_dir / binary_name
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                print(f"  Installed: {binary_name}")
