"""
MAFFT binary download and extraction configuration.

Provides platform-specific download URLs and extraction logic for MAFFT.
Used by the shared install_binary.py utility during venv setup.

MAFFT binaries are downloaded from the official MAFFT website.
"""

from __future__ import annotations

import stat
import tarfile
from pathlib import Path

# Official MAFFT pre-compiled binaries (version 7.526)
# From https://mafft.cbrc.jp/alignment/software/
URLS = {
    # macOS (universal build works for both ARM and x86_64)
    ("Darwin", "arm64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.526-mac.zip",
    ("Darwin", "x86_64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.526-mac.zip",
    # Linux x86_64
    ("Linux", "x86_64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.526-linux.tgz",
}


def extract(archive_path: Path, bin_dir: Path) -> None:
    """
    Extract MAFFT binaries from the official tarball/zip into bin_dir.

    MAFFT requires both bin/ and libexec/ directories. The mafft script
    needs to find libexec/ binaries, so we extract the entire mafftdir structure.

    The archive structure is:
    - Linux: mafft-linux64/mafftdir/{bin/, libexec/}
    - macOS: mafft-mac/mafftdir/{bin/, libexec/}
    """
    import shutil
    import zipfile

    # Create libexec directory next to bin
    libexec_dir = bin_dir.parent / "libexec"
    libexec_dir.mkdir(exist_ok=True)

    # Check if it's a zip file (macOS) or tar.gz (Linux)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                parts = Path(member).parts
                if len(parts) < 2:
                    continue

                # Extract files from mafftdir/bin/ to bin/
                if parts[-2] == "bin" and not member.endswith("/"):
                    binary_name = parts[-1]
                    zf.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member
                    dest = bin_dir / binary_name
                    src.replace(dest)
                    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                    print(f"  Installed bin/{binary_name}")

                # Extract files from mafftdir/libexec/ to libexec/
                elif parts[-2] == "libexec" and not member.endswith("/"):
                    binary_name = parts[-1]
                    zf.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member
                    dest = libexec_dir / binary_name
                    src.replace(dest)
                    # Make executable if it's a binary
                    if not binary_name.endswith((".fa", ".1", ".pl")):
                        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
    else:
        # Linux tar.gz
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue

                # Extract files from mafftdir/bin/ to bin/
                if parts[-2] == "bin" and member.isfile():
                    binary_name = parts[-1]
                    tar.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member.name
                    dest = bin_dir / binary_name
                    src.replace(dest)
                    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                    print(f"  Installed bin/{binary_name}")

                # Extract files from mafftdir/libexec/ to libexec/
                elif parts[-2] == "libexec" and member.isfile():
                    binary_name = parts[-1]
                    tar.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member.name
                    dest = libexec_dir / binary_name
                    src.replace(dest)
                    # Make executable if it's a binary
                    if not binary_name.endswith((".fa", ".1", ".pl")):
                        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

    # Clean up extracted temp directories
    for item in bin_dir.parent.iterdir():
        if item.is_dir() and item.name.startswith("mafft-"):
            shutil.rmtree(item)

    print(f"  MAFFT binaries and libexec installed successfully")
