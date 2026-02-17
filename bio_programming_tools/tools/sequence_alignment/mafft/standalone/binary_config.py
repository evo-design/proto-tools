"""
MAFFT binary download and extraction configuration.

Provides platform-specific download URLs and extraction logic for MAFFT.
Used by the shared install_binary.py utility during venv setup.

Pre-built binaries are used on x86_64 Linux and macOS. On aarch64 Linux
(no official pre-built binary), the source tarball is downloaded and
compiled with `make`.
"""

from __future__ import annotations

import platform
import stat
import tarfile
from pathlib import Path

# Official MAFFT pre-compiled binaries (version 7.526) and source (7.525)
# From https://mafft.cbrc.jp/alignment/software/
URLS = {
    # macOS (universal build works for both ARM and x86_64)
    ("Darwin", "arm64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.526-mac.zip",
    ("Darwin", "x86_64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.526-mac.zip",
    # Linux x86_64 (pre-built)
    ("Linux", "x86_64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.526-linux.tgz",
    # Linux aarch64 — compile from source (no official pre-built binary)
    ("Linux", "arm64"): "https://mafft.cbrc.jp/alignment/software/mafft-7.525-without-extensions-src.tgz",
}

_IS_SOURCE_ARCHIVE = {("Linux", "arm64")}


def _extract_prebuilt(archive_path: Path, bin_dir: Path) -> None:
    """Extract pre-built MAFFT binaries from the official tarball/zip.

    Archive structure:
    - Linux: mafft-linux64/mafftdir/{bin/, libexec/}
    - macOS: mafft-mac/mafftdir/{bin/, libexec/}
    """
    import shutil
    import zipfile

    libexec_dir = bin_dir.parent / "libexec"
    libexec_dir.mkdir(exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.namelist():
                parts = Path(member).parts
                if len(parts) < 2:
                    continue

                if parts[-2] == "bin" and not member.endswith("/"):
                    binary_name = parts[-1]
                    zf.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member
                    dest = bin_dir / binary_name
                    src.replace(dest)
                    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                    print(f"  Installed bin/{binary_name}")

                elif parts[-2] == "libexec" and not member.endswith("/"):
                    binary_name = parts[-1]
                    zf.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member
                    dest = libexec_dir / binary_name
                    src.replace(dest)
                    if not binary_name.endswith((".fa", ".1", ".pl")):
                        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
    else:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2:
                    continue

                if parts[-2] == "bin" and member.isfile():
                    binary_name = parts[-1]
                    tar.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member.name
                    dest = bin_dir / binary_name
                    src.replace(dest)
                    dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                    print(f"  Installed bin/{binary_name}")

                elif parts[-2] == "libexec" and member.isfile():
                    binary_name = parts[-1]
                    tar.extract(member, path=bin_dir.parent)
                    src = bin_dir.parent / member.name
                    dest = libexec_dir / binary_name
                    src.replace(dest)
                    if not binary_name.endswith((".fa", ".1", ".pl")):
                        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

    for item in bin_dir.parent.iterdir():
        if item.is_dir() and item.name.startswith("mafft-"):
            shutil.rmtree(item)

    print("  MAFFT binaries and libexec installed successfully")


def _extract_from_source(archive_path: Path, bin_dir: Path) -> None:
    """Compile MAFFT from source and install into the venv."""
    import subprocess
    import tempfile

    prefix = bin_dir.parent

    with tempfile.TemporaryDirectory() as build_dir:
        build_path = Path(build_dir)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=build_path)

        # Find the extracted source directory
        src_dirs = [d for d in build_path.iterdir() if d.is_dir()]
        if len(src_dirs) != 1:
            raise RuntimeError(
                f"Expected one source directory, found: {src_dirs}"
            )
        core_dir = src_dirs[0] / "core"

        print(f"  Compiling MAFFT from source in {core_dir}...")
        subprocess.check_call(
            [
                "make", "clean",
                f"PREFIX={prefix}",
                f"BINDIR={bin_dir}",
                f"LIBDIR={prefix / 'libexec' / 'mafft'}",
            ],
            cwd=core_dir,
        )
        subprocess.check_call(
            [
                "make", "-j4",
                f"PREFIX={prefix}",
                f"BINDIR={bin_dir}",
                f"LIBDIR={prefix / 'libexec' / 'mafft'}",
            ],
            cwd=core_dir,
        )
        subprocess.check_call(
            [
                "make", "install",
                f"PREFIX={prefix}",
                f"BINDIR={bin_dir}",
                f"LIBDIR={prefix / 'libexec' / 'mafft'}",
            ],
            cwd=core_dir,
        )

    print("  MAFFT compiled and installed successfully")


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract or compile MAFFT into bin_dir."""
    arch = platform.machine()
    system = platform.system()
    # Normalize arch the same way install_binary.py does
    arch_aliases = {"aarch64": "arm64", "AMD64": "x86_64"}
    arch = arch_aliases.get(arch, arch)

    if (system, arch) in _IS_SOURCE_ARCHIVE:
        _extract_from_source(archive_path, bin_dir)
    else:
        _extract_prebuilt(archive_path, bin_dir)
