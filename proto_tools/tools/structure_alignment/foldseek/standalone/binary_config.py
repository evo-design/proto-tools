"""Foldseek binary download/extraction config (archive layout: foldseek/bin/foldseek)."""

import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

# Pinned to v10: mmseqs.com master-HEAD builds regress (heap corruption in
# easy-multimercluster); v10 is clean. https://github.com/steineggerlab/foldseek/issues/584
_FOLDSEEK_RELEASE_TAG = "10-941cd33"
_RELEASE_URL = f"https://github.com/steineggerlab/foldseek/releases/download/{_FOLDSEEK_RELEASE_TAG}"

# GPU mode needs an NVIDIA driver at least this new (Ampere full speed, Turing reduced).
_MIN_GPU_DRIVER = (525, 60, 13)


def _should_use_gpu_build() -> bool:
    """Return whether a compatible NVIDIA GPU is present for the Linux x86_64 GPU build."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        try:
            version = tuple(int(p) for p in line.strip().split(".")[:3])
        except ValueError:
            continue
        if version >= _MIN_GPU_DRIVER:
            return True
    return False


URLS = {
    ("Darwin", "arm64"): f"{_RELEASE_URL}/foldseek-osx-universal.tar.gz",
    ("Darwin", "x86_64"): f"{_RELEASE_URL}/foldseek-osx-universal.tar.gz",
    ("Linux", "x86_64"): f"{_RELEASE_URL}/foldseek-linux-{'gpu' if _should_use_gpu_build() else 'avx2'}.tar.gz",
    ("Linux", "arm64"): f"{_RELEASE_URL}/foldseek-linux-arm64.tar.gz",
}


def extract(archive_path: Path, bin_dir: Path) -> None:
    """Extract the Foldseek binary into bin_dir and record the build variant for run.py."""
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[1] == "bin" and member.isfile():
                binary_name = parts[2]
                member.name = binary_name
                tar.extract(member, path=bin_dir)
                dest = bin_dir / binary_name
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
                print(f"  Installed: {binary_name}")
    variant = archive_path.name.removeprefix("foldseek-").removesuffix(".tar.gz")
    (bin_dir / ".foldseek_build_variant").write_text(variant)
