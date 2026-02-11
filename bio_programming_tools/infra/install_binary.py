"""
Shared utility for downloading platform-specific bioinformatics binaries into a venv.

Called from standalone/setup.sh scripts during EnvManager venv creation.
Each tool provides its own `binary_config.py` in its standalone/ directory with:
    URLS: dict mapping (system, machine) tuples to download URLs
    extract(archive_path, bin_dir): function to extract binaries from the archive

This script discovers and loads the tool's config automatically.

Usage:
    python install_binary.py <tool_name>
"""

from __future__ import annotations

import importlib.util
import platform
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

# Canonical architecture names. Tool binary_config.py files should use these
# in their URLS dicts. Raw platform.machine() values are normalized to these
# before lookup so that e.g. Linux "aarch64" matches a ("Linux", "arm64") key.
_ARCH_ALIASES: dict[str, str] = {
    "x86_64": "x86_64",
    "AMD64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


def _find_tool_config(tool_name: str) -> Path:
    """Find a tool's binary_config.py by scanning for standalone/ directories.

    Uses the same discovery pattern as EnvManager._determine_valid_model_name().
    """
    tools_dir = Path(__file__).parent.parent  # infra's parent = tools

    for item in tools_dir.rglob("*"):
        if (
            item.is_dir()
            and item.name == tool_name
            and (item / "standalone").exists()
        ):
            config_path = item / "standalone" / "binary_config.py"
            if config_path.exists():
                return config_path

    raise ValueError(
        f"No binary_config.py found for tool: {tool_name}. "
        f"Expected at: <tool_dir>/standalone/binary_config.py"
    )


def _load_tool_config(config_path: Path):
    """Dynamically load a tool's binary_config module."""
    spec = importlib.util.spec_from_file_location("binary_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Validate the module has required attributes
    if not hasattr(mod, "URLS"):
        raise AttributeError(f"{config_path} must define a URLS dict")
    if not hasattr(mod, "extract"):
        raise AttributeError(f"{config_path} must define an extract(archive_path, bin_dir) function")

    return mod


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a tqdm progress bar, falling back to simple logging."""
    try:
        from tqdm import tqdm

        response = urllib.request.urlopen(url)
        total = int(response.headers.get("Content-Length", 0))

        with (
            tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading", file=sys.stdout) as pbar,
            open(dest, "wb") as f,
        ):
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    except ImportError:
        # Fallback: simple percentage-based progress
        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Downloading: {mb:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

        urllib.request.urlretrieve(url, dest, _reporthook)
        print()  # newline after progress


def install_binary(tool_name: str) -> None:
    """Download and install a platform-specific binary into the active venv.

    Discovers the tool's binary_config.py, reads its URLS and extract function,
    downloads the appropriate archive for the current platform, and extracts
    binaries into the venv's bin/ directory.
    """
    config_path = _find_tool_config(tool_name)
    config = _load_tool_config(config_path)

    system = platform.system()
    raw_machine = platform.machine()
    machine = _ARCH_ALIASES.get(raw_machine, raw_machine)
    key = (system, machine)

    if key in config.URLS:
        url = config.URLS[key]
    else:
        # Fallback: try any available binary for this OS.
        # Covers cases like macOS ARM falling back to x86_64 via Rosetta 2.
        os_matches = {k: v for k, v in config.URLS.items() if k[0] == system}
        if os_matches:
            fallback_key = next(iter(os_matches))
            warnings.warn(
                f"No native {machine} binary for {tool_name}, "
                f"falling back to {fallback_key[1]}."
            )
            key = fallback_key
            url = os_matches[fallback_key]
        else:
            supported = ", ".join(f"{s}/{m}" for s, m in config.URLS)
            raise RuntimeError(
                f"No {tool_name} binary available for {system}/{machine} "
                f"(raw arch: {raw_machine}). "
                f"Supported platforms: {supported}"
            )

    bin_dir = Path(sys.executable).parent

    print(f"Installing {tool_name} for {key[0]}/{key[1]}...")

    with tempfile.TemporaryDirectory() as tmp:
        filename = url.split("/")[-1]
        archive_path = Path(tmp) / filename

        _download_with_progress(url, archive_path)
        print(f"  Download complete ({archive_path.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"  Extracting binaries to {bin_dir}...")

        config.extract(archive_path, bin_dir)

    print(f"{tool_name} installation complete!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <tool_name>", file=sys.stderr)
        sys.exit(1)

    install_binary(sys.argv[1])
