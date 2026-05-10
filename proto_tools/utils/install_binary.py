"""proto_tools/utils/install_binary.py.

Called from standalone/setup.sh scripts during ToolInstance venv creation.
Each tool provides its own `binary_config.py` in its standalone/ directory with:
    URLS: dict mapping (system, machine) tuples to download URLs
    extract(archive_path, bin_dir): function to extract binaries from the archive

This script discovers and loads the tool's config automatically.

Usage:
    python install_binary.py <toolkit>
"""

import importlib.util
import platform
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any

_MAX_DOWNLOAD_RETRIES = 5
_INITIAL_RETRY_DELAY_SECONDS = 5.0
_BACKOFF_MULTIPLIER = 2.0
_MAX_RETRY_DELAY_SECONDS = 60.0
_SOCKET_TIMEOUT_SECONDS = 60.0

# Canonical architecture names. Tool binary_config.py files should use these
# in their URLS dicts. Raw platform.machine() values are normalized to these
# before lookup so that e.g. Linux "aarch64" matches a ("Linux", "arm64") key.
_ARCH_ALIASES: dict[str, str] = {
    "x86_64": "x86_64",
    "AMD64": "x86_64",
    "arm64": "arm64",
    "aarch64": "arm64",
}


def _find_tool_config(toolkit: str) -> Path:
    """Find a tool's binary_config.py by scanning for standalone/ directories.

    Uses the same discovery pattern as ToolInstance._determine_valid_model_name().

    Args:
        toolkit (str): Name of the tool directory to search for.
    """
    tools_dir = Path(__file__).parent.parent / "tools"  # utils/ -> proto_tools/ -> tools/

    for item in tools_dir.rglob("*"):
        if item.is_dir() and item.name == toolkit and (item / "standalone").exists():
            config_path = item / "standalone" / "binary_config.py"
            if config_path.exists():
                return config_path

    raise ValueError(
        f"No binary_config.py found for tool: {toolkit}. Expected at: <tool_dir>/standalone/binary_config.py"
    )


def _load_tool_config(config_path: Path) -> Any:
    """Dynamically load a tool's binary_config module."""
    spec = importlib.util.spec_from_file_location("binary_config", config_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Validate the module has required attributes
    if not hasattr(mod, "URLS"):
        raise AttributeError(f"{config_path} must define a URLS dict")
    if not hasattr(mod, "extract"):
        raise AttributeError(f"{config_path} must define an extract(archive_path, bin_dir) function")

    return mod


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` with a progress bar; raise if truncated.

    A per-read socket timeout makes stalled connections fail fast so the
    retry loop in ``install_binary`` can take over. Compares the final size
    to ``Content-Length`` so silent truncation surfaces as a retryable error.

    Args:
        url (str): URL to download from.
        dest (Path): Local file path to save the download to.
    """
    try:
        from tqdm import tqdm

        tqdm_cls: Any = tqdm
    except ImportError:
        tqdm_cls = None

    with urllib.request.urlopen(url, timeout=_SOCKET_TIMEOUT_SECONDS) as response:  # noqa: S310 -- URL from trusted config
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -1
        pbar = (
            tqdm_cls(total=total, unit="B", unit_scale=True, desc="  Downloading", file=sys.stdout)
            if tqdm_cls
            else None
        )

        with open(dest, "wb") as f:
            try:
                while chunk := response.read(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
                    elif total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct != last_pct:
                            mb, mb_total = downloaded / 1024 / 1024, total / 1024 / 1024
                            print(f"\r  Downloading: {mb:.1f}/{mb_total:.1f} MB ({pct}%)", end="", flush=True)
                            last_pct = pct
            finally:
                if pbar is not None:
                    pbar.close()
                elif last_pct >= 0:
                    print()

    if total and downloaded != total:
        raise OSError(f"Download truncated from {url}: got {downloaded}/{total} bytes")


def install_binary(toolkit: str) -> None:
    """Download and install a platform-specific binary into the active venv.

    Discovers the tool's binary_config.py, reads its URLS and extract function,
    downloads the appropriate archive for the current platform, and extracts
    binaries into the venv's bin/ directory.

    Args:
        toolkit (str): Name of the tool to install binaries for.
    """
    config_path = _find_tool_config(toolkit)
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
            warnings.warn(f"No native {machine} binary for {toolkit}, falling back to {fallback_key[1]}.", stacklevel=2)
            key = fallback_key
            url = os_matches[fallback_key]
        else:
            supported = ", ".join(f"{s}/{m}" for s, m in config.URLS)
            raise RuntimeError(
                f"{toolkit}: no binary for {system}/{machine} (raw arch {raw_machine!r}); supported: {supported}"
            )

    bin_dir = Path(sys.executable).parent

    print(f"Installing {toolkit} for {key[0]}/{key[1]}...")

    last_error: BaseException | None = None
    for attempt in range(1, _MAX_DOWNLOAD_RETRIES + 1):
        try:
            with tempfile.TemporaryDirectory() as tmp:
                filename = url.split("/")[-1]
                archive_path = Path(tmp) / filename

                _download_with_progress(url, archive_path)
                size_mb = archive_path.stat().st_size / 1024 / 1024
                print(f"  Download complete ({size_mb:.1f} MB)")
                print(f"  Extracting binaries to {bin_dir}...")

                try:
                    config.extract(archive_path, bin_dir)
                except (OSError, EOFError, tarfile.TarError) as e:
                    raise OSError(
                        f"{toolkit}: extract failed for {archive_path} -> {bin_dir}: {type(e).__name__}: {e}"
                    ) from e

            print(f"{toolkit} installation complete!")
            return
        except (OSError, EOFError, tarfile.TarError, urllib.error.URLError) as exc:  # noqa: PERF203 -- retry loop
            last_error = exc
            # Leading \n breaks out of the progress bar's trailing \r so this line survives in CI logs.
            print(f"\n  Download attempt {attempt}/{_MAX_DOWNLOAD_RETRIES} failed: {exc}", flush=True)
            if attempt < _MAX_DOWNLOAD_RETRIES:
                delay = min(
                    _INITIAL_RETRY_DELAY_SECONDS * _BACKOFF_MULTIPLIER ** (attempt - 1), _MAX_RETRY_DELAY_SECONDS
                )
                print(f"  Retrying in {delay:.0f}s...", flush=True)
                time.sleep(delay)

    raise RuntimeError(f"Failed to download {toolkit} after {_MAX_DOWNLOAD_RETRIES} attempts. Last error: {last_error}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <toolkit>", file=sys.stderr)
        sys.exit(1)

    install_binary(sys.argv[1])
