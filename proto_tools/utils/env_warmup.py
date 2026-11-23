"""proto_tools/utils/env_warmup.py.

Pre-read a tool environment's shared libraries when page faults on its filesystem are slow.

``import torch`` memory-maps the environment's ``.so`` files rather than reading them, so the
bytes arrive as page faults. Where those faults are slow, a 4 GB environment takes minutes to
import. Measured as sequential mmap page faults over a cold stretch of ``libtorch_cuda.so``:

=========================  ==================
storage                    fault throughput
=========================  ==================
Lustre                     0.3 - 2.9 MB/s
WekaFS                     170 - 620 MB/s
local ext4                 ~2,200 MB/s
already in the page cache  ~15,000 MB/s
=========================  ==================

On Lustre a plain sequential read of the same files runs at ~1.3 GB/s, and it fills the same
page cache the faults would. Pre-reading 5.4 GB took 5 seconds and moved a worker that had
been stalled for 27 minutes straight into compute. On WekaFS the same pre-read costs 6-8
seconds to save about one. Two measurements decide whether warm-up is useful.

* Is the environment already warm? :func:`_resident_fraction` asks the kernel with ``mincore``,
  which reads nothing. At ``_WARM_FRACTION`` or more of the libraries' bytes resident, the
  warm-up stops there. Below it, a re-read fetches only what is missing, because the resident
  pages come from memory, so an env that was half evicted costs half a warm-up.
* Is this filesystem slow to fault? :func:`_fault_throughput` times sequential page faults for
  at most ``_PROBE_SECONDS``, and only a throughput below ``_SLOW_FAULT_BYTES_PER_SECOND``
  triggers a read. It probes the library with the most uncached pages, not the largest, which
  another env sharing it by hardlink may have cached, and times that library's longest stretch
  of uncached pages. Reusing a window touched by an earlier probe would measure
  cached reads and incorrectly classify a cold environment as fast.
* The libraries to read are listed once per build and kept in the environment, so a process
  that does not need to warm pays for one small file read and the probe.
* Reading is bounded by a wall-clock budget and a concurrency cap. The caller warms under the
  environment's build lock, so processes starting together on a node warm it
  once and the rest find it cached.
* Failures are caught so an unsuccessful warm-up does not prevent a run.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

#: Only files at least this large are worth a sequential read; the handful of big CUDA
#: libraries are nearly all of the bytes, and opening hundreds of small ones is pure overhead.
_MIN_INTERESTING_BYTES = 16 * 2**20

#: Warm when page faults run slower than this. Lustre measures at most 2.9 MB/s and WekaFS at
#: least 170 MB/s, so this sits a factor of five or more from both.
_SLOW_FAULT_BYTES_PER_SECOND = 32 * 2**20

#: Treat the env as warm when at least this share of its libraries' bytes is resident.
_WARM_FRACTION = 0.9

#: How long the fault probe may take, and how much it may fault in.
_PROBE_SECONDS = 0.1
_PROBE_BYTES = 16 * 2**20

#: Stop after this budget so warm-up cannot delay startup indefinitely.
_DEFAULT_BUDGET_SECONDS = 30.0

#: Limit parallel readers to keep the filesystem busy without excessive concurrent reads.
_READERS = 8

#: Read size per request. Large enough that the syscall overhead disappears.
_CHUNK_BYTES = 4 * 2**20

#: The env's large libraries, listed once per build: the fingerprint, then paths relative to the env.
_LIBRARY_LIST = "warm_libraries.txt"

_WARM_ENV_VAR = "PROTO_WARM_ENV"


def _forced() -> bool | None:
    """Read the warm-up override, returning ``None`` to choose based on measurements."""
    raw = os.environ.get(_WARM_ENV_VAR, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _scan_libraries(env_path: Path, deadline: float) -> list[Path] | None:
    """Find large shared libraries to pre-read, largest first, or return ``None`` on timeout.

    Walks ``lib/``, which holds both ``site-packages`` and the libraries a conda env installs
    beside it. Symlinks are skipped, to files (``libfoo.so -> libfoo.so.1``, whose target is
    found on its own) and to directories (conda's ``lib/python3.1 -> python3.12``), so no
    library is read twice.
    """
    found: list[tuple[int, Path]] = []
    for directory, _, names in os.walk(env_path / "lib"):
        if time.monotonic() > deadline:
            return None
        for name in names:
            if ".so" not in name:
                continue
            path = Path(directory) / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            if not path.is_symlink() and stat.st_size >= _MIN_INTERESTING_BYTES:
                found.append((stat.st_size, path))
    return [path for _, path in sorted(found, reverse=True)]


def _large_shared_libraries(env_path: Path, fingerprint: str, deadline: float) -> list[Path]:
    """Return large libraries to pre-read, reusing the saved list if it is still current.

    The list is written on first use and reused while ``fingerprint`` matches and every library
    on it still exists, so the walk is paid once per build rather than once per process. An env
    that cannot be written to, such as a team's read-only one, is walked every time instead.
    """
    listing = env_path / _LIBRARY_LIST
    try:
        header, *rows = listing.read_text().splitlines()
        listed = [env_path / row for row in rows]
        if header == fingerprint and all(path.is_file() for path in listed):
            return listed
    except (OSError, ValueError):
        pass
    libraries = _scan_libraries(env_path, deadline)
    if libraries is None:
        return []
    staged = listing.with_name(f".{_LIBRARY_LIST}.{os.getpid()}")
    try:
        staged.write_text("\n".join([fingerprint, *(str(path.relative_to(env_path)) for path in libraries)]))
        staged.replace(listing)
    except OSError as error:
        logger.debug("env warm-up: could not record the library list in %s: %s", env_path, error)
        staged.unlink(missing_ok=True)
    return libraries


def _page_residency(path: Path) -> bytes | None:
    """Return one byte per file page, with the low bit set for cached pages.

    Ask the kernel with ``mincore`` without reading the file. Return ``None``
    outside Linux and macOS or when ``ctypes`` cannot reach libc.
    """
    size = path.stat().st_size
    if size == 0:
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    descriptor = os.open(path, os.O_RDONLY)
    try:
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
        if address is None or address == ctypes.c_void_p(-1).value:
            return None
        try:
            pages = (ctypes.c_ubyte * ((size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
            return bytes(pages) if libc.mincore(address, size, pages) == 0 else None
        finally:
            libc.munmap(address, size)
    finally:
        os.close(descriptor)


def _residency(libraries: list[Path]) -> dict[Path, bytes] | None:
    """Read each library's page residency, returning ``None`` if any cannot be measured."""
    residency = {}
    for path in libraries:
        try:
            pages = _page_residency(path)
        except (OSError, AttributeError):
            pages = None
        if pages is None:
            return None
        residency[path] = pages
    return residency


def _resident_fraction(residency: dict[Path, bytes] | None) -> float:
    """Return the fraction of library pages in the page cache, or ``0.0`` if unknown."""
    if not residency:
        return 0.0
    resident = sum(page & 1 for pages in residency.values() for page in pages)
    return resident / sum(len(pages) for pages in residency.values())


def _probe_window(pages: bytes) -> tuple[int, int] | None:
    """Find the longest uncached page range, or return ``None`` if all pages are cached."""
    best: tuple[int, int] | None = None
    start = None
    for index, page in enumerate([*pages, 1]):
        if not page & 1:
            start = index if start is None else start
        elif start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def _fault_throughput(path: Path, pages: bytes | None) -> float:
    """Bytes per second that sequential page faults on ``path`` deliver, over uncached pages.

    Starts at the longest run of ``pages`` not in the page cache, or at the file's middle where
    residency is unknown (``None``), and stops after ``_PROBE_BYTES``, the run's end, or
    ``_PROBE_SECONDS``, so a slow filesystem is recognised within that time rather than measured
    to completion.
    """
    window = _probe_window(pages) if pages is not None else None
    if pages is not None and window is None:
        return float("inf")  # every page is cached, so faults cost nothing
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        if window is None:
            start, end = len(mapped) // 2 // mmap.PAGESIZE * mmap.PAGESIZE, len(mapped)
        else:
            start, end = window[0] * mmap.PAGESIZE, min(len(mapped), window[1] * mmap.PAGESIZE)
        end = min(end, start + _PROBE_BYTES)
        offset = start
        began = time.perf_counter()
        while offset < end:
            mapped[offset]
            offset += mmap.PAGESIZE
            if time.perf_counter() - began > _PROBE_SECONDS:
                break
        elapsed = max(time.perf_counter() - began, 1e-9)
    return (offset - start) / elapsed


def _read_whole(path: Path, deadline: float) -> int:
    """Read into the page cache until ``deadline`` and return the bytes read."""
    total = 0
    try:
        with path.open("rb", buffering=0) as handle:
            while time.monotonic() < deadline:
                chunk = handle.read(_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
    except OSError as error:  # skip libraries that become unavailable during warm-up
        logger.debug("env warm-up: could not read %s: %s", path, error)
    return total


def warm_environment(
    env_path: Path,
    fingerprint: str,
    *,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> int:
    """Pre-read large shared libraries into the page cache and return the bytes read.

    Does nothing, and returns ``0``, when the libraries are already mostly in the page cache or
    page faults are fast. The probe uses the library with the most uncached pages.
    Environments hardlink shared libraries, so another environment may have
    cached the largest one while the rest remain cold. Set ``PROTO_WARM_ENV``
    to ``1`` or ``0`` to force or skip warm-up without taking either measurement.

    Warm-up failures are caught and leave the environment unchanged.

    Args:
        env_path (Path): The tool environment to warm.
        fingerprint (str): Identifies this build of the environment, so a rebuilt environment's
            libraries are listed again rather than taken from its predecessor's list.
        budget_seconds (float): Stop listing and reading libraries after this long.

    Returns:
        int: Bytes read into the page cache, or ``0`` when warm-up was skipped or failed.
    """
    try:
        forced = _forced()
        if forced is False:
            return 0
        started = time.monotonic()
        deadline = started + budget_seconds
        libraries = _large_shared_libraries(env_path, fingerprint, deadline)
        if not libraries:
            return 0
        if not forced:
            # Skip environments whose libraries are already mostly cached.
            residency = _residency(libraries)
            resident = _resident_fraction(residency)
            if resident >= _WARM_FRACTION:
                logger.debug("env warm-up: %.0f%% of %s is already cached, skipping", resident * 100, env_path)
                return 0
            if residency is None:
                probed, pages = libraries[0], None
            else:
                probed = max(libraries, key=lambda path: sum(not page & 1 for page in residency[path]))
                pages = residency[probed]
            # Measure uncached reads before paying for a full warm-up.
            throughput = _fault_throughput(probed, pages)
            if throughput >= _SLOW_FAULT_BYTES_PER_SECOND:
                logger.debug("env warm-up: page faults in %s run at %.0f MB/s, skipping", env_path, throughput / 2**20)
                return 0
            logger.info(
                "Pre-reading %s's shared libraries: page faults on its filesystem run at %.1f MB/s",
                env_path.name,
                throughput / 2**20,
            )

        # Read the libraries in parallel, with each reader observing the deadline.
        with ThreadPoolExecutor(max_workers=_READERS) as pool:
            total = sum(pool.map(lambda path: _read_whole(path, deadline), libraries))
        logger.debug(
            "env warm-up: read %.1f GB of %d libraries from %s in %.1fs",
            total / 2**30,
            len(libraries),
            env_path,
            time.monotonic() - started,
        )
        return total
    except Exception as error:  # an optimization must never break a run
        logger.debug("env warm-up: skipped after an unexpected error: %s", error)
        return 0
