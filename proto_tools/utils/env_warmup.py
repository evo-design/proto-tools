"""proto_tools/utils/env_warmup.py.

Pre-read a tool environment's shared libraries when page faults on its filesystem are slow.

``import torch`` memory-maps the environment's ``.so`` files rather than reading them, so the
bytes arrive as page faults. Where those faults are slow, a 4 GB environment takes minutes to
import. Measured as sequential mmap page faults over a cold stretch of ``libtorch_cuda.so``:

=======================================  ==================
storage                                  fault throughput
=======================================  ==================
Lustre (Sherlock, Oak and ``$SCRATCH``)  0.3 - 2.9 MB/s
WekaFS (Chimera)                         170 - 620 MB/s
local ext4                               ~2,200 MB/s
already in the page cache                ~15,000 MB/s
=======================================  ==================

On Lustre a plain sequential read of the same files runs at ~1.3 GB/s, and it fills the same
page cache the faults would. Pre-reading 5.4 GB took 5 seconds and moved a worker that had
been stalled for 27 minutes straight into compute. On WekaFS the same pre-read costs 6-8
seconds to save about one, so the decision is made by measuring rather than by filesystem type,
in two separate questions:

* Is the environment already warm? :func:`_resident_fraction` asks the kernel with ``mincore``,
  which reads nothing. At ``_WARM_FRACTION`` or more of the libraries' bytes resident, the
  warm-up stops there. Below it, a re-read fetches only what is missing, because the resident
  pages come from memory, so an env that was half evicted costs half a warm-up.
* Is this filesystem slow to fault? :func:`_fault_throughput` times sequential page faults on
  the largest library for at most ``_PROBE_SECONDS``, over its longest stretch of pages that are
  not cached, and only a throughput below ``_SLOW_FAULT_BYTES_PER_SECOND`` triggers a read.
  Timing uncached pages matters: a probe that re-timed a window an earlier probe had faulted in
  would measure the page cache and call a cold environment fast.
* The libraries to read are listed once per build and kept in the environment, so a process
  that does not need to warm pays for one small file read and the probe.
* Reading is bounded by a wall-clock budget and a concurrency cap.
* Every failure is swallowed. A warm-up that cannot run leaves behavior exactly as it was.
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

#: Give up after this long. A filesystem slow enough to blow this budget would also have made
#: the lazy path miserable, but a warm-up must never be the reason a run fails to start.
_DEFAULT_BUDGET_SECONDS = 30.0

#: Parallel readers. Enough to keep a parallel filesystem busy, few enough that a node-full of
#: ranks does not multiply into a flood.
_READERS = 8

#: Read size per request. Large enough that the syscall overhead disappears.
_CHUNK_BYTES = 4 * 2**20

#: The env's large libraries, listed once per build: the fingerprint, then paths relative to the env.
_LIBRARY_LIST = "warm_libraries.txt"

_WARM_ENV_VAR = "PROTO_WARM_ENV"


def _forced() -> bool | None:
    """Whether ``PROTO_WARM_ENV`` forces warming on or off; ``None`` to decide by measuring."""
    raw = os.environ.get(_WARM_ENV_VAR, "").strip().lower()
    if not raw:
        return None
    return raw not in {"0", "false", "no", "off"}


def _scan_libraries(env_path: Path, deadline: float) -> list[Path] | None:
    """The env's shared libraries worth pre-reading, largest first; ``None`` if out of time.

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
    """The env's libraries worth pre-reading, largest first, from the env's list when current.

    The list is written on first use and reused while ``fingerprint`` matches, so the walk is
    paid once per build rather than once per process. An env that cannot be written to, such
    as a team's read-only one, is walked every time instead.
    """
    listing = env_path / _LIBRARY_LIST
    try:
        header, *rows = listing.read_text().splitlines()
        if header == fingerprint:
            return [env_path / row for row in rows]
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
    """One byte per page of ``path``, odd where the page is in the page cache.

    Asks the kernel with ``mincore``, which reads nothing from the file. ``None`` where that is
    unavailable: outside Linux and macOS, or when ``ctypes`` cannot reach libc. On Linux a
    process that neither owns the file nor may write it is told only about pages it has mapped
    itself, so another user's env reads as uncached and is judged by the fault probe alone.
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


def _resident_fraction(libraries: list[Path]) -> float:
    """The share of the libraries' bytes in the page cache; ``0.0`` where it cannot be known."""
    resident = total = 0
    for path in libraries:
        try:
            pages = _page_residency(path)
        except (OSError, AttributeError):
            pages = None
        if pages is None:
            return 0.0
        resident += sum(page & 1 for page in pages)
        total += len(pages)
    return resident / total if total else 0.0


def _probe_window(pages: bytes) -> tuple[int, int] | None:
    """The longest run of uncached pages as ``(first, past the last)``; ``None`` if all are cached."""
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


def _fault_throughput(path: Path) -> float:
    """Bytes per second that sequential page faults on ``path`` deliver, over uncached pages.

    Starts at the longest run of pages not in the page cache, or at the file's middle where
    residency is unknown, and stops after ``_PROBE_BYTES``, the run's end, or
    ``_PROBE_SECONDS``, so a slow filesystem is recognised within that time rather than measured
    to completion.
    """
    pages = _page_residency(path)
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
    """Read ``path`` sequentially into the page cache, stopping at ``deadline``. Bytes read."""
    total = 0
    try:
        with path.open("rb", buffering=0) as handle:
            while time.monotonic() < deadline:
                chunk = handle.read(_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
    except OSError as error:  # a library removed or unreadable mid-walk is not our problem
        logger.debug("env warm-up: could not read %s: %s", path, error)
    return total


def warm_environment(
    env_path: Path,
    fingerprint: str,
    *,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> int:
    """Pre-read ``env_path``'s large shared libraries into the page cache. Bytes read.

    Does nothing, and returns ``0``, when the libraries are already mostly in the page cache or
    page faults on the largest library's uncached pages are fast. ``PROTO_WARM_ENV`` skips both
    measurements in either direction.

    Never raises. A warm-up that fails for any reason leaves the environment exactly as it was,
    and the caller is no worse off than before this function existed.

    Args:
        env_path (Path): The tool environment to warm.
        fingerprint (str): Identifies this build of the environment, so a rebuilt environment's
            libraries are listed again rather than taken from its predecessor's list.
        budget_seconds (float): Stop listing and reading libraries after this long.

    Returns:
        int: Bytes read into the page cache; ``0`` when warming was skipped or failed.
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
            resident = _resident_fraction(libraries)
            if resident >= _WARM_FRACTION:
                logger.debug("env warm-up: %.0f%% of %s is already cached, skipping", resident * 100, env_path)
                return 0
            throughput = _fault_throughput(libraries[0])
            if throughput >= _SLOW_FAULT_BYTES_PER_SECOND:
                logger.debug("env warm-up: page faults in %s run at %.0f MB/s, skipping", env_path, throughput / 2**20)
                return 0
            logger.info(
                "Pre-reading %s's shared libraries: page faults on its filesystem run at %.1f MB/s",
                env_path.name,
                throughput / 2**20,
            )

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
