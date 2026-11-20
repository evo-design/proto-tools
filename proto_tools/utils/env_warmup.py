"""proto_tools/utils/env_warmup.py.

Pre-read a tool environment's shared libraries so the first run on a node is not mmap-bound.

``import torch`` memory-maps the environment's ``.so`` files rather than reading them, so the
bytes arrive as page faults: ~4 KB each, synchronous, one after another. On a local disk that
is microseconds per fault and nobody notices. On a parallel filesystem each fault is a network
round-trip, and a 4 GB environment is roughly a million of them in series.

Measured on Sherlock with ``PROTO_HOME`` on Lustre, loading an environment holding 4.0 GB of
CUDA libraries across 161 files:

===========================================  ==============
access pattern                               throughput
===========================================  ==============
mmap page faults                             0.3 - 2.9 MB/s
plain sequential read of the same files      ~1.3 GB/s
===========================================  ==============

Reading a file sequentially populates the same page cache that an mmap fault would, so doing
it up front converts a million small synchronous reads into a handful of large ones. Warming
693 libraries took 5 seconds and moved a worker that had been stalled for 27 minutes straight
into compute.

This is an optimization, never a correctness fix, and it is written to behave that way:

* It only runs where it pays. A sequential pre-read fetches whole files, while mmap fetches
  only the pages actually touched, so on fast local storage it would read *more* than the lazy
  path and cost perhaps a second. :func:`_is_network_filesystem` gates on the filesystem under
  the environment, and anything local or unrecognised is skipped.
* It runs once per node per environment, tracked by a marker in a node-local tmpfs whose
  lifetime matches the page cache it stands for.
* It is bounded by a wall-clock budget and a concurrency cap, so it cannot stall startup and
  a large array job cannot turn into a thundering herd.
* Every failure is swallowed. A warm-up that cannot run leaves behavior exactly as it was.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

#: Filesystems whose mmap page faults cross a network. Warming is skipped on anything else,
#: including every local filesystem, so an unrecognised type costs nothing rather than
#: guessing wrong. ``fuse`` is included by prefix: those are almost always remote or
#: object-backed, and a local FUSE mount only loses the speedup.
_NETWORK_FILESYSTEMS = frozenset(
    {
        "lustre",
        "nfs",
        "nfs4",
        "gpfs",
        "beegfs",
        "ceph",
        "cephfs",
        "panfs",
        "wekafs",
        "gfs2",
        "glusterfs",
        "smb3",
        "cifs",
    }
)

#: Only files at least this large are worth a sequential read; the handful of big CUDA
#: libraries are nearly all of the bytes, and opening hundreds of small ones is pure overhead.
_MIN_INTERESTING_BYTES = 16 * 2**20

#: Give up after this long. A filesystem slow enough to blow this budget would also have made
#: the lazy path miserable, but a warm-up must never be the reason a run fails to start.
_DEFAULT_BUDGET_SECONDS = 30.0

#: Parallel readers. Enough to keep a parallel filesystem busy, few enough that a node-full of
#: ranks does not multiply into a flood.
_READERS = 8

#: Read size per request. Large enough that the syscall overhead disappears.
_CHUNK_BYTES = 4 * 2**20

_WARM_ENV_VAR = "PROTO_WARM_ENV"


def _opt_out() -> bool | None:
    """Whether ``PROTO_WARM_ENV`` forces warming on or off; ``None`` to decide automatically."""
    raw = os.environ.get(_WARM_ENV_VAR)
    if raw is None or raw == "":
        return None
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _mount_table() -> list[tuple[str, str]]:
    """``(mount point, filesystem type)`` for every mount, or ``[]`` where that is unknowable.

    Linux-only by nature. Anywhere without ``/proc/mounts`` the caller sees no mounts and
    therefore skips warming, which is the safe direction.
    """
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return []
    table = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            # Mount points are octal-escaped in /proc/mounts (a space is \040).
            table.append((parts[1].encode().decode("unicode_escape"), parts[2]))
    return table


def _filesystem_type(path: Path) -> str | None:
    """The filesystem type serving ``path``, by longest matching mount point.

    Longest match matters: a container can show a short prefix as a local stub while the real
    storage is mounted deeper. On Sherlock, ``/oak`` appears as local ``xfs`` inside Apptainer
    while ``/oak/stanford`` is the Lustre mount that actually holds the data, so matching the
    first or shortest entry would skip exactly the case this module exists for.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return None
    best: tuple[int, str] | None = None
    for mount_point, fs_type in _mount_table():
        mount = Path(mount_point)
        if resolved == mount or mount in resolved.parents:
            depth = len(mount.parts)
            if best is None or depth > best[0]:
                best = (depth, fs_type)
    return None if best is None else best[1]


def _is_network_filesystem(path: Path) -> bool:
    """Whether ``path`` sits on a filesystem whose page faults cross a network."""
    fs_type = _filesystem_type(path)
    if fs_type is None:
        return False
    return fs_type in _NETWORK_FILESYSTEMS or fs_type.startswith("fuse")


def _marker_path(env_path: Path, fingerprint: str) -> Path:
    """Where this node records that it has already warmed this environment.

    ``/dev/shm`` is tmpfs: node-local and cleared on reboot, which is exactly the lifetime of
    the page cache the marker stands for. Falls back to the temp directory when absent.
    """
    shm = Path("/dev/shm")  # noqa: S108 -- deliberate: tmpfs is node-local and reboot-scoped
    root = shm if shm.is_dir() else Path(tempfile.gettempdir())
    digest = hashlib.sha256(f"{env_path}:{fingerprint}".encode()).hexdigest()[:16]
    return root / f"proto-warm-{os.getuid()}-{digest}"


def _marker_present(marker: Path) -> bool:
    """Whether this node's warm-up marker exists as a real file.

    ``O_NOFOLLOW`` rather than :meth:`Path.exists`: the marker lives in a world-writable
    tmpfs, so another account on the node could plant a symlink at the name and have us
    follow it. A symlink here raises and is reported as "not warmed", which costs a redundant
    warm-up and nothing else.
    """
    try:
        handle = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    os.close(handle)
    return True


def _record_marker(marker: Path, contents: str) -> None:
    """Record that this node warmed this environment, refusing to write through a symlink.

    ``O_EXCL | O_NOFOLLOW`` means a planted symlink, or a racing process that got there
    first, makes this a no-op rather than a write to somewhere unintended.
    """
    try:
        handle = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        logger.debug("env warm-up: could not record the marker at %s: %s", marker, error)
        return
    try:
        os.write(handle, contents.encode())
    finally:
        os.close(handle)


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


def _large_shared_libraries(env_path: Path) -> list[Path]:
    """The environment's shared libraries worth pre-reading, largest first."""
    found: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    try:
        candidates = list(env_path.glob("lib/python3.*/site-packages/**/*.so*"))
    except OSError as error:
        logger.debug("env warm-up: could not walk %s: %s", env_path, error)
        return []
    for candidate in candidates:
        try:
            # Resolve first: conda leaves lib/python3.1 -> python3.12, which would otherwise
            # present every library twice.
            real = candidate.resolve()
            if real in seen:
                continue
            seen.add(real)
            size = real.stat().st_size
        except OSError:
            continue
        if size >= _MIN_INTERESTING_BYTES:
            found.append((size, real))
    return [path for _, path in sorted(found, reverse=True)]


def warm_environment(
    env_path: Path,
    fingerprint: str,
    *,
    budget_seconds: float = _DEFAULT_BUDGET_SECONDS,
) -> int:
    """Pre-read ``env_path``'s large shared libraries into the page cache. Bytes read.

    Does nothing, and returns ``0``, unless this is worth doing and has not been done: the
    environment must sit on a network filesystem, this node must not have warmed it already,
    and it must actually contain large shared libraries. ``PROTO_WARM_ENV`` overrides the
    filesystem test in either direction.

    Never raises. A warm-up that fails for any reason leaves the environment exactly as it was,
    and the caller is no worse off than before this function existed.

    Args:
        env_path (Path): The tool environment to warm.
        fingerprint (str): Identifies this build of the environment, so a rebuilt environment
            is warmed again rather than skipped against its predecessor's marker.
        budget_seconds (float): Stop reading after this long.

    Returns:
        int: Bytes read into the page cache; ``0`` when warming was skipped or failed.
    """
    try:
        forced = _opt_out()
        if forced is False:
            return 0
        if not forced and not _is_network_filesystem(env_path):
            logger.debug("env warm-up: %s is not on a network filesystem, skipping", env_path)
            return 0

        marker = _marker_path(env_path, fingerprint)
        if _marker_present(marker):
            logger.debug("env warm-up: %s already warmed on this node", env_path)
            return 0

        libraries = _large_shared_libraries(env_path)
        if not libraries:
            return 0

        started = time.monotonic()
        deadline = started + budget_seconds
        with ThreadPoolExecutor(max_workers=_READERS) as pool:
            total = sum(pool.map(lambda path: _read_whole(path, deadline), libraries))
        elapsed = time.monotonic() - started

        # Written even on a truncated read: a partial warm-up still helped, and retrying it on
        # every dispatch would cost more than the remainder is worth.
        _record_marker(marker, f"{env_path}\n{total}\n")

        logger.debug(
            "env warm-up: read %.1f GB of %d libraries from %s in %.1fs",
            total / 2**30,
            len(libraries),
            env_path,
            elapsed,
        )
        return total
    except Exception as error:  # an optimization must never break a run
        logger.debug("env warm-up: skipped after an unexpected error: %s", error)
        return 0
