"""proto_tools/utils/download_progress.py.

Request deadlines that do not charge the time a worker spends downloading weights.

A tool's first request can download gigabytes of weights at whatever speed the link allows, so
a fixed wall-clock timeout kills healthy downloads on slow links. ``ProgressDeadline`` bounds a
request by its timeout, but adds back every interval in which the toolkit's download folders
(``HF_HOME``, ``TORCH_HOME``, and its weights directory) grew. Compute, including JIT compiles,
is charged in full. A stalled download stops growing, so the clock runs again and the remaining
timeout bounds it.

Growth is measured by summing file sizes under those folders, which covers every download path
(Hugging Face ``*.incomplete`` blobs, ``download_weights`` temp files, and libraries that write
the final filename directly) with no hook in the worker. The walk runs at most once per
``PROTO_DOWNLOAD_POLL_SECONDS`` (default 30), and not at all for a request that finishes
within its first poll.
"""

import logging
import os
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_POLL_SECONDS = 30.0
# Growth below this between two checks is not a download; caches and logs write less.
_MIN_GROWTH_BYTES = 1 << 20
# Most download time one request may exclude, so a writer mistaken for a download cannot hold a
# wedged worker open forever.
_DEFAULT_MAX_CREDIT_SECONDS = 12 * 3600.0


def _positive_env_seconds(name: str, default: float) -> float:
    """Read environment variable ``name`` as a positive float, else ``default``."""
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return value if value > 0 else default


def _poll_seconds() -> float:
    """Read ``PROTO_DOWNLOAD_POLL_SECONDS``, defaulting to 30."""
    return _positive_env_seconds("PROTO_DOWNLOAD_POLL_SECONDS", _DEFAULT_POLL_SECONDS)


def _max_credit_seconds() -> float:
    """Read ``PROTO_DOWNLOAD_MAX_SECONDS``, defaulting to 12 hours."""
    return _positive_env_seconds("PROTO_DOWNLOAD_MAX_SECONDS", _DEFAULT_MAX_CREDIT_SECONDS)


def _tree_bytes(root: Path) -> int:
    """Sum the sizes of regular files under ``root``, without following symlinks.

    Hugging Face snapshots are symlinks into ``blobs/``, so following them would count each file
    twice. Missing directories and files removed mid-walk count as zero.

    Args:
        root (Path): Directory to walk.

    Returns:
        int: Total size in bytes.
    """
    total = 0
    stack = [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:  # noqa: PERF203 -- a file can vanish mid-walk
                        continue
        except OSError:
            continue
    return total


class DownloadWatch:
    """Reports whether the total size of a set of folders grew since the last check.

    Args:
        dirs (Iterable[Path]): Folders to watch. Duplicates and missing folders are fine.
    """

    def __init__(self, dirs: Iterable[Path]) -> None:
        """Initialize DownloadWatch."""
        self._dirs = tuple(dict.fromkeys(Path(d) for d in dirs))
        self._size: int | None = None

    def grew(self) -> bool:
        """Return whether the folders grew by at least ``_MIN_GROWTH_BYTES`` since the previous call.

        The first call records a baseline and returns False.

        Returns:
            bool: True if the total size grew by at least 1 MiB since the previous call.
        """
        size = sum(_tree_bytes(d) for d in self._dirs)
        grew = self._size is not None and size - self._size >= _MIN_GROWTH_BYTES
        self._size = size
        return grew


class ProgressDeadline:
    """A request deadline that is extended by every interval in which downloads progressed.

    Args:
        timeout (float | None): Seconds of non-download time allowed. None never expires.
        dirs (Iterable[Path]): The worker's download folders (see ``_download_dirs``).
        label (str): Toolkit name for log messages.
    """

    def __init__(self, timeout: float | None, dirs: Iterable[Path], *, label: str) -> None:
        """Initialize ProgressDeadline."""
        self._watch = DownloadWatch(dirs)
        self._label = label
        self._poll = _poll_seconds()
        self._max_credit = _max_credit_seconds()
        now = time.monotonic()
        self._deadline = None if timeout is None else now + timeout
        # The first measurement is a baseline, so take it early enough that a short timeout still
        # gets a second one before it expires.
        self._next_check = now + (self._poll if timeout is None else min(self._poll, timeout / 2))
        self._last_check = now
        self.download_seconds = 0.0
        self.downloading = False

    def wait_slice(self) -> float | None:
        """Return how long to block before calling ``tick`` again (None blocks indefinitely).

        Returns:
            float | None: Seconds until the next check or the deadline, whichever is sooner.
        """
        if self._deadline is None:
            return None
        return max(0.0, min(self._deadline, self._next_check) - time.monotonic())

    def tick(self) -> None:
        """Measure download progress if a check is due, and extend the deadline if it grew."""
        if self._deadline is None:
            return
        now = time.monotonic()
        if now < self._next_check and now < self._deadline:
            return
        self.downloading = self._watch.grew()
        if self.downloading:
            credit = min(now - self._last_check, self._max_credit - self.download_seconds)
            if credit > 0:
                if not self.download_seconds:
                    logger.info(
                        "Downloading weights for %s; download time does not count against the timeout", self._label
                    )
                self._deadline += credit
                self.download_seconds += credit
        self._last_check = now
        self._next_check = now + self._poll

    def expired(self) -> bool:
        """Return whether the deadline has passed.

        Returns:
            bool: True once the remaining non-download time is used up.
        """
        return self._deadline is not None and time.monotonic() >= self._deadline

    def timeout_note(self) -> str:
        """Return a suffix for a timeout message noting the download time that was excluded.

        Returns:
            str: ``" (excluding Ns spent downloading weights)"``, or ``""`` if nothing downloaded.
        """
        return f" (excluding {self.download_seconds:.0f}s spent downloading weights)" if self.download_seconds else ""

    def wait_process(self, proc: "subprocess.Popen[str]") -> int:
        """Wait for ``proc`` to exit under this deadline.

        Args:
            proc (subprocess.Popen[str]): The running subprocess.

        Returns:
            int: The process's exit code.

        Raises:
            subprocess.TimeoutExpired: If the deadline passes first. The process is left running.
        """
        while True:
            try:
                return proc.wait(timeout=self.wait_slice())
            except subprocess.TimeoutExpired:  # noqa: PERF203 -- each slice ends in a timeout until exit
                self.tick()
                if self.expired():
                    raise
