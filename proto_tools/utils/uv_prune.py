"""proto_tools/utils/uv_prune.py.

Prune the uv package cache in the background after an env build.

Envs hardlink their packages from ``UV_CACHE_DIR``, so ``uv cache prune``, which removes only the
entries uv no longer references, frees space without touching any env. A build only starts this
module in a detached process and returns. The prune runs at most once a day, one at a time, at idle
priority. Its stamp, lock and log live in ``{UV_CACHE_DIR}.prune/``, beside the cache, because
``uv cache prune`` deletes anything in the cache it does not recognize.

Run directly with::

    python -m proto_tools.utils.uv_prune <uv> <uv_cache_dir>
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Seconds between prunes of one cache
PRUNE_INTERVAL_SECONDS = 24 * 3600


def state_dir(cache_dir: Path) -> Path:
    """The folder beside ``cache_dir`` that holds the prune's stamp, lock and log."""
    return cache_dir.with_name(f"{cache_dir.name}.prune")


def schedule(uv: Path, cache_dir: Path) -> None:
    """Start a detached prune of ``cache_dir`` with ``uv``, unless one ran within the last day.

    Args:
        uv (Path): The uv binary that built the env.
        cache_dir (Path): The ``UV_CACHE_DIR`` the build used.
    """
    # A recent prune leaves nothing worth spawning for
    if _pruned_recently(cache_dir):
        return

    # Detach, so the build that triggered it returns at once
    state = state_dir(cache_dir)
    state.mkdir(parents=True, exist_ok=True)
    with open(state / "prune.log", "a") as log:
        subprocess.Popen(
            [sys.executable, "-m", __name__, str(uv), str(cache_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )


def prune(uv: Path, cache_dir: Path) -> bool:
    """Run ``uv cache prune`` on ``cache_dir``, unless another prune is running or one ran recently.

    Args:
        uv (Path): The uv binary to prune with.
        cache_dir (Path): The uv cache.

    Returns:
        bool: Whether a prune ran.

    Raises:
        subprocess.CalledProcessError: If ``uv cache prune`` fails.
    """
    state = state_dir(cache_dir)
    state.mkdir(parents=True, exist_ok=True)
    with open(state / "prune.lock", "a") as lock:
        # One prune at a time; a second would find nothing left to do
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if _pruned_recently(cache_dir):
            return False

        # Prune at idle I/O priority where ionice exists, then record when
        ionice = ["ionice", "-c", "3"] if shutil.which("ionice") else []
        subprocess.run(
            [*ionice, str(uv), "cache", "prune"], env={**os.environ, "UV_CACHE_DIR": str(cache_dir)}, check=True
        )
        (state / "last_prune").touch()
        return True


def _pruned_recently(cache_dir: Path) -> bool:
    """Whether ``cache_dir`` was pruned within ``PRUNE_INTERVAL_SECONDS``."""
    stamp = state_dir(cache_dir) / "last_prune"
    return stamp.exists() and time.time() - stamp.stat().st_mtime < PRUNE_INTERVAL_SECONDS


def main(argv: list[str] | None = None) -> None:
    """Prune at idle CPU priority, logging and swallowing any failure: it never affects a build."""
    uv, cache_dir = (Path(arg) for arg in (argv if argv is not None else sys.argv[1:]))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.nice(19)
    try:
        if prune(uv, cache_dir):
            logger.info("Pruned the uv cache at %s", cache_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Could not prune the uv cache at %s: %s", cache_dir, exc)


if __name__ == "__main__":
    main()
