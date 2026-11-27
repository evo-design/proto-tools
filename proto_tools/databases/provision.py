"""Provisioning of registered datasets: download, verify, build, and the provisioned check.

Generic over every :class:`~proto_tools.databases.registry.DatasetEntry`: each entry declares its
own ``downloads`` and ``build`` steps, so adding a dataset is a new entry in
``databases/entries/`` with no change here. Steps run with ``cwd`` set to the dataset's cache
directory and resolve their programs from ``PATH``; callers whose steps need a tool binary, such as
``mmseqs``, put it there first.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from proto_tools.databases.registry import (
    BuildStep,
    DatasetEntry,
    DatasetRegistry,
    DownloadSpec,
    dataset_slug,
    get_dataset_dir,
)
from proto_tools.utils.system_info import available_memory_bytes

logger = logging.getLogger(__name__)

# Fraction of the cgroup-aware budget allotted to mmseqs index builds (mmseqs sizes splits from physical RAM, so the cap keeps `createindex` from OOMing under Slurm / container memory limits).
_INDEX_MEMORY_SAFETY_FRACTION = 0.7

# Suffix an in-flight download occupies until it is verified and renamed into place.
_PARTIAL_SUFFIX = ".part"

# Read size for checksumming, large enough to keep multi-gigabyte datasets I/O bound.
_HASH_BLOCK_BYTES = 1024 * 1024


@contextmanager
def provision_lock(cache_dir: Path) -> Iterator[None]:
    """Hold the cross-process lock that serializes provisioning of the dataset at ``cache_dir``.

    Several callers can want the same dataset at once (parallel test workers locally, containers
    sharing one volume remotely); without the lock they race on a half-written cache dir. The second
    arrival blocks, and should then re-check :func:`is_provisioned` before doing any work.

    Args:
        cache_dir (Path): The dataset's cache directory. The lock file sits beside it.

    Yields:
        None: While the lock is held.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.provision.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def is_provisioned(entry: DatasetEntry, cache_dir: Path) -> bool:
    """Return whether every file ``entry`` declares in ``build.output_files`` is in ``cache_dir``."""
    for output_file in entry.build.output_files:
        pattern = output_file.replace("{name}", dataset_slug(entry.name))
        # Glob patterns are allowed for MMseqs2's split database files
        if "*" in pattern:
            if not list(cache_dir.glob(pattern)):
                return False
        elif not (cache_dir / pattern).exists():
            return False
    return True


def provision(name: str, workdir: Path | None = None, *, force: bool = False) -> Path:
    """Download and build a single dataset by registry name.

    Args:
        name (str): Registered dataset key (e.g. ``"uniref30-2302"``).
        workdir (Path | None): Override the standard cache root. When set,
            the dataset goes to ``<workdir>/<slug>/`` (still isolated per dataset).
        force (bool): Re-download / re-run build steps even if the dataset
            looks provisioned.

    Returns:
        Path: The dataset's cache directory.
    """
    entry = DatasetRegistry.get(name)
    cache_dir = _resolve_cache_dir(name, workdir)
    logger.info(
        "%s (%s, %s) at %s: ~%.1f GB download / ~%.1f GB built",
        name,
        entry.display_name,
        entry.molecule_type,
        cache_dir,
        entry.total_download_bytes / 1e9,
        entry.total_disk_bytes / 1e9,
    )

    if not force and is_provisioned(entry, cache_dir):
        logger.info("%s is already provisioned (force to re-run)", name)
        return cache_dir

    # Forcing wipes the cache dir so stale outputs don't collide with this run (e.g. ``zstd -d``
    # prompting on an existing FASTA, or ``mmseqs createdb`` refusing to overwrite a partial DB).
    if force and cache_dir.exists():
        logger.info("Wiping %s before re-provisioning", cache_dir)
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Inside a worker log lines reach no caller, so the long phases also travel as status records.
    logger.info("Downloading sources for %s", entry.display_name, extra={"update_status": True})
    downloader = _which_downloader()
    for spec in entry.downloads:
        _download_file(spec, cache_dir, downloader)

    logger.info("Building %s", entry.display_name, extra={"update_status": True})
    # Only meaningful to a recipe that asks for it: a reference genome that just decompresses would
    # otherwise report an mmseqs memory budget it never uses.
    wants_memory_limit = any("{split_memory_limit}" in arg for step in entry.build.steps for arg in step.command)
    split_memory_limit = _split_memory_limit() if wants_memory_limit else "0"
    if split_memory_limit != "0":
        logger.info("Capping mmseqs index splits at %siB", split_memory_limit)
    for step in entry.build.steps:
        _run_step(step, cache_dir, name, split_memory_limit=split_memory_limit)

    logger.info("%s ready at %s", name, cache_dir)
    return cache_dir


def _resolve_cache_dir(name: str, workdir: Path | None) -> Path:
    """Return where ``name`` is installed: its registry cache dir, or ``<workdir>/<slug>/``."""
    if workdir is None:
        return get_dataset_dir(name)
    return workdir / dataset_slug(name)


def _which_downloader() -> tuple[str, list[str]]:
    """Pick the best available HTTP downloader: aria2c (parallel) > curl > wget.

    Returns ``(name, base_argv)`` where ``base_argv`` is the static prefix
    to which we append output / URL.
    """
    if shutil.which("aria2c"):
        return "aria2c", [
            "aria2c",
            "--max-connection-per-server=8",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
        ]
    if shutil.which("curl"):
        return "curl", ["curl", "-L", "--fail"]
    if shutil.which("wget"):
        return "wget", ["wget", "--no-verbose"]
    raise RuntimeError("No download tool found in PATH. Install aria2c, curl, or wget.")


def _wrong_size(path: Path, spec: DownloadSpec) -> str | None:
    """Describe how ``path``'s size differs from ``spec``, or None when it matches or is unknown."""
    if spec.expected_bytes is None:
        return None
    actual = path.stat().st_size
    if actual == spec.expected_bytes:
        return None
    return f"{actual} bytes on disk, expected {spec.expected_bytes}"


def _wrong_checksum(path: Path, spec: DownloadSpec) -> str | None:
    """Describe how ``path``'s SHA-256 differs from ``spec``, or None when it matches or is unset."""
    if spec.sha256 is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual == spec.sha256:
        return None
    return f"sha256 {actual}, expected {spec.sha256}"


def _download_file(spec: DownloadSpec, cache_dir: Path, downloader: tuple[str, list[str]]) -> bool:
    """Download a single ``DownloadSpec`` into ``cache_dir``. Returns True on success.

    Idempotent: an existing file is reused only when its size agrees with ``spec``, so an
    incomplete file left by an earlier run is fetched again rather than trusted. The transfer
    lands on a ``.part`` file and is renamed only once verified, so an interrupted download
    never occupies the final path. ``downloader`` is the ``(name, base_argv)`` tuple resolved
    once per provisioning run by the caller (so we don't re-``shutil.which`` on every file).
    """
    target = cache_dir / spec.filename
    if target.exists():
        stale = _wrong_size(target, spec)
        if stale is None:
            logger.info("%s already present", spec.filename)
            return True
        logger.warning("%s is stale (%s); downloading again", spec.filename, stale)
        target.unlink()

    name, base_argv = downloader
    logger.info("Downloading %s via %s: %s", spec.filename, name, spec.url)
    partial = target.with_name(target.name + _PARTIAL_SUFFIX)
    partial.unlink(missing_ok=True)

    if name == "aria2c":
        cmd = [*base_argv, "-o", partial.name, "-d", str(cache_dir), spec.url]
    elif name == "curl":
        cmd = [*base_argv, "-o", str(partial), spec.url]
    else:  # wget
        cmd = [*base_argv, "-O", str(partial), spec.url]

    try:
        subprocess.run(cmd, check=True)
        for problem in (_wrong_size(partial, spec), _wrong_checksum(partial, spec)):
            if problem is not None:
                msg = f"{spec.filename} did not download completely: {problem}"
                raise RuntimeError(msg)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        partial.unlink(missing_ok=True)
        if not spec.required:
            logger.warning("Optional file %s failed to download (continuing): %s", spec.filename, e)
            return True
        raise
    partial.replace(target)
    return True


def _split_memory_limit() -> str:
    """``--split-memory-limit`` value sized to the cgroup-aware memory budget.

    Formatted with mmseqs's ``G`` (GiB) suffix: mmseqs misreads a bare byte
    count as an enormous limit (e.g. ``60129542144`` → ``57344T``, effectively
    unbounded), so the suffix is required for the cap to take effect. Returns
    ``"0"`` (mmseqs default = all available) when the budget can't be detected.
    """
    budget = available_memory_bytes()
    if budget <= 0:
        return "0"
    gib = int(budget * _INDEX_MEMORY_SAFETY_FRACTION / 1024**3)
    return f"{max(1, gib)}G"


def _run_step(step: BuildStep, cache_dir: Path, name: str, *, split_memory_limit: str = "0") -> None:
    """Run a single ``BuildStep`` argv with ``cwd=cache_dir``.

    Substitutes template placeholders in each argv element: ``{name}`` →
    the dataset's slug, and ``{split_memory_limit}`` → a cgroup-aware ``mmseqs`` memory cap.
    """
    cmd = [
        arg.replace("{name}", dataset_slug(name)).replace("{split_memory_limit}", split_memory_limit)
        for arg in step.command
    ]
    logger.info("%s: $ %s", step.description, " ".join(cmd))
    subprocess.run(cmd, cwd=cache_dir, check=True)
