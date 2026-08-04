"""Resolution of a config value that is either a registered dataset name or a local path."""

import logging
from pathlib import Path

from proto_tools.databases.registry import DatasetRegistry, get_dataset_dir

logger = logging.getLogger(__name__)


def is_registered_dataset(value: str) -> bool:
    """Whether ``value`` names a registered dataset rather than a filesystem path.

    The discriminator for every "slug or path" config field. A registered name is the only form a
    remote worker can resolve, since a path means something only on the machine that wrote it.
    """
    return value in DatasetRegistry.list_all()


def dataset_file(name: str, filename: str, *, provision_if_missing: bool = True) -> Path:
    """Return the path to ``filename`` inside registered dataset ``name``, provisioning if needed.

    Args:
        name (str): Registered dataset name, e.g. ``"grch38"``.
        filename (str): File within the dataset directory.
        provision_if_missing (bool): Download and run the entry's index recipe when the file is
            absent and the entry declares ``auto_provision``. Set False where the caller must not
            spend minutes and gigabytes without being asked.

    Returns:
        Path: The resolved file. May not exist when provisioning was declined or unavailable —
            callers report that as :class:`~proto_tools.utils.tool_io.MissingAssetError` so the
            "not staged here" case stays distinguishable from a genuine failure.
    """
    cache_dir = get_dataset_dir(name)
    path = cache_dir / filename
    if path.exists():
        return path

    entry = DatasetRegistry.get(name)
    if not (provision_if_missing and entry.auto_provision):
        return path

    import fcntl

    # Imported here rather than at module scope: provisioning lives with the mmseqs2 toolkit, which
    # pulls in its own dependencies, and every caller of this module resolves paths far more often
    # than it provisions. Lifting `provision` out of that toolkit would let this be a plain import.
    from proto_tools.tools.sequence_alignment.mmseqs2.setup_databases import provision

    # Serialize across processes, the same way the mmseqs2 fixture path does. Several callers can
    # want the same dataset at once — parallel test workers locally, or containers sharing one
    # volume remotely — and without this they race on a half-written cache dir. The second arrival
    # blocks, then finds the file present and skips its own download.
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.provision.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.exists():
            logger.info("%s was provisioned by another process while waiting", name)
            return path
        logger.warning(
            "%s is not provisioned; downloading ~%.1f GB now (one time, shared by later calls)",
            name,
            entry.total_download_bytes / 1e9,
        )
        provision(name)
    return cache_dir / filename
