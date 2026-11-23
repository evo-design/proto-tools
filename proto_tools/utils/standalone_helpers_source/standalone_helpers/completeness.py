"""Know a downloaded or built directory is complete by the files it holds, not by a marker alone.

A completion marker can remain after files are deleted, for example during a scratch purge.
Record the directory's files in the marker and check that each one still exists. Rebuild the
directory if any are missing. ``proto_record_files`` and ``proto_recorded_files_complete`` in ``standalone_helpers.sh``
read and write the same marker format, so bash and Python callers agree.

Layers, each usable on its own:

- ``list_files``: the files worth recording under a directory (dotfiles and in-progress downloads
  are not).
- ``missing_files``: which of a list of files under a directory are gone.
- ``record_files`` / ``missing_recorded_files``: write a directory's marker; list what it is missing.
- ``ensure_complete``: build a directory unless it is complete, under a lock, then record it. Every
  step (what to build, what to record, whether to clear a partial copy first, building into a
  staging directory) can be customized.
"""

import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from .proto_logging import get_logger

logger = get_logger(__name__)

COMPLETE_MARKER = ".proto_complete"
# In-progress downloads and locks, which are renamed or removed when the download finishes.
TRANSIENT_SUFFIXES = (".incomplete", ".lock", ".tmp")


def list_files(root: str | Path, *, skip_dirs: Iterable[str] = ()) -> list[str]:
    """Every file under ``root`` worth recording, as sorted paths relative to ``root``.

    Dotfiles and dot-directories (locks, temps, markers, ``.git``), in-progress downloads
    (``TRANSIENT_SUFFIXES``), and directories named in ``skip_dirs`` (caches a tool regenerates) are
    left out.

    Args:
        root (str | Path): Directory to walk; a missing one has no files.
        skip_dirs (Iterable[str]): Directory names to leave out wherever they occur.

    Returns:
        list[str]: Relative paths, ``/``-separated.
    """
    root = Path(root)
    skip = set(skip_dirs)
    files: list[str] = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = [name for name in subdirs if not name.startswith(".") and name not in skip]
        rel_dir = Path(directory).relative_to(root)
        files.extend(
            (rel_dir / name).as_posix()
            for name in names
            if not name.startswith(".") and not name.endswith(TRANSIENT_SUFFIXES)
        )
    return sorted(files)


def record_files(dest: str | Path, files: Iterable[str] | None = None, *, marker: str = COMPLETE_MARKER) -> None:
    """Mark ``dest`` complete by writing the files it holds to ``dest / marker``.

    Args:
        dest (str | Path): The completed directory.
        files (Iterable[str] | None): Paths relative to ``dest`` that make it complete; ``None``
            records ``list_files(dest)``.
        marker (str): Marker file name inside ``dest``.
    """
    dest = Path(dest)
    listed = sorted(set(list_files(dest) if files is None else files))
    tmp = dest / f"{marker}.tmp"
    tmp.write_text("".join(f"{rel}\n" for rel in listed))
    os.replace(tmp, dest / marker)


def missing_recorded_files(dest: str | Path, *, marker: str = COMPLETE_MARKER) -> list[str] | None:
    """The files ``dest``'s marker records that are gone, or ``None`` when it has no marker.

    An empty marker records nothing and so cannot vouch for ``dest``: it reports the marker itself
    as missing. Files are checked with one directory listing per directory, which stays cheap for
    tens of thousands of files.

    Args:
        dest (str | Path): Directory to check.
        marker (str): Marker file name inside ``dest``.

    Returns:
        list[str] | None: Missing relative paths (empty when complete), or ``None`` with no marker.
    """
    dest = Path(dest)
    try:
        listed = [rel for rel in (dest / marker).read_text().splitlines() if rel]
    except FileNotFoundError:
        return None
    if not listed:
        return [marker]
    return missing_files(dest, listed)


def missing_files(root: str | Path, files: Iterable[str]) -> list[str]:
    """Which of ``files`` (paths relative to ``root``) are gone.

    Checks with one directory listing per directory rather than a stat per file, which stays cheap
    for tens of thousands of files on network filesystems. A directory that cannot be read (another
    user's shared cache) is skipped: nothing could be restored into it anyway.

    Args:
        root (str | Path): Directory the paths are relative to.
        files (Iterable[str]): ``/``-separated relative paths.

    Returns:
        list[str]: The missing paths, in the order given.
    """
    root = Path(root)
    present_in: dict[str, set[str] | None] = {}
    missing = []
    for rel in files:
        rel_dir = os.path.dirname(rel)
        if rel_dir not in present_in:
            try:
                present_in[rel_dir] = set(os.listdir(root / rel_dir))
            except (FileNotFoundError, NotADirectoryError):
                present_in[rel_dir] = set()
            except PermissionError:
                present_in[rel_dir] = None
        present = present_in[rel_dir]
        if present is not None and os.path.basename(rel) not in present:
            missing.append(rel)
    return missing


def ensure_complete(
    dest: str | Path,
    build: Callable[[Path], object],
    *,
    marker: str = COMPLETE_MARKER,
    lock: str | Path | None = None,
    progress: str | Path | None = None,
    record: Callable[[Path], Iterable[str]] | None = None,
    reset: bool = True,
    stage: bool = False,
    adopt: Callable[[Path], bool] | None = None,
    force: bool = False,
) -> Path:
    """Build ``dest`` unless every file its marker records is there, then record what it holds.

    Concurrent callers (threads or processes, on any node sharing the filesystem) take turns under
    a lock and re-check after acquiring it, so one builds and the rest find ``dest`` complete.

    Args:
        dest (str | Path): Directory to make complete.
        build (Callable[[Path], object]): Creates and fills the directory it is given: ``dest``,
            which an incomplete copy has been removed from (unless ``reset=False``), or with
            ``stage`` an empty staging directory. Its return value is ignored. A build that leaves no
            files to record raises, and is neither recorded nor swapped into place.
        marker (str): Marker file name inside ``dest``.
        lock (str | Path | None): Lock file; ``None`` uses ``.<name>.extract.lock`` beside ``dest``, the
            name ``proto_with_lock`` callers use for a directory, so bash and Python builds take turns.
        progress (str | Path | None): Path or glob whose growth shows the holder is still building
            (see ``file_lock``); ``None`` watches the staging directory, or ``dest`` itself.
        record (Callable[[Path], Iterable[str]] | None): Lists the paths, relative to the directory
            it is given, that make it complete; ``None`` records every file ``list_files`` finds.
        reset (bool): Remove an incomplete ``dest`` before building, so no stale partial copy
            survives. Pass ``False`` for a ``build`` that resumes or only adds what is missing.
        stage (bool): Build into a fresh directory beside ``dest`` and swap it into place, so readers
            never see a half-built ``dest``.
        adopt (Callable[[Path], bool] | None): For a ``dest`` with no record yet (built before it was
            recorded), whether the copy there is complete by an older check; if so its files are
            recorded instead of building it again, so valid weights are never downloaded twice. A copy
            with no files to record is built again.
        force (bool): Build even when ``dest`` is complete.

    Returns:
        Path: ``dest``.
    """
    from .weights import file_lock

    dest = Path(dest)
    # Return early, without the lock, when every recorded file is present
    if not force and missing_recorded_files(dest, marker=marker) == []:
        return dest
    # Place the lock beside dest under the name bash proto_with_lock callers also use
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(lock) if lock is not None else dest.parent / f".{dest.name}.extract.lock"
    # Watch the staging dirs (or dest) grow so waiters can tell the holder is still building
    if progress is None:
        progress = f"{dest.parent}/.{dest.name}.stage.*" if stage else str(dest)
    with file_lock(lock_path, progress=progress):
        # Re-check under the lock, since another caller may have finished building meanwhile
        missing = missing_recorded_files(dest, marker=marker)
        if not force and missing == []:
            return dest
        # Record an unmarked copy that passes the older check instead of building it again
        if not force and missing is None and adopt is not None and dest.is_dir() and adopt(dest):
            adopted = _files_to_record(dest, record)
            if adopted:
                record_files(dest, adopted, marker=marker)
                return dest
        if missing:
            logger.warning("%s lost %d file(s) (e.g. %s); building it again", dest, len(missing), missing[0])
        # Build into a staging dir beside dest, record it, then swap it into place
        if stage:
            staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.stage.", dir=dest.parent))
            try:
                build(staging)
                record_files(staging, _built_files(staging, record, dest), marker=marker)
                # Replace any old dest with the finished staging dir
                _remove(dest)
                os.replace(staging, dest)
            finally:
                # Clean up the staging dir if the build or swap failed
                shutil.rmtree(staging, ignore_errors=True)
        else:
            # Otherwise build in place: clear a partial copy unless resuming, and always drop the old marker
            if reset:
                _remove(dest)
            (dest / marker).unlink(missing_ok=True)
            # Not created here: some builds (boltz, git clone) act only when dest does not exist
            build(dest)
            # Confirm the build created dest, then record the files it left
            if not dest.is_dir():
                raise RuntimeError(f"building {dest} did not create it")
            record_files(dest, _built_files(dest, record, dest), marker=marker)
    return dest


def _files_to_record(directory: Path, record: Callable[[Path], Iterable[str]] | None) -> list[str]:
    return sorted(set(list_files(directory) if record is None else record(directory)))


def _built_files(directory: Path, record: Callable[[Path], Iterable[str]] | None, dest: Path) -> list[str]:
    """The files to record for ``directory`` (``dest`` or its staging copy); raises when there are none.

    A build that exits cleanly but leaves nothing behind (a download that fetched nothing) must not
    be handed back as complete.
    """
    files = _files_to_record(directory, record)
    if not files:
        raise RuntimeError(f"building {dest} produced no files")
    return files


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()
