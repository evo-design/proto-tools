"""Weights location and download helpers for standalone scripts.

Resolves the canonical location where a tool should store its model weights,
following the ``PROTO_HOME`` / ``PROTO_MODEL_CACHE`` / ``PROTO_{TOOL}_WEIGHTS_DIR``
precedence documented in ``notes/storage.md``, hands that location to libraries
that read their own environment variable, and downloads files into it safely.
"""

import contextlib
import fcntl
import glob
import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from .completeness import ensure_complete
from .proto_logging import get_logger

logger = get_logger(__name__)


def resolve_weights_dir(toolkit: str) -> str | None:
    """Resolve the weights directory for a tool based on PROTO_MODEL_CACHE.

    Precedence:
        1. PROTO_{TOOL}_WEIGHTS_DIR (per-tool override, always wins)
        2. PROTO_MODEL_CACHE:
           - (default): {PROTO_HOME}/proto_model_cache/{toolkit}/weights/ (survives env rebuilds)
           - "/absolute/path": /absolute/path/{toolkit}/weights/  (shared directory)
           (the ``weights/`` subfolder of the {toolkit}/ folder that also holds its
           HF, torch, and JAX caches)
           - "IN_ENV": {TOOL_VENV_PATH}/model_weight_cache/ (legacy, per-venv)
           - "NONE": {VENV_PATH}/weights/ (pass-through, matches shell helper)

    Args:
        toolkit (str): The tool's directory name (e.g., "fampnn", "protenix").

    Returns:
        str | None: Absolute path string to the weights directory. Returns ``None``
            only in ``NONE`` mode when no venv path is available (caller falls back
            to its own default). Creates the directory if it doesn't exist.

    Raises:
        RuntimeError: If ``PROTO_MODEL_CACHE=IN_ENV`` but neither
            ``TOOL_VENV_PATH`` nor ``VENV_PATH`` is set (cannot resolve a venv-local
            weights directory without a venv).
    """
    # 1. Per-tool override always wins
    override_var = f"PROTO_{toolkit.upper()}_WEIGHTS_DIR"
    override = os.environ.get(override_var)
    if override:
        os.makedirs(override, exist_ok=True)
        return override

    # 2. PROTO_MODEL_CACHE
    mode = os.environ.get("PROTO_MODEL_CACHE", "")

    if mode == "NONE":
        # Pass-through: no managed cache, but match the shell helper's fallback
        # (setup.sh downloads to ${VENV_PATH}/weights in NONE mode)
        venv_path = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv_path:
            path = os.path.join(venv_path, "weights")
            os.makedirs(path, exist_ok=True)
            return path
        return None

    if mode == "IN_ENV":
        # Legacy: weights inside the tool's venv
        venv_path = os.environ.get("TOOL_VENV_PATH") or os.environ.get("VENV_PATH")
        if venv_path:
            path = os.path.join(venv_path, "model_weight_cache")
            os.makedirs(path, exist_ok=True)
            return path
        raise RuntimeError(
            f"resolve_weights_dir({toolkit!r}): PROTO_MODEL_CACHE=IN_ENV but TOOL_VENV_PATH/VENV_PATH unset"
        )

    if mode:
        # Explicit path (absolute or relative)
        cache_dir = mode
    else:
        # Default: PROTO_HOME/proto_model_cache/ directory  # noqa: ERA001 -- path description, not code
        proto_home = os.environ.get("PROTO_HOME", "")
        if not proto_home:
            # Same default as get_proto_home(): ~/.proto/
            proto_home = os.path.join(os.path.expanduser("~"), ".proto")
            logger.warning(
                "PROTO_HOME not set in subprocess environment. Falling back to %s. Set PROTO_HOME to customize.",
                proto_home,
            )
        cache_dir = os.path.join(proto_home, "proto_model_cache")

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, toolkit, "weights")
    os.makedirs(path, exist_ok=True)
    return path


def require_weights_dir(toolkit: str) -> str:
    """Return ``resolve_weights_dir(toolkit)``, raising instead of returning ``None``.

    Use this where a tool must download or read weights and has no sensible location of its own,
    instead of falling back to a temp directory that silently re-downloads on every machine.

    Args:
        toolkit (str): The name whose weights directory to resolve.

    Returns:
        str: Absolute path string to the weights directory (created if missing).

    Raises:
        RuntimeError: If no directory can be resolved (``PROTO_MODEL_CACHE=NONE`` outside a venv).
    """
    path = resolve_weights_dir(toolkit)
    if path is None:
        raise RuntimeError(
            f"{toolkit}: no weights directory (PROTO_MODEL_CACHE=NONE and no venv); "
            f"set PROTO_{toolkit.upper()}_WEIGHTS_DIR"
        )
    return path


def export_weights_dir(env_var: str, toolkit: str, *, prepend: bool = False) -> str:
    """Point a library's own weights environment variable at the managed weights directory.

    Libraries such as Foundry (``FOUNDRY_CHECKPOINT_DIRS``), Chai-1 (``CHAI_DOWNLOADS_DIR``), and
    Protenix (``PROTENIX_ROOT_DIR``) read their download location from an environment variable.
    Setting it through this helper keeps every toolkit on one rule: the managed directory wins,
    and ``PROTO_{TOOLKIT}_WEIGHTS_DIR`` is the way to choose another location.

    Args:
        env_var (str): The library's environment variable.
        toolkit (str): The name whose weights directory to resolve.
        prepend (bool): For a search-path variable, put the managed directory first and keep any
            directories already listed after it, instead of replacing the value.

    Returns:
        str: The managed weights directory now named by ``env_var``.
    """
    path = require_weights_dir(toolkit)
    existing = os.environ.get(env_var, "")
    if prepend and existing:
        others = [p for p in existing.split(os.pathsep) if p and p != path]
        os.environ[env_var] = os.pathsep.join([path, *others])
    else:
        os.environ[env_var] = path
    return path


# Locks this thread holds, so taking one again (a helper calling a helper) doesn't wait on itself
_held_locks = threading.local()


def _env_seconds(name: str, default: float) -> float:
    """Read a seconds setting from the environment, falling back to ``default`` when unset or invalid."""
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _progress_size(pattern: str) -> int:
    """Return the total size of what ``pattern`` (a path or glob) matches, recursing into directories."""
    total = 0
    for match in glob.glob(pattern):
        path = Path(match)
        if path.is_dir():
            for root, _dirs, files in os.walk(path):
                for name in files:
                    with contextlib.suppress(OSError):
                        total += (Path(root) / name).stat().st_size
        else:
            with contextlib.suppress(OSError):
                total += path.stat().st_size
    return total


@contextlib.contextmanager
def file_lock(
    lock_path: str | Path, timeout: float | None = None, *, progress: str | Path | None = None
) -> Iterator[None]:
    """Hold an exclusive cross-process ``flock`` on ``lock_path`` so concurrent workers don't race.

    The kernel releases the lock when its holder exits, so a crashed downloader leaves no stale
    lock. The 0-byte lock file is never unlinked; deleting it would let a second process lock a
    fresh inode while a third still holds the old one. This is the Python side of
    ``proto_with_lock`` in ``standalone_helpers.sh``: same lock files, same waiting rules, and
    the same ``PROTO_LOCK_*`` settings, so bash and Python callers for one target take turns.

    Waiting for another holder has no fixed deadline. With ``progress`` (a path or glob, such as
    the holder's temp file), the waiter keeps waiting while it grows and gives up only after
    ``PROTO_LOCK_STALL_SECONDS`` (default 600) without growth, checking every
    ``PROTO_LOCK_POLL_SECONDS`` (default 60). ``timeout``, or ``PROTO_LOCK_TIMEOUT_SECONDS``
    when it is ``None`` (default 0, off), caps the total wait.

    Re-entrant: a lock this thread already holds, or one a parent ``proto_with_lock`` holds
    (listed in ``_PROTO_HELD_LOCKS``), is entered without waiting.

    Args:
        lock_path (str | Path): Lock file to create and lock.
        timeout (float | None): Seconds to wait before giving up; ``None`` uses
            ``PROTO_LOCK_TIMEOUT_SECONDS`` (0 waits without a cap).
        progress (str | Path | None): Path or glob whose growth shows the holder is still working.

    Raises:
        TimeoutError: If the cap is reached, or ``progress`` stops growing for the stall window.
    """
    lock_path = Path(lock_path)
    key = str(lock_path)
    held: set[str] = getattr(_held_locks, "paths", None) or set()
    _held_locks.paths = held
    if key in held or key in os.environ.get("_PROTO_HELD_LOCKS", "").split(":"):
        yield
        return

    cap = timeout if timeout is not None else _env_seconds("PROTO_LOCK_TIMEOUT_SECONDS", 0)
    capped = timeout is not None or cap > 0
    stall = _env_seconds("PROTO_LOCK_STALL_SECONDS", 600)
    poll = _env_seconds("PROTO_LOCK_POLL_SECONDS", 60)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        started = last_check = idle_since = time.monotonic()
        last_size: int | None = None
        waiting = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as err:
                now = time.monotonic()
                if not waiting:
                    logger.info("Waiting for another process to finish with %s", lock_path.name)
                    waiting = True
                if capped and now - started >= cap:
                    raise TimeoutError(f"timed out after {now - started:.0f}s waiting for lock {lock_path}") from err
                if progress is not None and now - last_check >= poll:
                    last_check = now
                    size = _progress_size(str(progress))
                    if size != last_size:
                        last_size, idle_since = size, now
                    elif now - idle_since >= stall:
                        raise TimeoutError(
                            f"gave up waiting for lock {lock_path}: {progress} made no progress for "
                            f"{now - idle_since:.0f}s (the holder may be stuck; raise PROTO_LOCK_STALL_SECONDS "
                            "to wait longer)"
                        ) from err
                time.sleep(min(1.0, poll))
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, read in 1 MiB chunks.

    Args:
        path (str | Path): File to hash.
        algorithm (str): A ``hashlib`` algorithm name, e.g. ``"sha256"`` or ``"md5"``.

    Returns:
        str: Lowercase hex digest.
    """
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_trusted_hf_url(url: str) -> bool:
    """Return whether ``url`` is HTTPS on ``huggingface.co`` or a real subdomain (not ``evil-huggingface.co``)."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co"))


def redact_url(url: str) -> str:
    """Return ``url`` reduced to its origin so signed paths, queries, and userinfo stay out of logs.

    Args:
        url (str): URL that may carry credentials.

    Returns:
        str: Scheme and host (and port) with any path or query replaced by fixed markers.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.hostname or ""
        if ":" in netloc:
            netloc = f"[{netloc}]"
        if parsed.port:  # accessing .port raises ValueError for a malformed port
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        return "<unparseable-url>"
    cleaned = parsed._replace(
        netloc=netloc,
        path=("/<redacted-path>" if parsed.path else ""),
        params="",
        query=("REDACTED" if parsed.query else ""),
        fragment="",
    )
    return urllib.parse.urlunparse(cleaned)


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse non-HTTPS redirects and drop ``Authorization`` once a hop leaves Hugging Face.

    Weights are often unpickled after download, so an HTTPS-to-HTTP downgrade would let a network
    attacker swap in an arbitrary pickle; stripping the token alone is not enough.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise urllib.error.HTTPError(newurl, code, "refusing insecure (non-HTTPS) redirect", headers, fp)
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _is_trusted_hf_url(newurl):
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _shared_cache_file_mode(cache_dir: Path) -> int:
    """Return an owner-writable mode readable by whoever may traverse ``cache_dir`` (shared team caches)."""
    directory_mode = cache_dir.stat().st_mode
    mode = stat.S_IRUSR | stat.S_IWUSR
    if directory_mode & stat.S_IXGRP:
        mode |= stat.S_IRGRP
    if directory_mode & stat.S_IXOTH:
        mode |= stat.S_IROTH
    return mode


def _verified(path: Path, sha256: str | None, md5: str | None) -> bool:
    if not _is_regular_file(path):
        return False
    if sha256 and file_digest(path, "sha256") != sha256:
        return False
    return not (md5 and file_digest(path, "md5") != md5)


def _stamp_path(path: Path) -> Path:
    """Where this user's verification of ``path`` is remembered.

    One stamp per user: on a shared cache a stamp is trusted only by the user who wrote it (see
    ``_read_own_stamp``), so users sharing a file each keep their own rather than replacing each
    other's.
    """
    return path.with_name(f".{path.name}.verified.{os.getuid()}")


def _stamp_line(path: Path, sha256: str | None, md5: str | None) -> str:
    """The line that vouches for ``path`` exactly as it is now, with the digests it matched.

    Device, inode and change time make any replacement or rewrite of the file a mismatch, including
    a same-size one whose modification time was restored: ``ctime`` cannot be set by hand, and a
    rename into place gives a new inode.
    """
    st = path.lstat()
    return f"{st.st_dev} {st.st_ino} {st.st_size} {st.st_mtime_ns} {st.st_ctime_ns} {sha256 or '-'} {md5 or '-'}\n"


def _read_own_stamp(stamp: Path) -> tuple[str, int] | None:
    """A stamp's line and modification time, or ``None`` unless this user alone could have written it.

    Opened without following symlinks, and trusted only when it is a regular file owned by this user
    that neither group nor others can write: anyone who can write a shared cache directory could
    otherwise stamp a file they swapped in.
    """
    try:
        fd = os.open(stamp, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    with os.fdopen(fd) as handle:
        st = os.fstat(handle.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            return None
        return handle.read(), st.st_mtime_ns


def _stamp_vouches(path: Path, line: str) -> bool:
    """Whether this user's stamp vouches for ``path`` as ``line`` describes it now.

    Filesystems record times on a coarse clock (a few milliseconds per tick), so a rewrite in the
    same tick as the stamp keeps the change time it recorded. As with git's "racy" index entries, a
    stamp written in the same tick as the file's last change is not trusted, and the file is hashed
    again; the stamp written then, a tick later, is.
    """
    stamp = _read_own_stamp(_stamp_path(path))
    if stamp is None:
        return False
    recorded, stamped_at = stamp
    return recorded == line and path.lstat().st_ctime_ns < stamped_at


def _remember_verified(path: Path, line: str) -> None:
    """Record that ``path``, as ``line`` describes it, matched its digests.

    The stamp is written to a fresh temp file and renamed into place, so a symlink planted at the
    stamp's name is replaced rather than followed, and readers never see a partial stamp.
    """
    stamp = _stamp_path(path)
    with contextlib.suppress(OSError):  # a read-only shared cache just hashes again next time
        fd, tmp_name = tempfile.mkstemp(prefix=f"{stamp.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(line)
            os.replace(tmp_name, stamp)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)


def _verified_in_place(path: Path, sha256: str | None, md5: str | None) -> bool:
    """``_verified`` for a file already in place, hashing it only once while it stays unchanged.

    Hashing a multi-GB checkpoint on every load adds seconds to every worker start, so a file that
    matched is stamped (``.{name}.verified.<uid>`` beside it) and later checks compare the stamp
    instead.
    """
    if not _is_regular_file(path):
        return False
    if not (sha256 or md5):
        return True
    line = _stamp_line(path, sha256, md5)
    if _stamp_vouches(path, line):
        return True
    if not _verified(path, sha256, md5):
        return False
    # Stamp only the file that was hashed: one replaced while it was being hashed is checked again
    # next time rather than vouched for.
    if _is_regular_file(path) and _stamp_line(path, sha256, md5) == line:
        _remember_verified(path, line)
    return True


def download_weights(
    url: str,
    dest: str | Path,
    *,
    sha256: str | None = None,
    md5: str | None = None,
    timeout: float = 120,
    retries: int = 3,
) -> Path:
    """Download ``url`` to ``dest`` once, safely, for concurrent workers on a shared cache.

    A ``dest`` that already exists (and matches the checksum, when one is given) is returned
    without a request; a file that matched is hashed only once while it stays unchanged.
    Otherwise the download holds ``.{name}.download.lock`` beside ``dest``,
    writes an exclusively created temp file, verifies it, and renames it into place, so readers
    never see a partial file. ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) is sent only to ``https://huggingface.co`` and is
    dropped if a redirect leaves it; non-HTTPS URLs and redirects are refused. This is the
    Python counterpart of ``proto_download_verified`` in ``standalone_helpers.sh``.

    Args:
        url (str): HTTPS URL to fetch.
        dest (str | Path): Final file path; its parent is created.
        sha256 (str | None): Expected SHA-256 hex digest.
        md5 (str | None): Expected MD5 hex digest, for sources that only publish MD5.
        timeout (float): Per-request socket timeout in seconds.
        retries (int): Attempts for transient network errors (``URLError`` or HTTP 5xx).

    Returns:
        Path: ``dest``.

    Raises:
        ValueError: If ``url`` is not HTTPS.
        RuntimeError: If the downloaded file does not match the expected checksum.
    """
    dest = Path(dest)
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"refusing non-HTTPS weights URL {redact_url(url)!r}")
    if _verified_in_place(dest, sha256, md5):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    temps = f"{glob.escape(str(dest.parent))}/.{glob.escape(dest.name)}.*.tmp"
    with file_lock(dest.parent / f".{dest.name}.download.lock", progress=temps):
        if _verified_in_place(dest, sha256, md5):
            return dest
        for attempt in range(1, retries + 1):
            # mkstemp, not a predictable name: on a shared cache another user must not be able
            # to preplant a symlink that this download would follow and truncate
            fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
            tmp = Path(tmp_name)
            handle = os.fdopen(fd, "wb")
            try:
                logger.info("Downloading %s to %s", redact_url(url), dest)
                request = urllib.request.Request(url)  # noqa: S310 -- scheme checked above (HTTPS only)
                # The worker resolves file-based tokens (hf auth login, git credentials) into HF_TOKEN
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if token and _is_trusted_hf_url(url):
                    request.add_header("Authorization", f"Bearer {token}")
                opener = urllib.request.build_opener(_SecureRedirectHandler())
                with handle:
                    with opener.open(request, timeout=timeout) as response:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                    handle.flush()
                    os.fchmod(handle.fileno(), _shared_cache_file_mode(dest.parent))
                if not _verified(tmp, sha256, md5):
                    raise RuntimeError(f"checksum mismatch for {redact_url(url)} (expected {sha256 or md5})")
                os.replace(tmp, dest)
                if sha256 or md5:
                    # The rename keeps the inode just hashed; stamp it as it now sits at dest
                    _remember_verified(dest, _stamp_line(dest, sha256, md5))
                return dest
            except (urllib.error.URLError, TimeoutError) as err:
                is_client_error = isinstance(err, urllib.error.HTTPError) and err.code < 500
                if is_client_error or attempt == retries:
                    raise
                logger.warning("Download of %s failed (%s); retry %d/%d", redact_url(url), err, attempt, retries - 1)
                time.sleep(5 * attempt)
            finally:
                handle.close()
                with contextlib.suppress(FileNotFoundError):
                    tmp.unlink()
    return dest


def download_archive(
    url: str,
    dest_dir: str | Path,
    *,
    marker: str,
    sha256: str | None = None,
    md5: str | None = None,
) -> Path:
    """Download and unpack an archive into ``dest_dir`` once, so readers never see a partial tree.

    ``dest_dir`` counts as complete while every file the archive unpacked is there (see
    ``ensure_complete``), so a tree that lost files since (e.g. to a scratch purge) is unpacked again.
    A tree unpacked before files were recorded is kept, and recorded, when it holds ``marker``.
    Otherwise, under a lock, the archive is fetched with ``download_weights``, unpacked into a staging
    directory beside ``dest_dir``, checked for ``marker``, and swapped into place; the archive is
    then deleted.

    Args:
        url (str): HTTPS URL of a ``.tar``, ``.tar.gz``, ``.tgz``, or ``.zip`` archive.
        dest_dir (str | Path): Directory that receives the archive's contents.
        marker (str): Path, relative to ``dest_dir``, that the archive must contain.
        sha256 (str | None): Expected SHA-256 of the archive.
        md5 (str | None): Expected MD5 of the archive.

    Returns:
        Path: ``dest_dir``.

    Raises:
        RuntimeError: If the unpacked archive does not contain ``marker``.
    """
    dest_dir = Path(dest_dir)
    name = Path(urllib.parse.urlparse(url).path).name
    fmt = _archive_format(name)

    def unpack(staging: Path) -> None:
        archive = download_weights(url, dest_dir.parent / f".{dest_dir.name}.{name}", sha256=sha256, md5=md5)
        try:
            try:
                # "data" refuses absolute paths, ../ escapes, and device files in the archive
                shutil.unpack_archive(str(archive), str(staging), format=fmt, filter="data")
            except TypeError:  # Python without extraction filters (< 3.12 and unpatched)
                shutil.unpack_archive(str(archive), str(staging), format=fmt)
        finally:
            archive.unlink(missing_ok=True)
            _stamp_path(archive).unlink(missing_ok=True)
        if not (staging / marker).exists():
            raise RuntimeError(f"archive from {redact_url(url)} does not contain {marker}")

    # The holder's archive download (and its temp file) and staging dir all sit beside dest_dir
    staged = f"{glob.escape(str(dest_dir.parent))}/.*{glob.escape(dest_dir.name)}.*"
    # A tree unpacked before trees were recorded is kept when it holds ``marker``, the old check
    return ensure_complete(dest_dir, unpack, stage=True, progress=staged, adopt=lambda d: (d / marker).exists())


def _archive_format(name: str) -> str:
    """Return the ``shutil.unpack_archive`` format for an archive filename."""
    if name.endswith((".tar.gz", ".tgz")):
        return "gztar"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    raise ValueError(f"unsupported archive type: {name}")


def link_to_cache(link: str | Path, target: str | Path) -> Path:
    """Make ``link`` a symlink to ``target``, replacing a stale link or a directory in its place.

    For libraries that only look for weights at a hardcoded path inside their installed package:
    the files stay in the managed cache, and the package path points at them. A link left over
    from before a cache move is repointed.

    Args:
        link (str | Path): The path the library reads, e.g. ``site-packages/lib/model-weights``.
        target (str | Path): The weights directory in the managed cache.

    Returns:
        Path: ``link``.
    """
    link, target = Path(link), Path(target)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return link
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    elif link.exists():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return link
