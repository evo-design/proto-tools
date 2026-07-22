"""PARADE standalone inference implementation for venv execution."""

import contextlib
import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import parade_scoring
from standalone_helpers import (
    get_logger,
    move_model_to_device,
    resolve_weights_dir,
    serialize_output,
    set_torch_seed,
)

logger = get_logger(__name__)

LOCK_TIMEOUT_SECONDS = 600


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-process advisory lock via ``fcntl.flock``; concurrent workers don't race downloads.

    The kernel releases an ``flock`` automatically when the holding process exits, so a crashed
    downloader leaves no stale lock — no PID inspection, no inspect-then-unlink race, and no
    ten-minute wedge. The lock is tied to the open file description (mutually exclusive across
    both processes and threads), and the 0-byte lock file is intentionally never unlinked, which
    would reintroduce a delete-the-inode-someone-else-just-locked race.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as err:
                if time.monotonic() - started > LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(f"parade: timed out waiting for lock {lock_path}") from err
                time.sleep(1)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular_cache_file(path: Path) -> bool:
    """Return whether a cache entry is a regular file without following symlinks."""
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _installed_cache_file_mode(cache_dir: Path) -> int:
    """Return a private-writable mode readable by principals allowed through the cache directory."""
    directory_mode = cache_dir.stat().st_mode
    mode = stat.S_IRUSR | stat.S_IWUSR
    if directory_mode & stat.S_IXGRP:
        mode |= stat.S_IRGRP
    if directory_mode & stat.S_IXOTH:
        mode |= stat.S_IROTH
    return mode


def _weights_dir() -> Path:
    resolved = resolve_weights_dir("parade")
    if resolved:
        return Path(resolved)
    fallback = Path(tempfile.gettempdir()) / "proto_parade_weights"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _is_trusted_hf_url(url: str) -> bool:
    """True only for an HTTPS URL on ``huggingface.co`` or a real subdomain.

    Requires HTTPS (never send the bearer token over plaintext) and an exact host or a real
    ``.huggingface.co`` subdomain — ``endswith`` alone would match ``evil-huggingface.co``.
    """
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co"))


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Abort insecure redirects and drop ``Authorization`` when a hop leaves trusted HF.

    A checkpoint is unpickled after download, so a network attacker who can force an
    HTTPS→HTTP downgrade (then MITM the plaintext hop) could swap in an arbitrary pickle —
    stripping the token is not enough. Refuse any non-HTTPS redirect destination outright,
    and for HTTPS hops that leave a trusted HF host, drop the bearer token that Python's
    default handler would otherwise forward verbatim.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise urllib.error.HTTPError(
                newurl, code, "parade: refusing insecure (non-HTTPS) checkpoint redirect", headers, fp
            )
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _is_trusted_hf_url(newurl):
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def _redact_url(url: str) -> str:
    """Strip path/query credentials and userinfo so signed URLs don't reach logs.

    HTTPS checkpoint URLs may carry credentials in userinfo, path segments, or a signed
    ``?token=…`` query; the full value must never land in a log line or exception message.
    Preserve only the origin for diagnosis and replace every non-empty path with a fixed marker.
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


def _download_checkpoint(url: str, expected_md5: str, filename: str, cache_dir: Path) -> Path:
    """Download a PARADE checkpoint into the cache, verifying its MD5 checksum."""
    if not url:
        raise FileNotFoundError("parade: checkpoint_path is unavailable and checkpoint_url is empty")
    if (
        not filename
        or filename != Path(filename).name
        or "\\" in filename
        or any(ord(character) < 32 or ord(character) == 127 for character in filename)
        or not filename.endswith(".ckpt")
    ):
        raise ValueError("parade: checkpoint_filename must be a .ckpt basename without path separators")
    if urllib.parse.urlparse(url).scheme != "https":
        # The checkpoint is unpickled; never fetch it over a downgradeable/MITM-able transport.
        raise ValueError(f"parade: checkpoint URL must use https, got {_redact_url(url)!r}")

    dest = cache_dir / filename
    if _is_regular_cache_file(dest) and (not expected_md5 or _md5(dest) == expected_md5):
        return dest

    with _file_lock(cache_dir / f".{filename}.download.lock"):
        if _is_regular_cache_file(dest) and (not expected_md5 or _md5(dest) == expected_md5):
            return dest
        # Use an exclusively created 0600 file rather than a predictable PID-derived path.
        # A shared writable cache must not let another process preplant a symlink that this
        # downloader then follows and truncates outside the cache.
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
        tmp = Path(tmp_name)
        tmp_handle = os.fdopen(tmp_fd, "wb")
        try:
            logger.info("Downloading PARADE checkpoint from %s to %s", _redact_url(url), dest)
            request = urllib.request.Request(url)
            # Only over HTTPS to a real HF host; the redirect handler strips the token if a
            # later hop leaves that trust boundary (plaintext or a non-HF destination).
            if _is_trusted_hf_url(url) and os.environ.get("HF_TOKEN"):
                request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            opener = urllib.request.build_opener(_SecureRedirectHandler())
            # Wrap the already-open exclusive descriptor directly; reopening by pathname would
            # reintroduce a symlink-swap race between creation and writing.
            with tmp_handle as handle:
                with opener.open(request, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                handle.flush()
                if expected_md5:
                    observed = _md5(tmp)
                    if observed != expected_md5:
                        raise RuntimeError(
                            f"parade: checkpoint checksum mismatch for {_redact_url(url)}; "
                            f"expected {expected_md5}, got {observed}"
                        )
                # Keep the partial file at mkstemp's 0600 while downloading. Once verified,
                # make it readable to the same group/world principals that can traverse the
                # cache directory, preserving documented cross-user model-cache reuse.
                os.fchmod(handle.fileno(), _installed_cache_file_mode(cache_dir))
            os.replace(tmp, dest)
        finally:
            with contextlib.suppress(OSError):
                tmp_handle.close()
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
    return dest


def _resolve_checkpoint(input_dict: dict[str, Any]) -> str:
    """Resolve a local checkpoint path, downloading the pinned artifact if needed."""
    local = input_dict.get("checkpoint_path") or ""
    if local:
        candidate = Path(local).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"parade: checkpoint_path does not exist: {local}")

    checkpoint = _download_checkpoint(
        url=input_dict["checkpoint_url"],
        expected_md5=input_dict.get("checkpoint_md5", ""),
        filename=input_dict["checkpoint_filename"],
        cache_dir=_weights_dir(),
    )
    return str(checkpoint)


class ParadeModel:
    """Caches a loaded PARADE LegNet module and delegates inference to ``parade_scoring``."""

    def __init__(self) -> None:
        """Initialize an unloaded PARADE model wrapper."""
        self.model: Any = None
        self.device: str | None = None
        self.checkpoint_key: str | None = None

    def load(self, *, checkpoint_path: str, device: str, verbose: bool = False) -> None:
        """Load a PARADE checkpoint (once per checkpoint) and move it to the device."""
        if self.model is not None and self.checkpoint_key == checkpoint_path:
            self.to_device(device)
            return
        if verbose:
            logger.info("Loading PARADE checkpoint from %s", checkpoint_path)
        self.model = parade_scoring.load_legnet(checkpoint_path)
        self.checkpoint_key = checkpoint_path
        self.device = "cpu"
        self.to_device(device)

    def to_device(self, device: str) -> None:
        """Move the cached LegNet module to a device."""
        if self.model is None or self.device == device:
            return
        previous_device = self.device or "cpu"
        self.model = move_model_to_device(self.model, previous_device, device)
        self.device = device

    def score_activity(
        self, *, sequences: list[str], construct_type: str, cell_types: list[str], batch_size: int, device: str
    ) -> list[dict[str, float]]:
        """Predict per-cell-type activity for each sequence."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(device)
        return parade_scoring.predict_activity(  # type: ignore[no-any-return]
            self.model,
            sequences=sequences,
            construct_type=construct_type,
            cell_types=cell_types,
            batch_size=batch_size,
            device=device,
        )

    def score_stability(self, *, sequences: list[str], batch_size: int, device: str) -> list[float]:
        """Predict the RNA/gDNA log-ratio for each 3' UTR sequence."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(device)
        return parade_scoring.predict_stability(  # type: ignore[no-any-return]
            self.model, sequences=sequences, batch_size=batch_size, device=device
        )

    def compute_gradient(self, **kwargs: Any) -> dict[str, Any]:
        """Compute a differentiable PARADE activity loss and optional gradient."""
        if self.model is None:
            raise ValueError("parade: model is not loaded")
        self.to_device(kwargs["device"])
        return parade_scoring.compute_gradient(self.model, **kwargs)  # type: ignore[no-any-return]


_model = ParadeModel()


def dispatch(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Entry point for both persistent-worker and one-shot execution."""
    set_torch_seed(input_dict.get("seed"))
    operation = input_dict["operation"]
    if operation not in {"activity", "stability", "gradient"}:
        raise ValueError(f"parade: unknown operation {operation!r}; valid: ['activity', 'stability', 'gradient']")

    checkpoint_path = _resolve_checkpoint(input_dict)
    _model.load(
        checkpoint_path=checkpoint_path,
        device=input_dict["device"],
        verbose=bool(input_dict.get("verbose", False)),
    )

    if operation == "gradient":
        return _model.compute_gradient(
            logits_list=input_dict["logits"],
            temperature=float(input_dict["temperature"]),
            construct_type=input_dict["construct_type"],
            loss_terms=input_dict["loss_terms"],
            soft=float(input_dict.get("soft", 1.0)),
            hard=float(input_dict.get("hard", 0.0)),
            compute_gradient=bool(input_dict.get("compute_gradient", True)),
            device=input_dict["device"],
        )

    batch_size = max(1, int(input_dict.get("batch_size", 1)))
    if operation == "activity":
        return {
            "scores": _model.score_activity(
                sequences=input_dict["sequences"],
                construct_type=input_dict["construct_type"],
                cell_types=input_dict["cell_types"],
                batch_size=batch_size,
                device=input_dict["device"],
            )
        }
    return {
        "log_ratios": _model.score_stability(
            sequences=input_dict["sequences"],
            batch_size=batch_size,
            device=input_dict["device"],
        )
    }


def to_device(device: str) -> dict[str, Any]:
    """Move the model to the specified device."""
    _model.to_device(device)
    return {"success": True, "device": device}


def get_memory_stats() -> dict[str, Any]:
    """Report GPU memory usage."""
    from standalone_helpers import get_pytorch_memory_stats

    return get_pytorch_memory_stats(_model.device or 0)  # type: ignore[no-any-return]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("parade: usage: python inference.py <input_json_path> <output_json_path>")

    with open(sys.argv[1]) as f:
        input_data = json.load(f)

    result = dispatch(input_data)

    with open(sys.argv[2], "w") as f:
        json.dump(serialize_output(result), f)
