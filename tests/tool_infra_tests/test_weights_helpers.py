"""Tests for the standalone weights download and env-export helpers."""

import fcntl
import hashlib
import io
import logging
import os
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers import weights
from proto_tools.utils.standalone_helpers_source.standalone_helpers.weights import (
    _SecureRedirectHandler,
    download_archive,
    download_weights,
    export_weights_dir,
    file_lock,
    link_to_cache,
    redact_url,
    require_weights_dir,
)

_PAYLOAD = b"model weights"
_SHA256 = hashlib.sha256(_PAYLOAD).hexdigest()


class _FakeOpener:
    """Stands in for ``urllib.request.build_opener(...)``; records requests and serves ``_PAYLOAD``."""

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.requests: list[urllib.request.Request] = []
        self.failures = list(failures or [])

    def open(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        self.requests.append(request)
        if self.failures:
            raise self.failures.pop(0)
        return io.BytesIO(_PAYLOAD)


@pytest.fixture
def opener(monkeypatch) -> _FakeOpener:
    fake = _FakeOpener()
    monkeypatch.setattr(weights.urllib.request, "build_opener", lambda *handlers: fake)
    monkeypatch.setattr(weights.time, "sleep", lambda _s: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return fake


def test_download_verifies_and_skips_when_present(tmp_path: Path, opener: _FakeOpener):
    """A verified download is renamed into place, and a second call makes no request."""
    dest = tmp_path / "sub" / "model.pt"

    assert download_weights("https://example.org/model.pt", dest, sha256=_SHA256) == dest
    download_weights("https://example.org/model.pt", dest, sha256=_SHA256)

    assert dest.read_bytes() == _PAYLOAD
    assert len(opener.requests) == 1
    assert [p.name for p in dest.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_download_redownloads_corrupt_file(tmp_path: Path, opener: _FakeOpener):
    """An existing file that fails its checksum is replaced."""
    dest = tmp_path / "model.pt"
    dest.write_bytes(b"truncated")

    download_weights("https://example.org/model.pt", dest, sha256=_SHA256)

    assert dest.read_bytes() == _PAYLOAD


def test_download_checksum_mismatch_leaves_nothing(tmp_path: Path, opener: _FakeOpener):
    """A payload that fails verification never reaches ``dest`` and leaves no temp file."""
    dest = tmp_path / "model.pt"

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_weights("https://example.org/model.pt", dest, md5="0" * 32)

    assert not dest.exists()
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_download_refuses_plain_http(tmp_path: Path, opener: _FakeOpener):
    """Weights are often unpickled, so a downgradeable transport is refused before any request."""
    with pytest.raises(ValueError, match="non-HTTPS"):
        download_weights("http://example.org/model.pt", tmp_path / "model.pt")
    assert opener.requests == []


def test_download_retries_transient_errors(tmp_path: Path, opener: _FakeOpener):
    """Connection errors and 5xx responses are retried; the last attempt's success wins."""
    opener.failures = [urllib.error.URLError("reset"), urllib.error.HTTPError("u", 503, "busy", {}, None)]

    download_weights("https://example.org/model.pt", tmp_path / "model.pt", retries=3)

    assert len(opener.requests) == 3


def test_download_does_not_retry_client_errors(tmp_path: Path, opener: _FakeOpener):
    """A 404 is permanent, so it raises on the first attempt."""
    opener.failures = [urllib.error.HTTPError("u", 404, "missing", {}, None)]

    with pytest.raises(urllib.error.HTTPError):
        download_weights("https://example.org/model.pt", tmp_path / "model.pt", retries=3)
    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    "url, sends_token",
    [
        ("https://huggingface.co/org/repo/resolve/main/m.pt", True),
        ("https://cdn-lfs.huggingface.co/m.pt", True),
        ("https://evil-huggingface.co/m.pt", False),
        ("https://zenodo.org/m.pt", False),
    ],
)
def test_hf_token_only_sent_to_huggingface(monkeypatch, tmp_path: Path, opener: _FakeOpener, url, sends_token):
    """HF_TOKEN goes to huggingface.co and its real subdomains only."""
    monkeypatch.setenv("HF_TOKEN", "secret")

    download_weights(url, tmp_path / "m.pt")

    assert (opener.requests[0].get_header("Authorization") == "Bearer secret") is sends_token


def test_redirect_handler_strips_token_off_huggingface_and_refuses_http():
    """Redirects keep the token only within huggingface.co and never downgrade to HTTP."""
    handler = _SecureRedirectHandler()
    request = urllib.request.Request("https://huggingface.co/m.pt", headers={"Authorization": "Bearer secret"})

    to_hf = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn-lfs.huggingface.co/m.pt")
    to_s3 = handler.redirect_request(request, None, 302, "Found", {}, "https://s3.amazonaws.com/m.pt")

    assert to_hf.get_header("Authorization") == "Bearer secret"
    assert to_s3.get_header("Authorization") is None
    with pytest.raises(urllib.error.HTTPError, match="non-HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://example.org/m.pt")


def test_redact_url_hides_credentials():
    """Signed paths, queries, and userinfo never reach a log line."""
    assert redact_url("https://user:pw@host.org:8443/signed/path?token=abc") == (
        "https://host.org:8443/<redacted-path>?REDACTED"
    )


@pytest.mark.parametrize(
    "existing, prepend, expected",
    [
        ("/other", False, "{dir}"),
        ("/other", True, "{dir}" + os.pathsep + "/other"),
        ("{dir}" + os.pathsep + "/other", True, "{dir}" + os.pathsep + "/other"),
        (None, True, "{dir}"),
    ],
)
def test_export_weights_dir(monkeypatch, tmp_path: Path, existing, prepend, expected):
    """The managed directory replaces the value, or leads a search path without duplicating itself."""
    managed = str(tmp_path / "w")
    monkeypatch.setenv("PROTO_MYTOOL_WEIGHTS_DIR", managed)
    if existing is None:
        monkeypatch.delenv("LIB_DIRS", raising=False)
    else:
        monkeypatch.setenv("LIB_DIRS", existing.format(dir=managed))

    assert export_weights_dir("LIB_DIRS", "mytool", prepend=prepend) == managed
    assert os.environ["LIB_DIRS"] == expected.format(dir=managed)


def test_require_weights_dir_raises_without_location(monkeypatch):
    """NONE mode outside a venv has no weights directory, which require_weights_dir reports."""
    monkeypatch.setenv("PROTO_MODEL_CACHE", "NONE")
    monkeypatch.delenv("PROTO_MYTOOL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("TOOL_VENV_PATH", raising=False)
    monkeypatch.delenv("VENV_PATH", raising=False)

    with pytest.raises(RuntimeError, match="PROTO_MYTOOL_WEIGHTS_DIR"):
        require_weights_dir("mytool")


class _InterruptingOpener:
    """An opener whose response read raises ``KeyboardInterrupt`` mid-download."""

    def open(self, request: urllib.request.Request, timeout: float) -> "_InterruptingOpener":
        return self

    def __enter__(self) -> "_InterruptingOpener":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self, _size: int) -> bytes:
        raise KeyboardInterrupt


def test_download_lock_excludes_and_releases(tmp_path: Path):
    """A held flock blocks a second holder until released; the lock file is kept."""
    lock_path = tmp_path / "d.lock"
    blocker = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(blocker, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError), file_lock(lock_path, timeout=0):
            pass
    finally:
        fcntl.flock(blocker, fcntl.LOCK_UN)
        os.close(blocker)

    with file_lock(lock_path):
        assert lock_path.exists()


_HELPERS_SH = Path(weights.__file__).resolve().parent.parent / "standalone_helpers.sh"
_needs_flock = pytest.mark.skipif(shutil.which("flock") is None, reason="flock(1) not available")


def _hold_with_bash(lock_path: Path, script: str) -> subprocess.Popen:
    """Hold ``lock_path`` through proto_with_lock while ``script`` runs; return once held."""
    held = lock_path.with_name(lock_path.name + ".held")
    proc = subprocess.Popen(
        ["bash", "-c", f'source "{_HELPERS_SH}"; proto_with_lock "{lock_path}" bash -c \'touch "{held}"; {script}\'']
    )
    for _ in range(50):
        if held.exists():
            return proc
        time.sleep(0.1)
    proc.kill()
    raise AssertionError("bash holder never took the lock")


def test_file_lock_waits_while_holder_makes_progress(tmp_path: Path, monkeypatch):
    """A holder whose progress file keeps growing is waited out, past the stall window."""
    monkeypatch.setenv("PROTO_LOCK_POLL_SECONDS", "0.2")
    monkeypatch.setenv("PROTO_LOCK_STALL_SECONDS", "0.6")
    lock_path, growing = tmp_path / ".d.lock", tmp_path / "d.tmp"

    def hold() -> None:
        with file_lock(lock_path):
            for _ in range(10):
                with open(growing, "ab") as f:
                    f.write(b"x")
                time.sleep(0.2)

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.1)
    started = time.monotonic()
    with file_lock(lock_path, progress=growing):
        waited = time.monotonic() - started
    holder.join()
    assert waited > 1.5  # well past the 0.6 s stall window


def test_file_lock_gives_up_on_a_stalled_holder(tmp_path: Path, monkeypatch):
    """A holder whose progress path stops growing is reported instead of waited on forever."""
    monkeypatch.setenv("PROTO_LOCK_POLL_SECONDS", "0.2")
    monkeypatch.setenv("PROTO_LOCK_STALL_SECONDS", "0.6")
    lock_path, growing = tmp_path / ".d.lock", tmp_path / "d.tmp"
    growing.write_bytes(b"x")
    release = threading.Event()

    def hold() -> None:
        with file_lock(lock_path):
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.1)
    try:
        with pytest.raises(TimeoutError, match="made no progress"), file_lock(lock_path, progress=growing):
            pass
    finally:
        release.set()
        holder.join()


def test_file_lock_has_no_cap_by_default(tmp_path: Path, monkeypatch):
    """Without a timeout or PROTO_LOCK_TIMEOUT_SECONDS, a waiter outlasts a slow holder."""
    monkeypatch.delenv("PROTO_LOCK_TIMEOUT_SECONDS", raising=False)
    lock_path = tmp_path / ".d.lock"

    def hold() -> None:
        with file_lock(lock_path):
            time.sleep(1.5)

    holder = threading.Thread(target=hold)
    holder.start()
    time.sleep(0.1)
    with file_lock(lock_path):
        pass
    holder.join()


def test_file_lock_is_reentrant_in_one_thread(tmp_path: Path):
    """Taking a lock this thread already holds runs straight through instead of deadlocking."""
    with file_lock(tmp_path / ".d.lock", timeout=1), file_lock(tmp_path / ".d.lock", timeout=1):
        pass


def test_file_lock_still_excludes_other_threads(tmp_path: Path):
    """Re-entrancy is per thread: another thread in the same process still waits."""
    lock_path = tmp_path / ".d.lock"
    with file_lock(lock_path):
        errors: list[BaseException] = []

        def other() -> None:
            try:
                with file_lock(lock_path, timeout=0.5):
                    pass
            except TimeoutError as err:
                errors.append(err)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
    assert len(errors) == 1


@_needs_flock
def test_python_waits_for_a_bash_holder(tmp_path: Path):
    """proto_with_lock in a setup.sh and file_lock in a worker take turns on one lock file."""
    lock_path = tmp_path / ".model.pt.download.lock"
    holder = _hold_with_bash(lock_path, "sleep 2")
    started = time.monotonic()
    with file_lock(lock_path):
        waited = time.monotonic() - started
    holder.wait()
    assert waited > 1.5


@_needs_flock
def test_bash_waits_for_a_python_holder(tmp_path: Path):
    """A setup.sh waits while a Python worker holds the same download lock."""
    lock_path = tmp_path / ".model.pt.download.lock"
    with file_lock(lock_path):
        started = time.monotonic()
        waiter = subprocess.Popen(
            ["bash", "-c", f'source "{_HELPERS_SH}"; proto_with_lock "{lock_path}" true'],
            env={**os.environ, "PROTO_LOCK_POLL_SECONDS": "1"},
        )
        time.sleep(1.5)
        assert waiter.poll() is None, "bash took a lock Python still held"
    assert waiter.wait(timeout=10) == 0
    assert time.monotonic() - started > 1.5


@_needs_flock
def test_python_inside_a_bash_lock_does_not_deadlock(tmp_path: Path):
    """A Python script run under the same proto_with_lock lock (setup.sh calling Python) enters it."""
    lock_path = tmp_path / ".model.pt.download.lock"
    script = tmp_path / "inner.py"
    script.write_text(
        "from standalone_helpers.weights import file_lock\n"
        f"with file_lock({str(lock_path)!r}, timeout=2):\n"
        "    print('entered')\n"
    )
    result = subprocess.run(
        ["bash", "-c", f'source "{_HELPERS_SH}"; proto_with_lock "{lock_path}" python3 "{script}"'],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(_HELPERS_SH.parent)},
    )
    assert result.returncode == 0, result.stderr
    assert "entered" in result.stdout


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://host.co/x.ckpt", "https://host.co/<redacted-path>"),
        ("https://[::1]:8443/x.ckpt", "https://[::1]:8443/<redacted-path>"),
        ("https://user:secret@host.co:8443/x.ckpt#frag", "https://host.co:8443/<redacted-path>"),
        ("https://example.com:99999/x.ckpt", "<unparseable-url>"),
    ],
)
def test_redact_url_edge_cases(url, expected):
    """IPv6 hosts and ports survive; userinfo and fragments don't; a bad port is not echoed."""
    assert redact_url(url) == expected


def test_url_secrets_absent_from_log_and_error(tmp_path: Path, opener: _FakeOpener):
    """Credentials carried in a URL's path or query never reach a log line or exception."""
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Handler()
    weights.logger.addHandler(handler)
    url = "https://downloads.example/private/PATH_TOKEN/model.ckpt?token=QUERY_TOKEN"
    try:
        with pytest.raises(RuntimeError) as err:
            download_weights(url, tmp_path / "model.ckpt", md5="0" * 32)
    finally:
        weights.logger.removeHandler(handler)

    rendered = str(err.value) + "\n".join(records)
    assert "downloads.example" in rendered
    assert "PATH_TOKEN" not in rendered and "QUERY_TOKEN" not in rendered


def test_download_ignores_planted_temp_symlink(tmp_path: Path, opener: _FakeOpener):
    """A symlink planted at a guessable temp name cannot redirect the write; shared-cache mode applies."""
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"safe")
    planted = tmp_path / f".model.ckpt.{os.getpid()}.tmp"
    planted.symlink_to(victim)
    tmp_path.chmod(0o770)

    dest = download_weights("https://example.com/model.ckpt", tmp_path / "model.ckpt")

    assert dest.read_bytes() == _PAYLOAD
    assert dest.stat().st_mode & 0o777 == 0o640
    assert victim.read_bytes() == b"safe"


def test_download_replaces_destination_symlink(tmp_path: Path, opener: _FakeOpener):
    """A symlink at ``dest`` is never accepted as a cache hit; it is replaced by the real file."""
    victim = tmp_path / "attacker.ckpt"
    victim.write_bytes(b"attacker-controlled-pickle")
    dest = tmp_path / "model.ckpt"
    dest.symlink_to(victim)

    download_weights("https://example.com/model.ckpt", dest)

    assert not dest.is_symlink()
    assert dest.read_bytes() == _PAYLOAD
    assert victim.read_bytes() == b"attacker-controlled-pickle"


def test_download_interrupt_leaves_no_temp_file(monkeypatch, tmp_path: Path):
    """An interrupted read removes its temp file and never creates ``dest``."""
    monkeypatch.setattr(weights.urllib.request, "build_opener", lambda *handlers: _InterruptingOpener())

    with pytest.raises(KeyboardInterrupt):
        download_weights("https://example.com/model.ckpt", tmp_path / "model.ckpt")

    assert list(tmp_path.glob(".model.ckpt.*.tmp")) == []
    assert not (tmp_path / "model.ckpt").exists()


def _tar_gz_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_archive_unpacks_once(monkeypatch, tmp_path: Path, opener: _FakeOpener):
    """The archive is fetched and unpacked once; the marker makes later calls free."""
    archive = _tar_gz_bytes({"model.pt": b"w", "vocab.txt": b"v"})
    monkeypatch.setattr(
        opener, "open", lambda request, timeout: (opener.requests.append(request), io.BytesIO(archive))[1]
    )
    dest = tmp_path / "model-weights"

    assert download_archive("https://example.org/w.tar.gz", dest, marker="model.pt") == dest
    download_archive("https://example.org/w.tar.gz", dest, marker="model.pt")

    assert (dest / "model.pt").read_bytes() == b"w"
    assert len(opener.requests) == 1
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.endswith(".lock")) == ["model-weights"]


def test_download_archive_replaces_incomplete_extraction(monkeypatch, tmp_path: Path, opener: _FakeOpener):
    """A directory without the marker is a failed earlier attempt and is replaced."""
    archive = _tar_gz_bytes({"model.pt": b"w"})
    monkeypatch.setattr(opener, "open", lambda request, timeout: io.BytesIO(archive))
    dest = tmp_path / "model-weights"
    (dest / "partial").mkdir(parents=True)

    download_archive("https://example.org/w.tar.gz", dest, marker="model.pt")

    assert sorted(p.name for p in dest.iterdir() if not p.name.startswith(".")) == ["model.pt"]


def test_download_archive_unpacks_again_when_a_file_was_deleted(monkeypatch, tmp_path: Path, opener: _FakeOpener):
    """An unpacked tree that lost a file other than the marker (e.g. to a scratch purge) is unpacked again."""
    archive = _tar_gz_bytes({"model.pt": b"w", "vocab.txt": b"v"})
    monkeypatch.setattr(
        opener, "open", lambda request, timeout: (opener.requests.append(request), io.BytesIO(archive))[1]
    )
    dest = tmp_path / "model-weights"
    download_archive("https://example.org/w.tar.gz", dest, marker="model.pt")

    (dest / "vocab.txt").unlink()
    download_archive("https://example.org/w.tar.gz", dest, marker="model.pt")

    assert (dest / "vocab.txt").read_bytes() == b"v"
    assert len(opener.requests) == 2


def test_download_archive_keeps_a_tree_unpacked_before_trees_were_recorded(
    monkeypatch, tmp_path: Path, opener: _FakeOpener
):
    """An earlier, valid extraction is recorded as it is rather than downloaded again."""
    monkeypatch.setattr(opener, "open", lambda request, timeout: (opener.requests.append(request), io.BytesIO(b""))[1])
    dest = tmp_path / "model-weights"
    (dest).mkdir()
    (dest / "model.pt").write_bytes(b"w")

    download_archive("https://example.org/w.tar.gz", dest, marker="model.pt")

    assert opener.requests == []
    assert (dest / ".proto_complete").read_text() == "model.pt\n"


def test_download_archive_missing_marker_raises(monkeypatch, tmp_path: Path, opener: _FakeOpener):
    """An archive without the expected file never becomes ``dest_dir``."""
    archive = _tar_gz_bytes({"other.pt": b"w"})
    monkeypatch.setattr(opener, "open", lambda request, timeout: io.BytesIO(archive))

    with pytest.raises(RuntimeError, match=r"does not contain model\.pt"):
        download_archive("https://example.org/w.tar.gz", tmp_path / "model-weights", marker="model.pt")
    assert not (tmp_path / "model-weights").exists()


@pytest.mark.parametrize("existing", ["stale_link", "directory", "none", "correct_link"])
def test_link_to_cache_repoints(tmp_path: Path, existing):
    """The package path ends up a symlink to the cache whatever was there before."""
    target = tmp_path / "cache" / "weights"
    target.mkdir(parents=True)
    link = tmp_path / "site-packages" / "lib" / "model-weights"
    link.parent.mkdir(parents=True)
    if existing == "stale_link":
        link.symlink_to(tmp_path / "old-cache")
    elif existing == "directory":
        (link / "model.pt").parent.mkdir()
        (link / "model.pt").write_bytes(b"in-package copy")
    elif existing == "correct_link":
        link.symlink_to(target)

    link_to_cache(link, target)

    assert link.is_symlink() and link.resolve() == target.resolve()
