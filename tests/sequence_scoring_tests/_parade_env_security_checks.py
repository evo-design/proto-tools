"""In-env security checks for the PARADE checkpoint-download path.

These exercise the standalone ``inference`` module (which imports ``torch`` and so
is not importable in the base test environment). They run **inside** the built
parade standalone env via :func:`proto_tools.utils.run_in_env`, invoked by
``test_parade_checkpoint_download_security_runs_in_env`` in ``test_parade.py``.

``pytest`` is not installed in the standalone env, so these are plain-``assert``
checks with manual patching/tempdirs. ``main()`` runs every check, prints a
per-check status line, and exits non-zero if any fail (which ``run_in_env``
surfaces to the caller as ``RuntimeError``). The leading ``_`` keeps pytest from
collecting this file in the base process.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import tempfile
import traceback
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import inference  # standalone module; on PYTHONPATH inside the parade env


@contextlib.contextmanager
def _expect(exc: type[BaseException], match: str | None = None) -> Iterator[None]:
    """Assert that the block raises ``exc`` (optionally containing ``match``)."""
    try:
        yield
    except exc as error:
        if match is not None and match not in str(error):
            raise AssertionError(f"expected {match!r} in error, got {error!r}") from None
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


@contextlib.contextmanager
def _patched(obj: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily set ``obj.name = value``, restoring the original after."""
    sentinel = object()
    old = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, old)


@contextlib.contextmanager
def _env_var(name: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable, restoring the original after."""
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


@contextlib.contextmanager
def _capture(logger: logging.Logger) -> Iterator[list[str]]:
    """Capture INFO+ messages emitted on ``logger`` for the duration of the block."""
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Handler()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class _FakeResponse:
    """Minimal urlopen-style response yielding ``chunks`` then EOF."""

    def __init__(self, chunks: Sequence[bytes] = (b"checkpoint",)) -> None:
        self._chunks = list(chunks)
        self._index = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self, _size: int) -> bytes:
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _opener_returning(chunks: Sequence[bytes]) -> Any:
    """Build a fake opener factory whose ``.open()`` returns a ``_FakeResponse``."""

    class _Opener:
        def open(self, request: Any, timeout: float = 0) -> _FakeResponse:
            return _FakeResponse(chunks)

    return lambda *a, **k: _Opener()


def check_hf_token_only_sent_to_real_hf_hosts() -> None:
    """HF_TOKEN attaches only for huggingface.co and real subdomains, never look-alikes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        captured: dict[str, str | None] = {}

        class _Opener:
            def open(self, request: Any, timeout: float = 0) -> _FakeResponse:
                captured["auth"] = request.get_header("Authorization")
                return _FakeResponse(())

        def auth_for(url: str, name: str) -> str | None:
            captured.clear()
            inference._download_checkpoint(url, "", name, tmp_path)
            return captured["auth"]

        with (
            _patched(inference.urllib.request, "build_opener", lambda *a, **k: _Opener()),
            _patched(inference, "_md5", lambda _p: ""),
            _env_var("HF_TOKEN", "secret-token"),
        ):
            assert auth_for("https://evil-huggingface.co/x.ckpt", "evil.ckpt") is None
            assert auth_for("https://huggingface.co.attacker.com/x.ckpt", "evil2.ckpt") is None
            assert auth_for("https://huggingface.co/x.ckpt", "hf.ckpt") == "Bearer secret-token"
            assert auth_for("https://cdn-lfs.huggingface.co/x.ckpt", "hf2.ckpt") == "Bearer secret-token"
            with _expect(ValueError, "https"):
                inference._download_checkpoint("http://huggingface.co/x.ckpt", "", "plain.ckpt", tmp_path)


def check_redirect_aborts_downgrade_and_strips_offhost_auth() -> None:
    """Non-HTTPS redirects abort; an off-host HTTPS redirect strips the token but proceeds."""
    handler = inference._SecureRedirectHandler()

    def redirect_to(newurl: str) -> str | None:
        req = urllib.request.Request("https://huggingface.co/x.ckpt")
        req.add_header("Authorization", "Bearer secret-token")
        new = handler.redirect_request(req, None, 302, "Found", {}, newurl)
        assert new is not None
        return new.get_header("Authorization")

    assert redirect_to("https://attacker.example/x.ckpt") is None
    with _expect(urllib.error.HTTPError):
        redirect_to("http://huggingface.co/x.ckpt")
    assert redirect_to("https://cdn-lfs.huggingface.co/x.ckpt") == "Bearer secret-token"


def check_download_lock_is_mutually_exclusive_and_auto_releases() -> None:
    """Advisory flock gives real mutual exclusion; a held lock blocks, release re-opens it."""
    with tempfile.TemporaryDirectory() as tmp:
        lock_path = Path(tmp) / "d.lock"
        with inference._file_lock(lock_path):
            pass
        with inference._file_lock(lock_path):
            pass
        assert lock_path.exists()

        blocker = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(blocker, fcntl.LOCK_EX)
            with _patched(inference, "LOCK_TIMEOUT_SECONDS", 0), _expect(TimeoutError):
                with inference._file_lock(lock_path):
                    pass
        finally:
            fcntl.flock(blocker, fcntl.LOCK_UN)
            os.close(blocker)

        with inference._file_lock(lock_path):
            assert lock_path.exists()


def check_redact_url_strips_credentials_and_query() -> None:
    """_redact_url drops userinfo, query, and fragment so secrets never reach logs/errors."""
    assert (
        inference._redact_url("https://user:secret@host.co/p/x.ckpt?sig=abc")
        == "https://host.co/<redacted-path>?REDACTED"
    )
    assert inference._redact_url("https://host.co/x.ckpt") == "https://host.co/<redacted-path>"
    redacted = inference._redact_url("https://user:secret@host.co:8443/x.ckpt#frag")
    assert "secret" not in redacted and "user" not in redacted and "frag" not in redacted
    assert "host.co:8443" in redacted
    assert inference._redact_url("https://[::1]:8443/x.ckpt") == "https://[::1]:8443/<redacted-path>"
    assert inference._redact_url("https://example.com:99999/x.ckpt") == "<unparseable-url>"


def check_checkpoint_url_path_secret_is_redacted_from_log_and_error() -> None:
    """Path/query-carried URL credentials never appear in download logs or checksum errors."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        path_secret = "PATH_TOKEN_7e48"
        query_secret = "QUERY_TOKEN_91bc"
        url = f"https://downloads.example/private/{path_secret}/model.ckpt?token={query_secret}"

        with (
            _patched(inference.urllib.request, "build_opener", _opener_returning((b"checkpoint",))),
            _capture(inference.logger) as logs,
        ):
            error_text = None
            try:
                inference._download_checkpoint(url, "0" * 32, "model.ckpt", tmp_path)
            except RuntimeError as error:
                error_text = str(error)
            assert error_text is not None, "expected RuntimeError from checksum mismatch"

        rendered = error_text + "\n" + "\n".join(logs)
        assert path_secret not in rendered
        assert query_secret not in rendered
        assert "downloads.example" in rendered
        assert "<redacted-path>" in rendered


def check_download_does_not_follow_predictable_temp_symlink() -> None:
    """A preplanted legacy PID-derived temp symlink cannot redirect checkpoint writes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        victim = tmp_path / "victim.txt"
        victim.write_bytes(b"safe")
        legacy_tmp = tmp_path / f".model.ckpt.{os.getpid()}.tmp"
        legacy_tmp.symlink_to(victim)
        tmp_path.chmod(0o770)

        with _patched(inference.urllib.request, "build_opener", _opener_returning((b"checkpoint",))):
            downloaded = inference._download_checkpoint("https://example.com/model.ckpt", "", "model.ckpt", tmp_path)

        assert downloaded.read_bytes() == b"checkpoint"
        assert downloaded.stat().st_mode & 0o777 == 0o640
        assert victim.read_bytes() == b"safe"
        assert legacy_tmp.is_symlink()


def check_download_does_not_accept_destination_symlink() -> None:
    """A destination symlink is replaced, never accepted as an unverified cache hit."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        victim = tmp_path / "attacker-controlled.ckpt"
        victim.write_bytes(b"attacker-controlled-pickle")
        destination = tmp_path / "model.ckpt"
        destination.symlink_to(victim)

        with _patched(inference.urllib.request, "build_opener", _opener_returning((b"downloaded-checkpoint",))):
            downloaded = inference._download_checkpoint(
                "https://example.com/model.ckpt", "", destination.name, tmp_path
            )

        assert downloaded == destination
        assert not downloaded.is_symlink()
        assert downloaded.read_bytes() == b"downloaded-checkpoint"
        assert victim.read_bytes() == b"attacker-controlled-pickle"


def check_download_cleans_temp_file_on_interrupt() -> None:
    """An interrupted response read does not leave a checkpoint temporary file behind."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        class _Interrupted:
            def __enter__(self) -> _Interrupted:
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def read(self, _size: int) -> bytes:
                raise KeyboardInterrupt

        class _Opener:
            def open(self, request: Any, timeout: float = 0) -> _Interrupted:
                return _Interrupted()

        with _patched(inference.urllib.request, "build_opener", lambda *a, **k: _Opener()):
            with _expect(KeyboardInterrupt):
                inference._download_checkpoint("https://example.com/model.ckpt", "", "model.ckpt", tmp_path)

        assert list(tmp_path.glob(".model.ckpt.*.tmp")) == []


def check_checkpoint_filename_cannot_escape_cache() -> None:
    """Malformed worker payloads cannot use checkpoint filenames for path traversal."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for filename in ("../x.ckpt", "sub/x.ckpt", "sub\\x.ckpt", "/x.ckpt", "x\n.ckpt", "x.bin", ""):
            with _expect(ValueError, "basename"):
                inference._download_checkpoint("https://example.com/x.ckpt", "", filename, tmp_path)


CHECKS = [
    check_hf_token_only_sent_to_real_hf_hosts,
    check_redirect_aborts_downgrade_and_strips_offhost_auth,
    check_download_lock_is_mutually_exclusive_and_auto_releases,
    check_redact_url_strips_credentials_and_query,
    check_checkpoint_url_path_secret_is_redacted_from_log_and_error,
    check_download_does_not_follow_predictable_temp_symlink,
    check_download_does_not_accept_destination_symlink,
    check_download_cleans_temp_file_on_interrupt,
    check_checkpoint_filename_cannot_escape_cache,
]


def _run_check(check: Any) -> str | None:
    """Run one check; print PASS/FAIL and return the check's name on failure, else None."""
    try:
        check()
    except Exception as error:
        print(f"FAIL {check.__name__}: {type(error).__name__}: {error}")
        traceback.print_exc()
        return str(check.__name__)
    print(f"PASS {check.__name__}")
    return None


def main() -> int:
    """Run every check, report per-check status, and return a non-zero code on any failure."""
    failures = [name for name in (_run_check(check) for check in CHECKS) if name is not None]
    if failures:
        print(f"FAILED {len(failures)}/{len(CHECKS)}: {failures}")
        return 1
    print(f"ALL CHECKS PASSED ({len(CHECKS)}/{len(CHECKS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
