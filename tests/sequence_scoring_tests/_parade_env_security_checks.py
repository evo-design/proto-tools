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

The download itself goes through ``standalone_helpers.download_weights``, whose lock, temp-file,
symlink, redirect, and redaction behavior is tested in the base env by
``tests/tool_infra_tests/test_weights_helpers.py``. These checks cover what PARADE adds on top:
filename validation and routing its checkpoint download through that helper.
"""

from __future__ import annotations

import contextlib
import hashlib
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
            _patched(urllib.request, "build_opener", lambda *a, **k: _Opener()),
            _env_var("HF_TOKEN", "secret-token"),
        ):
            assert auth_for("https://evil-huggingface.co/x.ckpt", "evil.ckpt") is None
            assert auth_for("https://huggingface.co.attacker.com/x.ckpt", "evil2.ckpt") is None
            assert auth_for("https://huggingface.co/x.ckpt", "hf.ckpt") == "Bearer secret-token"
            assert auth_for("https://cdn-lfs.huggingface.co/x.ckpt", "hf2.ckpt") == "Bearer secret-token"
            with _expect(ValueError, "HTTPS"):
                inference._download_checkpoint("http://huggingface.co/x.ckpt", "", "plain.ckpt", tmp_path)


def check_checkpoint_md5_is_verified() -> None:
    """The pinned MD5 is passed through: a matching payload lands, a mismatch never does."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        payload_md5 = hashlib.md5(b"checkpoint", usedforsecurity=False).hexdigest()
        with _patched(urllib.request, "build_opener", _opener_returning((b"checkpoint",))):
            ok = inference._download_checkpoint("https://example.com/ok.ckpt", payload_md5, "ok.ckpt", tmp_path)
            assert ok.read_bytes() == b"checkpoint"
            with _expect(RuntimeError, "checksum mismatch"):
                inference._download_checkpoint("https://example.com/bad.ckpt", "0" * 32, "bad.ckpt", tmp_path)
        assert not (tmp_path / "bad.ckpt").exists()


def check_checkpoint_filename_cannot_escape_cache() -> None:
    """Malformed worker payloads cannot use checkpoint filenames for path traversal."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for filename in ("../x.ckpt", "sub/x.ckpt", "sub\\x.ckpt", "/x.ckpt", "x\n.ckpt", "x.bin", ""):
            with _expect(ValueError, "basename"):
                inference._download_checkpoint("https://example.com/x.ckpt", "", filename, tmp_path)


CHECKS = [
    check_hf_token_only_sent_to_real_hf_hosts,
    check_checkpoint_md5_is_verified,
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
