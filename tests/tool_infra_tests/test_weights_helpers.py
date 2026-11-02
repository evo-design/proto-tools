"""Tests for the standalone weights download and env-export helpers."""

import hashlib
import io
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers import weights
from proto_tools.utils.standalone_helpers_source.standalone_helpers.weights import (
    _SecureRedirectHandler,
    download_weights,
    export_weights_dir,
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
