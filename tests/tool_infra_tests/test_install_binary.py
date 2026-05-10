"""Tests for `proto_tools.utils.install_binary` retry and integrity logic."""

import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from proto_tools.utils import install_binary


def _serve(payload: bytes, advertised: int | None = None) -> tuple[HTTPServer, str]:
    """Start a localhost server that advertises ``advertised`` bytes but writes ``payload``."""
    length = advertised if advertised is not None else len(payload)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args, **_kwargs):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/file"


@pytest.fixture
def stub_platform(tmp_path, monkeypatch):
    """Stub config discovery, extract, sleep, and platform so the retry loop can be driven without I/O."""
    monkeypatch.setattr(install_binary.time, "sleep", lambda _s: None)
    monkeypatch.setattr(install_binary, "_find_tool_config", lambda _t: tmp_path / "binary_config.py")
    fake_config = type(
        "M", (), {"URLS": {("Linux", "x86_64"): "http://example/file"}, "extract": staticmethod(lambda *_a: None)}
    )
    monkeypatch.setattr(install_binary, "_load_tool_config", lambda _p: fake_config)
    monkeypatch.setattr(install_binary.platform, "system", lambda: "Linux")
    monkeypatch.setattr(install_binary.platform, "machine", lambda: "x86_64")
    return monkeypatch


def test_download_raises_on_truncation(tmp_path):
    """The CI failure shape: server advertises Content-Length=N but writes M<N → OSError so retry loop kicks in."""
    server, url = _serve(b"x" * 100, advertised=10_000)
    try:
        with pytest.raises(OSError, match="truncated"):
            install_binary._download_with_progress(url, tmp_path / "out.bin")
    finally:
        server.shutdown()


def test_download_writes_full_payload(tmp_path):
    """Happy path: full payload streams to disk and integrity check passes."""
    payload = b"x" * 4096
    server, url = _serve(payload)
    try:
        dest = tmp_path / "out.bin"
        install_binary._download_with_progress(url, dest)
        assert dest.read_bytes() == payload
    finally:
        server.shutdown()


def test_install_binary_retries_until_success(stub_platform):
    """Transient failures are retried; install_binary returns once a download succeeds."""
    calls = 0

    def flaky(_url, dest: Path):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("simulated truncation")
        dest.write_bytes(b"ok")

    stub_platform.setattr(install_binary, "_download_with_progress", flaky)
    install_binary.install_binary("dummy")
    assert calls == 3


def test_install_binary_exhausts_retries_with_exponential_backoff(stub_platform):
    """All attempts fail → RuntimeError; sleeps follow exp-backoff capped at _MAX_RETRY_DELAY_SECONDS."""
    sleeps: list[float] = []
    stub_platform.setattr(install_binary.time, "sleep", sleeps.append)
    attempts = 0

    def always_fail(_url, _dest):
        nonlocal attempts
        attempts += 1
        raise urllib.error.URLError("network down")

    stub_platform.setattr(install_binary, "_download_with_progress", always_fail)

    with pytest.raises(RuntimeError, match="Failed to download dummy"):
        install_binary.install_binary("dummy")

    assert attempts == install_binary._MAX_DOWNLOAD_RETRIES
    expected = [
        min(
            install_binary._INITIAL_RETRY_DELAY_SECONDS * install_binary._BACKOFF_MULTIPLIER**i,
            install_binary._MAX_RETRY_DELAY_SECONDS,
        )
        for i in range(install_binary._MAX_DOWNLOAD_RETRIES - 1)
    ]
    assert sleeps == expected
