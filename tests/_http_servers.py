"""Localhost HTTP servers that misbehave the way slow public APIs do, for deadline tests."""

from __future__ import annotations

import gzip
import http.server
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

TRICKLE_BODY = b"x" * 400


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests_seen = 0
    url = ""

    def log_message(self, *args: object) -> None:
        pass

    def _count(self) -> None:
        type(self).requests_seen += 1


class TricklingHandler(_QuietHandler):
    """Send a small body one byte every 50ms, so no single socket read ever times out."""

    def do_GET(self) -> None:
        self._count()
        self.send_response(200)
        self.send_header("Content-Length", str(len(TRICKLE_BODY)))
        self.end_headers()
        for byte in TRICKLE_BODY:
            self.wfile.write(bytes([byte]))
            self.wfile.flush()
            time.sleep(0.05)


class ChunkedTricklingHandler(_QuietHandler):
    """Trickle a chunked-encoding body, one single-byte chunk every 50ms."""

    def do_GET(self) -> None:
        self._count()
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for byte in TRICKLE_BODY:
            self.wfile.write(b"1\r\n" + bytes([byte]) + b"\r\n")
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"0\r\n\r\n")


class StallingHandler(_QuietHandler):
    """Promise a large body, send a few bytes, then go silent."""

    def do_GET(self) -> None:
        self._count()
        self.send_response(200)
        self.send_header("Content-Length", str(1 << 20))
        self.end_headers()
        self.wfile.write(b"x" * 16)
        self.wfile.flush()
        time.sleep(30)


class RetryAfterHandler(_QuietHandler):
    """Answer every request with ``503`` and ``Retry-After: 20``."""

    retry_after = "20"

    def do_GET(self) -> None:
        self._count()
        body = b"busy"
        self.send_response(503)
        self.send_header("Retry-After", self.retry_after)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RetryAfterZeroHandler(RetryAfterHandler):
    """Answer every request with ``503`` and ``Retry-After: 0``, which urllib3 treats as "use the backoff"."""

    retry_after = "0"


class GzipRedirectHandler(_QuietHandler):
    """Redirect ``/start`` to ``/data``, which returns gzip-encoded JSON."""

    def do_GET(self) -> None:
        self._count()
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", "/data")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = gzip.compress(b'{"ok": true}')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def serve(handler: type[_QuietHandler]) -> Iterator[type[_QuietHandler]]:
    """Run ``handler`` on a free localhost port and yield a handler class carrying its ``url``.

    Each call serves a fresh subclass, so ``requests_seen`` counts only this server's requests.
    """
    counted = type(handler.__name__, (handler,), {"requests_seen": 0})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), counted)
    server.daemon_threads = True
    counted.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield counted
    finally:
        server.shutdown()
        server.server_close()
