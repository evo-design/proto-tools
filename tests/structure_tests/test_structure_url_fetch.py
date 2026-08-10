"""A structure URL cannot be pointed at the fetching machine's own network.

``Structure(structure="https://...")`` fetches transparently, which is convenient when the person
supplying the URL owns the machine doing the fetch. A server accepting structures from callers is
the case where they do not, and there the field is a request-forgery primitive: cloud instance
metadata at ``169.254.169.254``, or a service reachable only from inside a private network.

Offline throughout -- no request leaves the machine.
"""

from __future__ import annotations

import pytest

from proto_tools.entities.structures.structure import _fetch_structure_url

#: Addresses a fetch must refuse, one per reason it is not routable on the public internet.
PRIVATE_URLS = [
    "http://169.254.169.254/latest/meta-data/",  # cloud instance metadata
    "http://127.0.0.1:8000/health",  # loopback
    "http://10.0.0.5/internal",  # private range
    "http://192.168.1.1/",  # private range
    "http://[::1]:8000/",  # loopback, IPv6
]


@pytest.fixture
def no_requests(monkeypatch):
    """Fail loudly if anything actually issues a request."""
    import requests

    def _forbidden(*args, **kwargs):
        raise AssertionError(f"a request was issued: {args!r}")

    monkeypatch.setattr(requests, "get", _forbidden)


@pytest.mark.parametrize("url", PRIVATE_URLS)
def test_a_private_address_is_refused(url, no_requests):
    """Refused before the request, so nothing reaches the address even once."""
    with pytest.raises(ValueError, match="non-public address"):
        _fetch_structure_url(url)


def test_a_public_hostname_resolving_somewhere_private_is_refused(monkeypatch, no_requests):
    """Pattern-matching the URL would miss this; the check resolves the host instead."""
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("10.1.2.3", 0))])
    with pytest.raises(ValueError, match="non-public address"):
        _fetch_structure_url("https://structures.example.com/1abc.cif")


def test_an_unresolvable_host_is_refused(monkeypatch, no_requests):
    """Unresolvable is refused rather than attempted, so failure is one clear error."""
    import socket

    def _fail(host, port):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    with pytest.raises(ValueError, match="could not resolve"):
        _fetch_structure_url("https://nonexistent.invalid/1abc.cif")


def test_a_redirect_is_not_followed(monkeypatch):
    """The address check runs before the request, so a redirect would escape it.

    ``raise_for_status`` does not catch this on its own: a 3xx is not an error status.
    """
    import requests

    captured: dict = {}

    class _Redirect:
        status_code = 302
        is_redirect = True
        text = ""

        def __init__(self) -> None:
            self.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}

        def raise_for_status(self):
            return None

    def _fake_get(url, **kwargs):
        captured.update(kwargs)
        return _Redirect()

    monkeypatch.setattr(requests, "get", _fake_get)

    with pytest.raises(ValueError, match="redirected"):
        _fetch_structure_url("https://files.rcsb.org/download/1TIM.cif")
    assert captured["allow_redirects"] is False, "requests follows redirects unless told not to"


def test_a_public_url_still_fetches(monkeypatch):
    """The guard must not break the ordinary case it exists to protect."""
    import requests

    class _Ok:
        status_code = 200
        is_redirect = False
        text = "data_1TIM\n"

        def __init__(self) -> None:
            self.headers: dict = {}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _Ok())

    assert _fetch_structure_url("https://files.rcsb.org/download/1TIM.cif") == "data_1TIM\n"
