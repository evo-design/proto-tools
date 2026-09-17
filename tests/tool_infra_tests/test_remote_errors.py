"""Tests for conservative remote-error retry classification."""

import errno
from urllib.error import HTTPError, URLError

import pytest

from proto_tools.utils.remote_errors import is_retryable_remote_error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (URLError(TimeoutError()), True),
        (OSError(errno.ENETUNREACH, "unreachable"), False),
        (HTTPError("url", 429, "busy", {}, None), True),
        (HTTPError("url", 403, "denied", {}, None), False),
        (ValueError("Cannot accept request, error code: -99"), False),
    ],
)
def test_transport_retry_policy(error: Exception, expected: bool) -> None:
    """Only known transient transport failures are retryable."""
    assert is_retryable_remote_error(error) is expected
