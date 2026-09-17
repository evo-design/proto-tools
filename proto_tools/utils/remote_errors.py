"""Conservative retry decisions for remote tool requests.

Unknown errors, rejected requests, and authentication errors require diagnosis;
they must not silently become retry loops or successful empty responses.
"""

import errno
from urllib.error import HTTPError, URLError


def is_retryable_remote_error(error: Exception) -> bool:
    """Recognize transient transport failures without depending on log wording."""
    if isinstance(error, HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504}
    if isinstance(error, URLError):
        reason = error.reason
        return isinstance(reason, Exception) and is_retryable_remote_error(reason)
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    if isinstance(error, OSError):
        return error.errno in {
            errno.ETIMEDOUT,
            errno.ECONNRESET,
            errno.ECONNABORTED,
            errno.EPIPE,
        }
    return False
