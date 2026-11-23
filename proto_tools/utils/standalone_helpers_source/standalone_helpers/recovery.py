"""Retry model loads that fail on a corrupted (typically truncated) downloaded checkpoint."""

import time
from collections.abc import Callable

from .proto_logging import get_logger

logger = get_logger(__name__)

# Substrings torch and zipfile raise when a checkpoint archive is incomplete.
_CORRUPTION_MARKERS = ("PytorchStreamReader", "central directory", "zip archive", "BadZipFile")


def is_checkpoint_corruption_error(exc: BaseException) -> bool:
    """Whether ``exc`` signals a malformed checkpoint archive."""
    message = str(exc)
    return any(marker in message for marker in _CORRUPTION_MARKERS)


def retry_on_checkpoint_corruption(
    load: Callable[[], None],
    *,
    tool_name: str,
    max_attempts: int,
    recover: Callable[[], None] | None = None,
    backoff_s: float = 0.0,
) -> None:
    """Run ``load``, retrying only while it fails on a corrupted checkpoint.

    ``recover`` runs between attempts to remove the corrupt file so the next load
    downloads it again. Unrelated errors and errors on the final attempt are raised
    unchanged.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            load()
            return
        except Exception as exc:  # noqa: PERF203 -- one bounded attempt per iteration
            if attempt == max_attempts or not is_checkpoint_corruption_error(exc):
                raise
            delay = backoff_s * 2 ** (attempt - 1)
            logger.warning(
                "%s: corrupted checkpoint (attempt %d/%d): %s; retrying in %.1fs",
                tool_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            if recover is not None:
                recover()
            time.sleep(delay)
