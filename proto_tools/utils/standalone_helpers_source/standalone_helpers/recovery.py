"""Bounded, cache-confined recovery for downloaded model checkpoints."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from .proto_logging import get_logger

logger = get_logger(__name__)

_CORRUPTION_MARKERS = (
    "PytorchStreamReader",
    "central directory",
    "zip archive",
    "BadZipFile",
)
_T = TypeVar("_T")


@dataclass(frozen=True)
class RecoveryAction:
    """One bounded mutation performed while recovering a model asset."""

    action: str
    path: str
    reason: str
    attempt: int

    def as_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable representation of this action."""
        return asdict(self)


def is_checkpoint_corruption_error(exc: BaseException) -> bool:
    """Return whether an exception identifies a malformed checkpoint archive."""
    message = str(exc)
    return any(marker in message for marker in _CORRUPTION_MARKERS)


def remove_checkpoint_within(
    checkpoint: str | Path,
    cache_root: str | Path,
    *,
    reason: str,
    attempt: int,
) -> RecoveryAction:
    """Remove one checkpoint only when it resolves inside its declared cache."""
    resolved_checkpoint = Path(checkpoint).resolve()
    resolved_root = Path(cache_root).resolve()
    if not resolved_checkpoint.is_relative_to(resolved_root):
        raise ValueError(f"refusing to remove checkpoint outside cache {resolved_root}: {resolved_checkpoint}")
    resolved_checkpoint.unlink(missing_ok=True)
    return RecoveryAction(
        action="removed_corrupt_checkpoint",
        path=str(resolved_checkpoint),
        reason=reason,
        attempt=attempt,
    )


def retry_checkpoint_operation(
    operation: Callable[[], _T],
    recover: Callable[[BaseException, int], Iterable[RecoveryAction]],
    *,
    tool_name: str,
    max_attempts: int,
    backoff_base_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[_T, list[RecoveryAction]]:
    """Retry an operation only after classified checkpoint corruption.

    ``recover`` owns tool-specific checkpoint discovery. This helper owns the
    shared corruption classifier, attempt bound, backoff, logging, and action
    collection. Unrelated failures propagate immediately.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff_base_seconds < 0:
        raise ValueError("backoff_base_seconds must be non-negative")

    actions: list[RecoveryAction] = []
    for attempt in range(1, max_attempts + 1):
        try:
            return operation(), actions
        except Exception as exc:  # noqa: PERF203 -- this is one bounded attempt
            if attempt >= max_attempts or not is_checkpoint_corruption_error(exc):
                raise
            recovered = list(recover(exc, attempt))
            if not recovered:
                raise RuntimeError(
                    f"{tool_name}: checkpoint corruption was detected but no recoverable asset was identified"
                ) from exc
            actions.extend(recovered)
            delay = backoff_base_seconds * (2 ** (attempt - 1))
            logger.warning(
                "%s checkpoint load failed (attempt %d/%d); performed %d recovery action(s)%s",
                tool_name,
                attempt,
                max_attempts,
                len(recovered),
                f" and retrying in {delay:.1f}s" if delay else " and retrying",
            )
            if delay:
                sleep(delay)

    raise AssertionError("checkpoint retry loop exhausted without returning or raising")
