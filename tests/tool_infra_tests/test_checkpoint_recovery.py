"""Tests for bounded, cache-confined checkpoint recovery."""

from pathlib import Path

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers.recovery import (
    RecoveryAction,
    remove_checkpoint_within,
    retry_checkpoint_operation,
)


def test_retry_recovers_once_and_reports_action(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"truncated")
    calls = 0

    def load() -> str:
        nonlocal calls
        calls += 1
        if checkpoint.exists():
            raise RuntimeError("PytorchStreamReader failed finding central directory")
        return "loaded"

    def recover(exc: BaseException, attempt: int) -> list[RecoveryAction]:
        return [
            remove_checkpoint_within(
                checkpoint,
                tmp_path,
                reason=str(exc),
                attempt=attempt,
            )
        ]

    result, actions = retry_checkpoint_operation(
        load,
        recover,
        tool_name="test-model",
        max_attempts=2,
    )

    assert result == "loaded"
    assert calls == 2
    assert [action.as_dict() for action in actions] == [
        {
            "action": "removed_corrupt_checkpoint",
            "path": str(checkpoint.resolve()),
            "reason": "PytorchStreamReader failed finding central directory",
            "attempt": 1,
        }
    ]


def test_retry_does_not_recover_unrelated_failure() -> None:
    recover_calls = 0

    def recover(_exc: BaseException, _attempt: int) -> list[RecoveryAction]:
        nonlocal recover_calls
        recover_calls += 1
        return []

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        retry_checkpoint_operation(
            lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
            recover,
            tool_name="test-model",
            max_attempts=3,
        )
    assert recover_calls == 0


def test_retry_is_bounded_and_applies_exponential_backoff(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    delays: list[float] = []
    attempts = 0

    def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("BadZipFile")

    def recover(exc: BaseException, attempt: int) -> list[RecoveryAction]:
        return [
            RecoveryAction(
                action="redownload_requested",
                path=str(checkpoint),
                reason=str(exc),
                attempt=attempt,
            )
        ]

    with pytest.raises(RuntimeError, match="BadZipFile"):
        retry_checkpoint_operation(
            fail,
            recover,
            tool_name="test-model",
            max_attempts=3,
            backoff_base_seconds=2.0,
            sleep=delays.append,
        )
    assert attempts == 3
    assert delays == [2.0, 4.0]


def test_remove_checkpoint_refuses_path_outside_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    foreign = tmp_path / "foreign.pt"
    foreign.write_bytes(b"keep")

    with pytest.raises(ValueError, match="outside cache"):
        remove_checkpoint_within(
            foreign,
            cache,
            reason="test",
            attempt=1,
        )
    assert foreign.read_bytes() == b"keep"
