"""tests/tool_infra_tests/test_checkpoint_recovery.py.

Tests for corrupted-checkpoint retry in the standalone helpers.
"""

import pytest

from proto_tools.utils.standalone_helpers_source.standalone_helpers.recovery import (
    is_checkpoint_corruption_error,
    retry_on_checkpoint_corruption,
)

_CORRUPTION = "PytorchStreamReader failed reading zip archive: failed finding central directory"


@pytest.mark.parametrize("message", [_CORRUPTION, "BadZipFile: File is not a zip file"])
def test_classifier_matches_corrupted_archives(message: str) -> None:
    assert is_checkpoint_corruption_error(RuntimeError(message))


@pytest.mark.parametrize("exc", [RuntimeError("CUDA out of memory"), OSError("Permission denied")])
def test_classifier_rejects_unrelated_errors(exc: Exception) -> None:
    assert not is_checkpoint_corruption_error(exc)


def test_retry_recovers_then_succeeds() -> None:
    corrupt = True

    def load() -> None:
        if corrupt:
            raise RuntimeError(_CORRUPTION)

    def recover() -> None:
        nonlocal corrupt
        corrupt = False

    retry_on_checkpoint_corruption(load, tool_name="test-model", max_attempts=2, recover=recover)
    assert not corrupt


def test_retry_propagates_unrelated_failure_without_recovering() -> None:
    recover_calls = 0

    def load() -> None:
        raise RuntimeError("CUDA out of memory")

    def recover() -> None:
        nonlocal recover_calls
        recover_calls += 1

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        retry_on_checkpoint_corruption(load, tool_name="test-model", max_attempts=3, recover=recover)
    assert recover_calls == 0


def test_retry_is_bounded_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("time.sleep", delays.append)
    attempts = 0

    def load() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("BadZipFile")

    with pytest.raises(RuntimeError, match="BadZipFile"):
        retry_on_checkpoint_corruption(load, tool_name="test-model", max_attempts=3, backoff_s=2.0)
    assert attempts == 3
    assert delays == [2.0, 4.0]
