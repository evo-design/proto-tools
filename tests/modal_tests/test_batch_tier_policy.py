"""The cost policy for batch-tier services: wall, retries, and the caller-facing warning.

A batch-tier call can occupy a GPU container for a day, so the three parts of the policy have to
agree about which services they cover and what they cost. Each is cheap to get wrong in a way that
only shows up on someone's Modal bill.
"""

from __future__ import annotations

import warnings

import pytest

from proto_tools.modal.app import NO_RETRIES, SERVICE_RETRIES, retries_for_service
from proto_tools.modal.client import LongRunningToolWarning, _warn_once_if_long_running
from proto_tools.modal.manifest import (
    BATCH_TIER,
    MODAL_MAX_TIMEOUT_SECONDS,
    SERVICE_MODAL_TIMEOUTS,
    SERVICE_TIERS,
    TIER_SECONDS,
    runs_for_hours,
)

BATCH_SERVICES = sorted(s for s, tier in SERVICE_TIERS.items() if tier == BATCH_TIER)


def test_batch_services_exist() -> None:
    """Guard the rest of this module: every assertion below is vacuous without a batch service."""
    assert BATCH_SERVICES


@pytest.mark.parametrize("service", BATCH_SERVICES)
def test_batch_services_get_no_retries(service: str) -> None:
    """Modal restarts the wall per retry, so retrying a day-long call bills the day again."""
    assert retries_for_service(service).max_retries == 0


def test_non_batch_services_keep_the_shared_retry_policy() -> None:
    """The helper must be a no-op for everything else, or adopting it changes unrelated services."""
    others = [s for s in SERVICE_TIERS if SERVICE_TIERS[s] != BATCH_TIER]
    assert others, "no non-batch services to check"
    assert all(retries_for_service(s) is SERVICE_RETRIES for s in others)
    assert SERVICE_RETRIES.max_retries > 0
    assert NO_RETRIES.max_retries == 0


def test_unknown_service_is_not_treated_as_batch() -> None:
    """An unlisted service has no tier; calling it expensive would warn about trivial work."""
    assert not runs_for_hours("NoSuchService")
    assert retries_for_service("NoSuchService") is SERVICE_RETRIES


def test_no_tier_exceeds_modal_ceiling() -> None:
    """Modal rejects a timeout above 24 hours at deploy time, so a tier above it cannot ship."""
    assert TIER_SECONDS[BATCH_TIER] == MODAL_MAX_TIMEOUT_SECONDS
    assert all(seconds <= MODAL_MAX_TIMEOUT_SECONDS for seconds in TIER_SECONDS.values())


def test_timeout_scale_cannot_push_a_wall_past_the_ceiling() -> None:
    """``PROTO_MODAL_TIMEOUT_SCALE`` lifts short tiers; on batch it must clamp, not fail the deploy."""
    assert all(seconds <= MODAL_MAX_TIMEOUT_SECONDS for seconds in SERVICE_MODAL_TIMEOUTS.values())


@pytest.mark.parametrize("service", BATCH_SERVICES)
def test_dispatch_warns_before_a_long_run(service: str) -> None:
    """The caller learns a call may run for hours before it starts, not by watching it not return."""
    _warn_once_if_long_running.cache_clear()
    with pytest.warns(LongRunningToolWarning, match="long-running pipeline"):
        _warn_once_if_long_running(f"{service}-tool", service)


def test_dispatch_does_not_warn_for_ordinary_tools() -> None:
    """A warning on every tool is a warning nobody reads."""
    _warn_once_if_long_running.cache_clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongRunningToolWarning)
        _warn_once_if_long_running("esm3-embedding", "ESM3Service")


def test_long_running_warning_is_said_once_per_tool() -> None:
    """A sweep dispatches the same tool repeatedly; the caller needs telling once."""
    _warn_once_if_long_running.cache_clear()
    service = BATCH_SERVICES[0]
    with pytest.warns(LongRunningToolWarning):
        _warn_once_if_long_running("repeat-tool", service)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongRunningToolWarning)
        _warn_once_if_long_running("repeat-tool", service)
