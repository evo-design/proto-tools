"""Secrets every service receives, and the variable that adds to them."""

from __future__ import annotations

import pytest

from proto_tools.modal.app import EXTRA_SECRETS_ENV, HF_TOKEN_SECRET, service_secrets


@pytest.fixture(autouse=True)
def _clear_extra_secrets(monkeypatch: pytest.MonkeyPatch):
    """A developer with the variable exported would otherwise fail the default-only assertions."""
    monkeypatch.delenv(EXTRA_SECRETS_ENV, raising=False)


def test_the_default_is_the_huggingface_secret_alone() -> None:
    """Every deployment gets weight downloads without naming anything."""
    assert service_secrets() == [HF_TOKEN_SECRET]


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("one,two", id="comma"),
        pytest.param("one two", id="whitespace"),
        pytest.param(" one , two ", id="padded"),
    ],
)
def test_named_secrets_are_added_not_substituted(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Replacing the default would silently drop the token gated weight downloads need."""
    monkeypatch.setenv(EXTRA_SECRETS_ENV, raw)
    secrets = service_secrets()
    assert secrets[0] is HF_TOKEN_SECRET
    assert len(secrets) == 3


def test_an_empty_value_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset variable and one set to empty must behave the same, not add a nameless secret."""
    monkeypatch.setenv(EXTRA_SECRETS_ENV, "  ")
    assert service_secrets() == [HF_TOKEN_SECRET]


def test_every_app_carries_the_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attached to the app so a service needs no entry of its own; Modal adds them to each function."""
    from proto_tools.modal.app import get_app

    get_app.cache_clear()
    monkeypatch.setenv(EXTRA_SECRETS_ENV, "probe-secret")
    try:
        app = get_app("proto-tools-secret-probe")
        assert len(app._local_state.secrets_default) == 2, "an app must carry the default secrets"
    finally:
        get_app.cache_clear()
