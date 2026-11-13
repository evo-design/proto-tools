"""tests/tool_infra_tests/test_requires_credential.py.

``@pytest.mark.requires_credential(...)`` skips a test unless every named credential is set, and
only accepts names a ``ConfigField(credential=...)`` declares.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import _declared_credentials, _skip_without_credentials


def _item(*names: str) -> SimpleNamespace:
    marker = SimpleNamespace(args=names)
    added: list = []
    return SimpleNamespace(
        nodeid="tests/x.py::t",
        get_closest_marker=lambda name: marker if name == "requires_credential" else None,
        add_marker=added.append,
        added=added,
    )


def test_the_declared_credentials_come_from_config_fields() -> None:
    assert {"GI_API_KEY", "INTERPROSCAN_EMAIL", "PAPERCLIP_API_KEY", "SYNBIOHUB_PASSWORD"} <= _declared_credentials()


def test_a_missing_credential_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GI_API_KEY", raising=False)
    item = _item("GI_API_KEY")
    _skip_without_credentials([item])
    assert len(item.added) == 1 and "GI_API_KEY" in item.added[0].kwargs["reason"]


def test_a_blank_credential_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNBIOHUB_EMAIL", "someone@example.org")
    monkeypatch.setenv("SYNBIOHUB_PASSWORD", "  ")
    item = _item("SYNBIOHUB_EMAIL", "SYNBIOHUB_PASSWORD")
    _skip_without_credentials([item])
    assert "SYNBIOHUB_PASSWORD" in item.added[0].kwargs["reason"]
    assert "SYNBIOHUB_EMAIL" not in item.added[0].kwargs["reason"]


def test_every_credential_set_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GI_API_KEY", "gi_test")
    item = _item("GI_API_KEY")
    _skip_without_credentials([item])
    assert item.added == []


@pytest.mark.parametrize("names", [("GI_APIKEY",), ()], ids=["typo", "none"])
def test_an_undeclared_name_is_a_usage_error(names: tuple[str, ...]) -> None:
    with pytest.raises(pytest.UsageError, match="ConfigField"):
        _skip_without_credentials([_item(*names)])
