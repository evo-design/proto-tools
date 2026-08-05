"""What a smoke test does when a deployed tool reports an asset it could not find."""

from __future__ import annotations

from typing import Any

import modal
import pytest

from proto_tools.modal import smoke

_MISSING_ASSET_RESULT = {
    "success": False,
    "errors": ["MissingAssetError: spliceai reference genome not found"],
    "missing_asset": True,
    "missing_asset_toolkit": "spliceai",
    "missing_asset_kind": "reference",
}


@pytest.fixture
def fake_service(monkeypatch: pytest.MonkeyPatch):
    """Replace the deployed class with one returning whatever result a test names."""

    def install(result: dict[str, Any]) -> None:
        class _Method:
            @staticmethod
            def remote(**_kwargs: Any) -> dict[str, Any]:
                return result

        class _Service:
            score = _Method()

        monkeypatch.setattr(modal.Cls, "from_name", staticmethod(lambda *_a, **_k: _Service))

    return install


def test_a_missing_asset_fails_the_smoke_test(fake_service) -> None:
    """An asset the container could not find means the deployment cannot serve the tool.

    SpliceAI's genome auto-provisions and the service stages it at warmup, so it is absent only
    when that staging broke. Reporting a skip would let exactly that regression deploy green.
    """
    fake_service(_MISSING_ASSET_RESULT)
    passed, detail = smoke.run_tool("SpliceAIService", "spliceai-score", "score")
    assert passed is False
    assert "MissingAssetError" in detail


def test_a_working_tool_still_passes(fake_service) -> None:
    """The failure path must not swallow an ordinary success."""
    fake_service({"success": True, "scores": []})
    passed, detail = smoke.run_tool("SpliceAIService", "spliceai-score", "score")
    assert passed is True
    assert detail.endswith("s")
