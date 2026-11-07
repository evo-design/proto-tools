"""tests/tool_infra_tests/test_outage.py.

Declared upstream outages: a call that reaches one is refused before dispatch with OutageError,
raised the way a validation error is, and every agent-facing surface shows the outage first.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pytest
from pydantic import Field, ValidationError

from proto_tools.cli import _summary_line
from proto_tools.mcp import tools as impl
from proto_tools.tools.tool_registry import ToolRegistry, tool
from proto_tools.utils import BaseConfig, ConfigField, Outage, OutageError
from proto_tools.utils.tool_io import BaseToolInput
from tests.tool_infra_tests.test_export_functionality import MockToolOutputBase

_TRACKING = "https://example.org/status/42"


class _OutageInput(BaseToolInput):
    """Inputs for the outage stand-in tool."""

    query: str = Field(default="q", description="Query string")


class _OutageConfig(BaseConfig):
    """Config with one mode that an outage can target."""

    device: str = ConfigField(default="cpu", title="Device", description="Device to run on")
    use_upstream: bool = ConfigField(default=False, title="Use Upstream", description="Reach the affected upstream")


class _OutageOutput(MockToolOutputBase):
    """Outputs for the outage stand-in tool."""

    echoed: str = Field(default="", description="Echoed query")


def _register(key: str, outage: Outage, calls: list[str]) -> None:
    """Register a trivially local tool declaring ``outage`` that records each real execution."""

    @tool(
        key=key,
        label="Outage Probe",
        category="testing",
        input_class=_OutageInput,
        config_class=_OutageConfig,
        output_class=_OutageOutput,
        description="Stand-in tool for outage tests",
        example_input=_OutageInput,
        outages=[outage],
    )
    def _run(inputs: _OutageInput, config: _OutageConfig, instance: Any = None) -> _OutageOutput:
        calls.append(inputs.query)
        return _OutageOutput(echoed=inputs.query)


@pytest.fixture
def offline_tool():
    """A tool whose whole upstream is down; yields its key and the list of real executions."""
    key, calls = "outage-probe-offline", []
    _register(key, Outage(reason="Upstream is down", since=date(2026, 1, 1), tracking_url=_TRACKING), calls)
    yield key, calls
    ToolRegistry._registry.pop(key, None)


@pytest.fixture
def degraded_tool():
    """A tool whose ``use_upstream`` mode is down; yields its key and the list of real executions."""
    key, calls = "outage-probe-degraded", []
    outage = Outage(
        reason="Upstream mode is down",
        since=date(2026, 1, 1),
        tracking_url=_TRACKING,
        affects="use_upstream",
        when=lambda inputs, config: config.use_upstream,
    )
    _register(key, outage, calls)
    yield key, calls
    ToolRegistry._registry.pop(key, None)


@pytest.mark.parametrize("capture", ["0", "1"])
def test_whole_tool_outage_is_raised_before_the_tool_runs(offline_tool, monkeypatch, capture):
    """Raised like a validation error, never captured into output.errors, and the tool never runs."""
    monkeypatch.setenv("PROTO_CAPTURE_ERRORS", capture)
    key, calls = offline_tool
    with pytest.raises(OutageError, match=r"Upstream is down \(since 2026-01-01; affects every call\)") as exc:
        ToolRegistry.get(key).function(_OutageInput(), _OutageConfig())
    assert exc.value.tool_key == key
    assert isinstance(exc.value, ValueError)
    assert calls == []


def test_scoped_outage_refuses_only_the_affected_mode(degraded_tool):
    """A call that avoids the affected mode runs normally; one that reaches it is refused."""
    key, calls = degraded_tool
    run = ToolRegistry.get(key).function

    assert run(_OutageInput(query="ok"), _OutageConfig(use_upstream=False)).echoed == "ok"
    with pytest.raises(OutageError, match="affects use_upstream"):
        run(_OutageInput(query="blocked"), _OutageConfig(use_upstream=True))
    assert calls == ["ok"]


def test_ignore_outages_env_lets_the_call_through_with_a_warning(degraded_tool, monkeypatch, caplog):
    """PROTO_IGNORE_OUTAGES=1 lets a caller check whether the upstream has recovered."""
    monkeypatch.setenv("PROTO_IGNORE_OUTAGES", "1")
    key, calls = degraded_tool
    with caplog.at_level(logging.WARNING, logger="proto_tools.utils.outage"):
        out = ToolRegistry.get(key).function(_OutageInput(query="probe"), _OutageConfig(use_upstream=True))
    assert out.echoed == "probe"
    assert calls == ["probe"]
    assert "ignoring declared outage" in caplog.text


def test_outage_is_refused_before_remote_dispatch(offline_tool, monkeypatch):
    """A remote device never receives a call that matches an outage."""
    import proto_tools.proto as proto_module

    def _must_not_dispatch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dispatched to a remote device despite the outage")

    monkeypatch.setattr(proto_module, "dispatch_to_proto", _must_not_dispatch, raising=False)
    key, _ = offline_tool
    with pytest.raises(OutageError):
        ToolRegistry.get(key).function(_OutageInput(), _OutageConfig(device="proto"))


@pytest.mark.parametrize(
    "kwargs",
    [{"affects": "mode"}, {"when": lambda inputs, config: True}],
    ids=["affects-without-when", "when-without-affects"],
)
def test_scoped_outage_needs_both_label_and_predicate(kwargs):
    """A scope without a test, or a test without a label for catalogues, is rejected."""
    with pytest.raises(ValidationError, match=r"affects and Outage\.when must be set together"):
        Outage(reason="r", since=date(2026, 1, 1), tracking_url=_TRACKING, **kwargs)


@pytest.mark.parametrize("device", ["local", "proto"])
def test_mcp_run_tool_refuses_with_the_outage(degraded_tool, monkeypatch, device):
    """Both MCP branches refuse: the remote one never reaches the wrapper's own check."""
    import proto_tools.proto as proto_module

    monkeypatch.setattr(
        proto_module, "dispatch_to_proto", lambda *a, **k: pytest.fail("dispatched despite outage"), raising=False
    )
    key, calls = degraded_tool
    answer = impl.run_tool(key, inputs={"query": "q"}, config={"use_upstream": True}, device=device)
    assert answer["ok"] is False
    assert answer["outage"]["affects"] == "use_upstream"
    assert "PROTO_IGNORE_OUTAGES" in answer["error"]
    assert calls == []


def test_mcp_surfaces_show_the_outage_before_a_run(degraded_tool):
    """Schema (step 1 of every run), info, and catalogue entries all carry the outage."""
    key, _ = degraded_tool
    expected = [
        {"reason": "Upstream mode is down", "since": "2026-01-01", "tracking_url": _TRACKING, "affects": "use_upstream"}
    ]
    assert impl.get_tool_schema(key)["outages"] == expected
    assert impl.get_tool_info(key)["outages"] == expected
    (entry,) = [e for e in impl.list_tools(deployed_only=False, device="local") if e["tool_key"] == key]
    assert entry["outages"] == expected
    assert "Outage: Upstream mode is down" in entry["note"]


@pytest.mark.parametrize(
    ("scopes", "marker"),
    [(None, ""), ([None], " (OFFLINE)"), (["include_msa"], " (DEGRADED: include_msa)"), (["a", None], " (OFFLINE)")],
)
def test_cli_summary_marks_outages(scopes, marker):
    """A whole-tool outage outranks scoped ones; scoped ones name the affected modes."""
    line = _summary_line("k", "cat", False, "desc", scopes)
    assert line == f"{'k':40s}  [cat]{marker}  desc"
