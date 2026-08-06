"""A stale deployment is reported to every caller it affects, not to the first one only.

Drift means the deployed tool was built from a different proto-tools than the one calling it. The
dispatch path already detects this and raises a ``DeploymentDriftWarning``, which is the right
shape for a session on a laptop and reaches nobody anywhere else: an MCP caller does not see the
server's ``warnings``, and the warning is emitted once per process, so on a shared server the first
caller to touch a stale deployment silences it for everyone after them.

So ``run_tool`` returns drift as part of the answer. These pin that it is returned, that it is
returned per call rather than once, and that it never costs a successful run.
"""

from __future__ import annotations

import pytest

from proto_tools.mcp import tools as impl

TOOL_KEY = "esm2-embedding"


@pytest.fixture
def stale(monkeypatch):
    """Report every tool as drifted, and count how many times the check is made."""
    calls: list[tuple[str, str | None, object]] = []

    def _warnings(tool_key, service_class, environment=None, client=None):
        calls.append((tool_key, environment, client))
        return [f"{tool_key}: the deployed tool's code differs from your local proto-tools."]

    monkeypatch.setattr("proto_tools.modal.fingerprint.drift_warnings", _warnings)
    return calls


def test_every_caller_is_told_not_just_the_first(stale):
    """``warnings.warn`` fires once per process. On a server that is one caller out of many."""
    first = impl.drift_for(TOOL_KEY, "modal", environment="proto-env", client=None)
    second = impl.drift_for(TOOL_KEY, "modal", environment="proto-env", client=None)
    assert first and "differs from your local proto-tools" in first[0]
    assert first == second, "the second caller was told less than the first"
    assert len(stale) == 2


def test_a_caller_who_names_no_environment_is_checked_where_they_actually_ran(stale, monkeypatch):
    """The default case, and the one an unresolved check silently misses.

    A dispatch resolves ``None`` to ``proto-env`` before it looks anything up. Passing ``None``
    straight through instead reads the manifest from the caller's *active* environment, which
    holds a different deployment or none at all -- so the feature reports nothing in exactly the
    configuration a caller who sends no environment header gets.
    """
    monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)

    impl.drift_for(TOOL_KEY, "modal", environment=None)

    assert stale == [(TOOL_KEY, "proto-env", None)], "the drift check looked somewhere the call did not go"


def test_the_check_is_made_against_the_deployment_the_call_reached(stale):
    """Two callers can name different environments and different workspaces.

    Checking ambiently would compare against whichever deployment this process happens to resolve,
    which for a hosted server is nobody's.
    """
    client = object()
    impl.drift_for(TOOL_KEY, "modal", environment="someones-env", client=client)
    assert stale == [(TOOL_KEY, "someones-env", client)]


@pytest.mark.parametrize("ran_on", ["local", "proto"])
def test_a_call_that_never_reached_modal_reports_nothing(stale, ran_on):
    """There is no deployment to be stale against, and the volume read would be wasted."""
    assert impl.drift_for(TOOL_KEY, ran_on) == []
    assert stale == []


def test_an_unserved_tool_reports_nothing(stale):
    """A tool with no deployment has no service to look a manifest up by."""
    assert impl.drift_for("not-a-real-tool", "modal") == []
    assert stale == []


def test_a_broken_drift_check_never_costs_a_result(monkeypatch):
    """The run already succeeded and cost GPU time. A bookkeeping read must not discard it."""

    def _explode(*args, **kwargs):
        raise RuntimeError("volume unreachable")

    # Patched below drift_warnings, which swallows its own failures -- that swallowing is what is
    # being checked, since drift_for deliberately adds no second layer of its own.
    monkeypatch.setattr("proto_tools.modal.fingerprint.read_manifest", _explode)
    assert impl.drift_for(TOOL_KEY, "modal") == []


def test_run_tool_attaches_drift_to_a_successful_result(monkeypatch, stale, tmp_path):
    """Where a caller actually meets it: alongside the result, not instead of it."""
    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(TOOL_KEY)
    example = ToolRegistry.get_example_input(TOOL_KEY)
    assert example is not None

    monkeypatch.setattr(impl, "_dispatch", lambda *a, **k: (_stub_output(spec), "modal"))

    answer = impl.run_tool(TOOL_KEY, inputs=example.model_dump(mode="json"), output_dir=str(tmp_path))

    assert answer["ok"] is True, answer
    assert answer["warnings"], "a drifted deployment produced a result with no warning on it"


def test_run_tool_says_nothing_when_the_deployment_is_aligned(monkeypatch, tmp_path):
    """An aligned deployment is the normal case and must not add noise to every response."""
    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(TOOL_KEY)
    example = ToolRegistry.get_example_input(TOOL_KEY)
    assert example is not None

    monkeypatch.setattr("proto_tools.modal.fingerprint.drift_warnings", lambda *a, **k: [])
    monkeypatch.setattr(impl, "_dispatch", lambda *a, **k: (_stub_output(spec), "modal"))

    answer = impl.run_tool(TOOL_KEY, inputs=example.model_dump(mode="json"), output_dir=str(tmp_path))

    assert answer["ok"] is True, answer
    assert "warnings" not in answer


def _stub_output(spec):
    """The smallest valid output for a tool, so the test exercises run_tool rather than a model."""
    return spec.output_model(tool_id=spec.key, execution_time=0.0, success=True, results=[])
