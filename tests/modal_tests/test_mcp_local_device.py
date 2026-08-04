"""The local backend, and what a remote session does with a tool that cannot leave this machine."""

from __future__ import annotations

import pytest

from proto_tools.mcp.device import DeviceUnavailableError, is_remote, resolve_device
from proto_tools.tools import ToolRegistry

# local_only, not local_cpu, and overriding neither config hook — the combination that used to
# reach a container and come back asking for a deployment that will never exist.
_LOCAL_ONLY_TOOL = "pyhmmer-hmmscan"


def test_local_is_resolved_and_is_not_remote() -> None:
    """The server can be pointed at this machine, and the rest of the code can tell."""
    assert resolve_device("local") == "local"
    assert not is_remote("local")
    assert is_remote("modal") and is_remote("proto")


def test_an_unknown_device_names_all_three() -> None:
    """The error is the only place a caller learns what the choices are."""
    with pytest.raises(DeviceUnavailableError, match="'modal', 'proto', or 'local'"):
        resolve_device("laptop")


def test_the_default_is_unchanged() -> None:
    """Adding a backend must not move anyone off the one they already use."""
    assert resolve_device(None) == "modal"


def test_every_registered_tool_is_available_locally() -> None:
    """Nothing is provisioned in advance here, so the catalogue is the registry itself."""
    from proto_tools.mcp import tools as impl

    listed = {entry["tool"] for entry in impl.list_tools(deployed_only=True, device="local")}
    assert listed == {spec.key for spec in ToolRegistry.list_all()}
    assert _LOCAL_ONLY_TOOL in listed, "a tool with no deployment still runs on this machine"


def test_the_local_catalogue_is_larger_than_the_deployable_one() -> None:
    """The dispatch table is the wrong universe locally: it omits tools that run here fine."""
    from proto_tools.mcp import tools as impl

    local = impl.list_tools(deployed_only=True, device="local")
    deployable = impl.list_tools(deployed_only=False, device="modal")
    assert len(local) > len(deployable)


def test_workspace_info_reports_no_account_and_no_deploys() -> None:
    """There is nothing to authenticate against and nothing to deploy to."""
    from proto_tools.mcp import tools as impl

    info = impl.workspace_info("local")
    assert info["device"] == "local"
    assert info["deployable"] is False
    assert info["tools_total"] == len(ToolRegistry.list_all())


def test_a_local_only_tool_runs_here_rather_than_asking_for_a_deployment(monkeypatch) -> None:
    """A remote session still answers a tool whose inputs live on this machine.

    The server is a stdio process on the caller's own machine, so "not available on a remote
    worker" is not a refusal — it names where the tool has to run, and that is here.
    """
    from proto_tools.mcp import tools as impl

    called: dict[str, object] = {}

    def fake_function(payload, cfg):
        called["device"] = cfg.device
        raise RuntimeError("ran in-process")

    spec = ToolRegistry.get(_LOCAL_ONLY_TOOL)
    monkeypatch.setattr(spec, "function", fake_function)

    example = ToolRegistry.get_example_input(_LOCAL_ONLY_TOOL)
    result = impl.run_tool(_LOCAL_ONLY_TOOL, inputs=example.model_dump(mode="json"), device="modal")

    assert called["device"] == "cpu", "the tool must be told it is running locally"
    assert result["ok"] is False
    assert "ran in-process" in result["error"], "the tool ran; it did not report a missing deployment"
    assert "needs_human" not in result, "nothing here needs a human to deploy anything"


def test_a_deployed_tool_is_still_dispatched(monkeypatch) -> None:
    """The fallback must not swallow the normal case."""
    from proto_tools.mcp import tools as impl
    from proto_tools.modal import client

    sent: dict[str, object] = {}

    def fake_dispatch(tool_key, payload, cfg):
        sent["tool"] = tool_key
        raise RuntimeError("dispatched")

    monkeypatch.setattr(client, "dispatch_to_modal", fake_dispatch)
    example = ToolRegistry.get_example_input("esm2-embedding")
    result = impl.run_tool("esm2-embedding", inputs=example.model_dump(mode="json"), device="modal")

    assert sent["tool"] == "esm2-embedding"
    assert result["ok"] is False


def test_a_deployed_local_cpu_tool_is_not_flagged_as_running_here() -> None:
    """local_cpu means a deployment is optional, not absent — two of them have one.

    Reporting these as in-process would misstate where the call went, and would contradict the
    ``deployed`` flag sitting beside it.
    """
    from proto_tools.mcp.tools import runs_in_process

    assert not runs_in_process("pdockq2")
    assert not runs_in_process("structure-metrics")


def test_an_undeployed_local_cpu_tool_is_flagged_as_running_here() -> None:
    """With nothing to serve it, the wrapper runs it in-process and ran_on must say so."""
    from proto_tools.mcp.tools import runs_in_process

    assert runs_in_process("pdb-fetch-entry")


def test_the_remote_catalogue_includes_tools_answered_in_process() -> None:
    """An agent on a remote device can still call a tool that has no deployment."""
    from proto_tools.mcp import tools as impl

    listed = {entry["tool"]: entry for entry in impl.list_tools(deployed_only=True, device="modal")}
    entry = listed.get(_LOCAL_ONLY_TOOL)
    assert entry is not None, "a local_only tool is usable on modal, so it belongs in the catalogue"
    assert entry["available"] is True
    assert entry["runs_in_process"] is True
    assert entry["deployed"] is False, "it is usable, but nothing is deployed for it"


def test_the_mcp_remote_guard_is_the_registry_one() -> None:
    """One definition of 'remote', derived from RemoteDevice, so the two cannot drift."""
    from proto_tools.mcp.device import is_remote
    from proto_tools.utils.device import is_remote_device

    assert is_remote is is_remote_device
