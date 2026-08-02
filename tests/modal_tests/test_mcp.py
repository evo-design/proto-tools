"""MCP server surface.

Offline only — anything needing a live deployment is covered by the deploy
smoke tests, not here.
"""

import asyncio

import pytest

_READ_ONLY_SURFACE = {
    "workspace_info",
    "list_tools",
    "search_tools",
    "get_tool_schema",
    "get_tool_example",
    "run_tool",
}


def test_server_registers_the_expected_surface():
    """The tool set is the agent-facing contract; changing it is a deliberate act."""
    from proto_tools.mcp import build_server

    assert {t.name for t in asyncio.run(build_server("modal").list_tools())} == _READ_ONLY_SURFACE | {"deploy_tool"}
    assert {t.name for t in asyncio.run(build_server("proto").list_tools())} == _READ_ONLY_SURFACE


def test_deploying_is_the_only_state_changing_tool():
    """Deployment incurs cost, and is the one such action exposed; nothing else mutates a workspace."""
    from proto_tools.mcp import build_server

    names = {t.name for t in asyncio.run(build_server("modal").list_tools())}
    unexpected = [n for n in names if any(w in n for w in ("stop", "delete", "destroy", "teardown"))]

    assert "deploy_tool" in names
    assert not unexpected, f"the MCP must not expose destructive actions: {unexpected}"


def test_proto_exposes_no_deploy_tool():
    """Proto's catalogue is fixed, so offering to deploy would promise something impossible."""
    from proto_tools.mcp import build_server

    names = {t.name for t in asyncio.run(build_server("proto").list_tools())}
    assert "deploy_tool" not in names


def test_every_tool_has_a_description():
    """Descriptions are how an agent decides what to call, including what it costs."""
    from proto_tools.mcp import build_server

    missing = [t.name for t in asyncio.run(build_server().list_tools()) if not (t.description or "").strip()]
    assert not missing, f"tools without descriptions: {missing}"


def test_get_tool_schema_returns_all_three_schemas():
    """An agent needs all three to build a valid call."""
    from proto_tools.mcp import tools as impl

    schema = impl.get_tool_schema("tmalign-alignment")
    assert schema["input_schema"]["properties"].keys() >= {"query_structure", "reference_structure"}
    assert "config_schema" in schema and "output_schema" in schema


def test_get_tool_example_matches_the_input_schema():
    """The example must be directly usable as run_tool inputs."""
    from proto_tools.mcp import tools as impl

    example = impl.get_tool_example("tmalign-alignment")
    assert example is not None
    assert set(example) <= set(impl.get_tool_schema("tmalign-alignment")["input_schema"]["properties"])


def test_run_tool_rejects_bad_arguments_without_dispatching(monkeypatch):
    """Validation must happen locally — a malformed call should never reach Modal."""
    from proto_tools.mcp import tools as impl
    from proto_tools.modal import client

    def explode(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("dispatched despite invalid arguments")

    monkeypatch.setattr(client, "dispatch_to_modal", explode)
    out = impl.run_tool("tmalign-alignment", {"nonsense": 1})
    assert out["ok"] is False
    assert "invalid arguments" in out["error"]
    assert "get_tool_schema" in out["hint"]


@pytest.mark.parametrize(
    "value,expect_saved",
    [("short string", False), ("x" * 5_000, True), ([1, 2, 3], False), (list(range(5_000)), True)],
)
def test_large_fields_are_written_to_disk(value, expect_saved, tmp_path):
    """Oversized fields must not land in an agent's context."""
    from proto_tools.mcp import tools as impl

    out = impl._summarize({"field": value}, "", tmp_path)
    saved = isinstance(out["field"], dict) and "_saved_to" in out["field"]
    assert saved is expect_saved
    if saved:
        assert tmp_path.joinpath(*out["field"]["_saved_to"].split("/")[-1:]).exists()


def test_summarize_recurses_into_nested_structures(tmp_path):
    """Large fields buried in nested output must still be written out."""
    from proto_tools.mcp import tools as impl

    out = impl._summarize({"outer": {"inner": "y" * 5_000}}, "", tmp_path)
    assert "_saved_to" in out["outer"]["inner"]


def test_search_matches_natural_language_queries():
    """Agents ask in prose; a literal substring search returns nothing for that.

    Regression for a real failure: search_tools("compare two protein
    structures") returned [] while "align" returned five correct hits, which
    reads to an agent as "no such tool exists".
    """
    from proto_tools.mcp import tools as impl

    catalogue = [
        {"tool": "tmalign-alignment", "app": "a", "deployed": True},
        {"tool": "esm2-score", "app": "b", "deployed": True},
    ]
    impl.list_tools = lambda **_kwargs: catalogue  # type: ignore[assignment]
    try:
        hits = impl.search_tools("compare two protein structures")
        assert [h["tool"] for h in hits][:1] == ["tmalign-alignment"], hits
    finally:
        del impl.list_tools


def test_example_elides_bulky_values():
    """An example shows shape; a structure tool's example is tens of KB of PDB."""
    from proto_tools.mcp import tools as impl

    elided = impl._elide({"structure": {"structure": "X" * 90_000, "structure_format": "pdb"}})
    inner = elided["structure"]
    assert inner["structure_format"] == "pdb", "small fields must survive"
    assert "elided" in inner["structure"] and len(inner["structure"]) < 200


def test_run_tool_requires_inputs_or_use_example():
    """Neither given is a caller error, not a crash."""
    from proto_tools.mcp import tools as impl

    out = impl.run_tool("tmalign-alignment")
    assert out["ok"] is False and "use_example" in out["error"]


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_mcp_help_prints_and_never_starts_the_server(flag, capsys, monkeypatch):
    """`--help` must print guidance and return, not launch a server that blocks on stdin."""
    from proto_tools.mcp import server

    def _must_not_run():
        raise AssertionError("--help must not build or run the stdio server")

    monkeypatch.setattr(server, "build_server", _must_not_run)
    server.main([flag])  # returns instead of blocking
    assert "proto-tools mcp" in capsys.readouterr().out


def test_mcp_no_args_would_start_the_server(monkeypatch):
    """With no flags, main proceeds to build and run the server (the normal path)."""
    from proto_tools.mcp import server

    started = {"ran": False}

    class _FakeServer:
        def run(self, show_banner):
            started["ran"] = True

    monkeypatch.setattr(server, "build_server", lambda device: started.update(device=device) or _FakeServer())
    server.main([])
    assert started["ran"] is True
    assert started["device"] == "modal", "the default backend must be modal"


# --- device switch ----------------------------------------------------------


def test_default_is_modal_even_with_an_api_key_present(monkeypatch):
    """Selecting proto off ambient env would make the backend depend on an exported variable."""
    from proto_tools.mcp.device import resolve_device

    monkeypatch.setenv("PROTO_API_KEY", "exported-for-something-else")
    monkeypatch.delenv("PROTO_MCP_DEVICE", raising=False)

    assert resolve_device() == "modal"


def test_proto_requires_a_key_and_says_so(monkeypatch):
    """The failure has to name the variable, or the caller cannot act on it."""
    import pytest as _pytest

    from proto_tools.mcp.device import DeviceUnavailableError, resolve_device

    monkeypatch.delenv("PROTO_API_KEY", raising=False)
    with _pytest.raises(DeviceUnavailableError, match="PROTO_API_KEY"):
        resolve_device("proto")


def test_unknown_device_is_rejected(monkeypatch):
    """A typo must not silently fall back to a working backend."""
    import pytest as _pytest

    from proto_tools.mcp.device import DeviceUnavailableError, resolve_device

    monkeypatch.delenv("PROTO_MCP_DEVICE", raising=False)
    with _pytest.raises(DeviceUnavailableError, match="unknown device"):
        resolve_device("cloud")


def test_only_modal_is_deployable():
    """Proto's catalogue is fixed; telling a caller to deploy to it sends them after nothing."""
    from proto_tools.mcp.device import is_deployable

    assert is_deployable("modal") is True
    assert is_deployable("proto") is False


def test_proto_unavailability_offers_no_deploy_command(monkeypatch):
    """On proto there is no command to relay, so suggesting one would mislead."""
    from proto_tools.mcp import tools as impl

    monkeypatch.setattr(
        impl, "_hosted_catalogue", lambda: {"x-tool": {"hosted": False, "unhosted_reason": "licensing"}}
    )
    out = impl._unavailable("proto", "x-tool", "boom")

    assert out["needs_human"] is False
    assert "licensing" in out["error"]
    assert "deploy" not in out["error"].lower()
    assert "cannot be deployed to" in out["fix"]


def test_modal_unavailability_is_actionable(monkeypatch):
    """On modal the user owns the workspace, so the deploy command is the fix."""
    from proto_tools.mcp import tools as impl

    out = impl._unavailable("modal", "x-tool", "app not deployed: run proto-tools deploy --apps x")

    assert out["needs_human"] is True
    assert "proto-tools deploy" in out["error"]


# --- deploy_tool ------------------------------------------------------------


def test_deploy_reports_each_build_phase():
    """A deploy takes minutes; without progress it is indistinguishable from a hang."""
    import asyncio as _asyncio
    from unittest.mock import patch

    from proto_tools.mcp import tools as impl

    def fake_deploy(app, environment=None, on_progress=None):
        for phase in ("building image", "running warmup", "deployed"):
            on_progress(phase)
        return True

    phases: list[str] = []

    async def report(phase: str) -> None:
        phases.append(phase)

    with patch("proto_tools.modal.deploy.deploy_app", fake_deploy):
        out = _asyncio.run(impl.deploy_tool("tmalign-alignment", "some-env", report))

    assert out["ok"] is True
    assert "building image" in phases
    assert phases[-1] == "deployed"


def test_deploy_rejects_a_tool_it_does_not_serve():
    """A wrong key must not reach the deploy path and spend anything."""
    import asyncio as _asyncio

    from proto_tools.mcp import tools as impl

    async def report(_phase: str) -> None:
        return None

    out = _asyncio.run(impl.deploy_tool("not-a-real-tool", "some-env", report))
    assert out["ok"] is False


def test_a_failed_deploy_points_at_the_build_log():
    """The build output is the only place the cause is recorded."""
    import asyncio as _asyncio
    from unittest.mock import patch

    from proto_tools.mcp import tools as impl

    async def report(_phase: str) -> None:
        return None

    with patch("proto_tools.modal.deploy.deploy_app", lambda *a, **k: False):
        out = _asyncio.run(impl.deploy_tool("tmalign-alignment", "some-env", report))

    assert out["ok"] is False
    assert "log" in out["error"]


def test_build_output_is_summarised_not_streamed():
    """Forwarding every line would bury the status an agent shows in one place."""
    from proto_tools.modal.deploy import describe_progress

    assert describe_progress("=> Step 3: RUN pip install numpy") == "RUN pip install numpy"
    assert describe_progress("Building image im-abc123") == "building image"
    assert "warmup" in (describe_progress("Running function _warmup") or "")
    assert describe_progress("✓ App deployed in 143.8s! 🎉") == "deployed"
    assert describe_progress("  Downloading numpy-2.4.6.whl (18 MB)") is None


# ============================================================================
# The MCP surface enforces the same guards as the registry
# ============================================================================
def test_run_tool_refuses_a_setting_the_device_cannot_honour():
    """The registry checks this before dispatching; run_tool does not go through the registry.

    Without the check, a custom checkpoint on Proto reaches a container and fails there, on a
    path that is the most agent-facing one this package ships.
    """
    from proto_tools.mcp.tools import run_tool
    from proto_tools.tools import ToolRegistry

    example = ToolRegistry.get_example_input("parade-gradient")
    result = run_tool(
        "parade-gradient",
        inputs=example.model_dump(mode="json"),
        config={"checkpoint": "https://example.invalid/x.ckpt"},
        device="proto",
    )

    assert result["ok"] is False
    assert result["not_supported_on"] == "proto"
    assert "custom checkpoint" in result["error"]


def test_every_deploy_route_records_fingerprints():
    """An absent manifest reads as "aligned", so a route that skips this is silently exempt.

    ``deploy_app`` owns the call, rather than each caller, because the MCP path reaches it
    without going through the CLI that used to do the recording.
    """
    import inspect

    from proto_tools.modal.deploy import deploy_app

    assert "record_fingerprints(" in inspect.getsource(deploy_app)
