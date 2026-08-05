"""MCP server surface.

Offline only — anything needing a live deployment is covered by the deploy
smoke tests, not here.
"""

import asyncio
import json
import os

import pytest

_READ_ONLY_SURFACE = {
    "workspace_info",
    "list_tools",
    "search_tools",
    "get_tool_schema",
    "get_tool_example",
    "get_tool_citation",
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


# ── Unknown tool keys ───────────────────────────────────────────────────────


@pytest.mark.parametrize("verb", ["get_tool_schema", "get_tool_example", "run_tool"])
def test_an_unknown_key_is_a_result_rather_than_a_protocol_error(verb: str):
    """A raise reaches the agent as ToolError — "this call is broken" — and ends the attempt.

    The key is the one argument an agent has to invent, so guessing it wrong has to be as
    recoverable as passing bad arguments already is.
    """
    import fastmcp

    from proto_tools.mcp import build_server

    async def call():
        async with fastmcp.Client(build_server("local")) as client:
            return await client.call_tool(verb, {"tool_key": "esmfold"})

    result = asyncio.run(call()).data
    assert result["ok"] is False
    assert "esmfold-prediction" in result["did_you_mean"]
    assert "<model>-<action>" in result["hint"]


def test_a_model_name_suggests_every_action_registered_for_it():
    """Whole-key edit distance returns nothing for these, which is the reported failure."""
    from proto_tools.tools import ToolRegistry

    assert ToolRegistry.suggest_keys("esm2") == ["esm2-embedding", "esm2-gradient", "esm2-sample", "esm2-score"]
    assert ToolRegistry.suggest_keys("boltz2") == ["boltz2-affinity", "boltz2-prediction"]
    assert ToolRegistry.suggest_keys("tmalign") == ["tmalign-alignment"], "difflib answers this with mafft-align"


def test_a_sibling_model_ranks_below_the_one_named():
    """`esmfold2` is a different model, so it must not displace `esmfold`'s own actions."""
    from proto_tools.tools import ToolRegistry

    assert ToolRegistry.suggest_keys("esmfold")[-1] == "esmfold2-prediction"


def test_a_misspelled_model_still_resolves():
    """The fallback exists for typos; whole-key distance scores them too poorly to return."""
    from proto_tools.tools import ToolRegistry

    assert "esmfold-prediction" in ToolRegistry.suggest_keys("esmfld")
    assert "esmfold-prediction" in ToolRegistry.suggest_keys("ESMFold"), "a name read from a paper is capitalised"


# ── Choosing a tool from the listing ────────────────────────────────────────


@pytest.mark.parametrize("device", ["modal", "proto", "local"])
def test_the_instructions_name_every_category_the_registry_has(device: str):
    """The category filter is useless if the vocabulary is not visible; a new one must not be missed."""
    from proto_tools.mcp.server import instructions_for
    from proto_tools.tools import ToolRegistry

    text = instructions_for(device)
    missing = sorted(c for c in {s.category for s in ToolRegistry.list_all()} if c not in text)
    assert not missing, f"categories absent from the {device} instructions: {missing}"


def test_listing_carries_what_choosing_a_tool_needs():
    """Names alone forced a schema fetch per candidate, which is the expensive way to choose."""
    from proto_tools.mcp import tools as impl

    entry = next(e for e in impl.list_tools(device="local") if e["tool"] == "esmfold-prediction")

    assert entry["category"] == "structure_prediction"
    assert entry["summary"] and entry["uses_gpu"] is True


def test_a_category_narrows_the_listing_and_an_unknown_one_names_the_choices():
    """An agent that mistypes a facet should learn the vocabulary rather than get an empty list."""
    from proto_tools.mcp import tools as impl

    listed = impl.list_tools(deployed_only=False, category="structure_prediction", device="local")
    assert listed and {e["category"] for e in listed} == {"structure_prediction"}
    assert len(listed) < len(impl.list_tools(deployed_only=False, device="local"))

    refused = impl.list_tools(category="structure-prediction", device="local")
    assert refused[0]["ok"] is False
    assert "structure_prediction" in refused[0]["categories"]


def test_an_in_process_tool_says_it_needs_no_deployment_and_why(monkeypatch):
    """Reading "not deployed" as "deploy it first" is the wrong move: it already runs here.

    The reason matters just as much — it is where a tool says it runs for hours, which its
    description does not.
    """
    from proto_tools.mcp import tools as impl

    monkeypatch.setattr(impl, "_registry", dict)
    monkeypatch.setattr(impl, "deployed_keys", lambda device: set())
    monkeypatch.setattr(impl, "answered_in_process_keys", lambda: {"bindcraft-design"})

    entry = next(e for e in impl.list_tools(device="modal") if e["tool"] == "bindcraft-design")

    assert entry["runs_in_process"] is True
    assert "no deployment is needed" in entry["note"]
    assert "multi-hour" in entry["note"], "the duration warning exists only in the local_only reason"


def test_a_citation_is_reachable_for_the_tools_that_register_one():
    """An agent reporting a result has to attribute the method, and 132 of 134 ship a cite.bib."""
    from proto_tools.mcp import tools as impl

    cite = impl.get_tool_citation("esmfold-prediction")
    assert cite["doi"] == "10.1126/science.ade2574"
    assert "@article" in cite["bibtex"]


def test_the_raised_error_still_names_the_key_and_stays_a_ValueError():
    """Callers outside the MCP — the CLI, the Python API — depend on both."""
    from proto_tools.tools import ToolRegistry

    with pytest.raises(ValueError, match="Unknown tool 'esm2'"):
        ToolRegistry.get("esm2")


def _unauthenticated(monkeypatch, config_path, **env):
    """Report what an unauthenticated caller sees, whatever this host's own credentials are."""
    import modal

    from proto_tools.mcp import tools as impl

    def refuse(*args, **kwargs):
        raise RuntimeError("AuthError: Token missing.")

    monkeypatch.setattr(modal.Client, "from_env", refuse)
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(config_path))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return impl.workspace_info("modal")


def test_the_auth_hint_names_the_environment_variables(monkeypatch, tmp_path):
    """`modal token new` is unactionable in a container, so the hint has to name the other path."""
    absent = tmp_path / "absent.toml"
    info = _unauthenticated(monkeypatch, absent)

    assert info["authenticated"] is False
    assert "MODAL_TOKEN_ID" in info["hint"] and "MODAL_TOKEN_SECRET" in info["hint"]
    assert info["credentials_checked"] == {
        "MODAL_TOKEN_ID": "unset",
        "MODAL_TOKEN_SECRET": "unset",
        "MODAL_PROFILE": "unset",
        "config_file": str(absent),
        "config_file_state": "absent",
    }


def test_a_half_configured_caller_sees_which_half_is_missing(monkeypatch, tmp_path):
    """One variable set without the other fails identically to setting neither."""
    checked = _unauthenticated(monkeypatch, tmp_path / "absent.toml", MODAL_TOKEN_ID="ak-secret")["credentials_checked"]

    assert checked["MODAL_TOKEN_ID"] == "set"
    assert checked["MODAL_TOKEN_SECRET"] == "unset"


def test_an_empty_variable_is_not_reported_as_absent(monkeypatch, tmp_path):
    """Modal reads an empty variable and fails, rather than falling back to a readable file."""
    config = tmp_path / "modal.toml"
    config.write_text("[default]\n")
    checked = _unauthenticated(monkeypatch, config, MODAL_TOKEN_ID="", MODAL_TOKEN_SECRET="")["credentials_checked"]

    assert checked["MODAL_TOKEN_ID"] == "empty", "reporting this as unset hides why a good config file is ignored"
    assert checked["config_file_state"] == "readable"


def test_credentials_are_reported_by_presence_and_never_by_value(monkeypatch, tmp_path):
    """The payload goes to an agent and into logs; a token in it would leak from both."""
    info = _unauthenticated(
        monkeypatch,
        tmp_path / "absent.toml",
        MODAL_TOKEN_ID="ak-leak-me",
        MODAL_TOKEN_SECRET="as-leak-me",
    )

    assert "leak-me" not in json.dumps(info)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode, so 0o000 proves nothing")
def test_an_unreadable_config_is_distinguished_from_an_absent_one(monkeypatch, tmp_path):
    """Both look like no file to an existence check, but only one is fixed by writing a token."""
    present = tmp_path / "modal.toml"
    present.write_text("[default]\n")
    assert _unauthenticated(monkeypatch, present)["credentials_checked"]["config_file_state"] == "readable"

    present.chmod(0o000)
    state = _unauthenticated(monkeypatch, present)["credentials_checked"]["config_file_state"]
    present.chmod(0o644)
    assert state == "unreadable"


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


def test_search_matches_natural_language_queries(monkeypatch):
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
    # setattr, not assign-then-del: deleting removes the real function for the rest of the
    # session rather than restoring it, and every later test calling it then fails.
    monkeypatch.setattr(impl, "list_tools", lambda **_kwargs: catalogue)
    found = impl.search_tools("compare two protein structures")
    assert [h["tool"] for h in found["hits"]][:1] == ["tmalign-alignment"], found


def test_search_is_capped_and_says_how_many_it_left_out():
    """Broad queries match half the catalogue; the agent should not pay for it to find that out."""
    from proto_tools.mcp import tools as impl

    found = impl.search_tools("compare two protein structures", deployed_only=False, limit=5, device="local")

    assert len(found["hits"]) == 5
    assert found["n_total"] > 5, "the total is what tells a caller whether raising the limit is worth it"
    assert found["hits"][0]["score"] >= found["hits"][-1]["score"], "hits are ordered by the score they carry"


def test_search_ranks_a_tool_named_for_the_query_above_one_that_merely_mentions_it():
    """Rewriting "fold" to "structure" used to discard the token that matches esmfold in the key."""
    from proto_tools.mcp import tools as impl

    hits = impl.search_tools("fold a protein", deployed_only=False, limit=133, device="local")["hits"]
    ranked = [h["tool"] for h in hits]

    assert ranked.index("esmfold-prediction") < ranked.index("esm-if1-sample"), "inverse folding is the opposite"


def test_a_query_that_matches_nothing_says_what_to_do_instead():
    """An empty list reads as "no such capability exists" rather than "rephrase"."""
    from proto_tools.mcp import tools as impl

    found = impl.search_tools("crystallography", deployed_only=False, device="local")

    assert found["hits"] == [] and found["n_total"] == 0
    assert "list_tools(category=" in found["hint"] and "structure_prediction" in found["hint"]


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
