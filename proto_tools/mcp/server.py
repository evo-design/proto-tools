"""Registration of the MCP tool surface, and the agent-facing instructions."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import sys
from dataclasses import dataclass
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.server.elicitation import AcceptedElicitation
from mcp.types import ToolAnnotations

from proto_tools.mcp import tools as impl
from proto_tools.mcp.device import Device, DeviceUnavailableError, resolve_device


@dataclass
class _Approval:
    """The one answer a deploy needs. MCP elicitation carries object schemas, not bare booleans."""

    approve: bool


_MODAL_INSTRUCTIONS = """Run bioinformatics tools on the user's own Modal deployment.

Users can choose to deploy tools to their own Modal environments. Deployed tools
retain their model weights between calls in the user's Modal storage, making
subsequent calls fast.

- `list_tools` reports the tools this user has deployed. Pass `deployed_only=false`
  for the full catalogue of deployable tools, which identifies what remains
  available to deploy.

- If a user needs a tool that is deployable but that they have not yet deployed,
  use `deploy_tool`. It prompts the user to confirm before proceeding.

- `deploy_tool` deploys into the Modal environment `workspace_info` reports, which
  users create while setting up their account (the documented name is
  `proto-env`). Pass `environment` only to deploy somewhere else, and ask the user
  first, since a workspace can hold several.

- Any deployed tool can then be run using `run_tool`.

IMPORTANT: Deploying and running tools bills activity to the user's Modal account.

Large outputs, such as predicted structures and embeddings, are written to disk
and returned as file paths rather than being returned inline.
"""

_PROTO_INSTRUCTIONS = """Run bioinformatics tools on Proto's hosted service.

Proto offers limited access to hosted deployments, available to users who hold an
API key. At present, API keys to the Proto service are provided only to a small
set of collaborators.

Begin with `list_tools` to establish what is hosted. Unlike Modal, where the user
deploys tools themselves, the catalogue available through Proto is fixed: do not
propose deploying a tool. Where a tool is unavailable, `run_tool` explains why and
refers to the `modal` backend, which the user would need to configure themselves.

Large outputs, such as predicted structures and embeddings, are written to disk
and returned as file paths rather than inline.
"""

_LOCAL_INSTRUCTIONS = """Run bioinformatics tools on this machine.

Tools execute in this process rather than on a remote backend, so there is nothing
to deploy. Every registered tool is available: begin with `list_tools` to see them.

Each tool builds its own isolated environment and downloads its model weights the
first time it runs, so a first call can take several minutes. Later calls reuse
both and are fast.

A tool that requires a GPU requires one on this machine. Check `workspace_info`
and the tool's description before calling one.

Large outputs, such as predicted structures and embeddings, are written to disk
and returned as file paths rather than inline.
"""

# Stated for every backend because a key is the one argument a caller has to invent, and
# guessing a model name read from a paper is the most common way a call goes wrong.
_KEY_CONVENTION = """Tool keys are `<model>-<action>`, such as `esmfold-prediction` or
`esm2-embedding`. A model name on its own is not a key: several actions usually exist for
one model. Use `search_tools` or `list_tools` to resolve a name into a key.
"""

_WORKFLOW = """Before calling `run_tool`, first call `get_tool_schema` for the selected tool. Then call
`get_tool_example`; if an example exists, use its structure as the template for the request. Do
not infer input field names or nesting from the biological task or tool name. Make only one
`run_tool` execution attempt after inspecting this metadata. `get_tool_schema` reports
`has_example`, so you know whether there is one to fetch, and `run_tool(use_example=true)` runs
the canonical example unchanged.

Use `get_tool_info` when reporting a result: the citation and DOI for the method, the model
authors' own repository, paper and weights, and the code that ran it. Attribute the work you used.
"""

INSTRUCTIONS = {
    device: text + "\n" + _WORKFLOW + "\n" + _KEY_CONVENTION
    for device, text in (("modal", _MODAL_INSTRUCTIONS), ("proto", _PROTO_INSTRUCTIONS), ("local", _LOCAL_INSTRUCTIONS))
}


def _read_only(title: str) -> ToolAnnotations:
    """Annotations for a tool that only reads, matching the hosted server's."""
    return ToolAnnotations(
        title=title, readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
    )


def instructions_for(device: Device) -> str:
    """Return the instructions for ``device``, naming the categories the registry holds.

    The category list is read rather than written down, so a new category reaches the agent
    without anyone remembering to update prose.
    """
    from proto_tools.tools import ToolRegistry

    categories = sorted({spec.category for spec in ToolRegistry.list_all()})
    return f"{INSTRUCTIONS[device]}\n`list_tools` accepts a `category` filter. The categories are: {', '.join(categories)}.\n"


def build_server(device: Device = "modal") -> FastMCP:
    """Construct the MCP server with its tool surface registered.

    Args:
        device (Device): Backend every call in this session goes to. Fixed at
            construction so listing and running can never disagree about what
            is available.

    Returns:
        FastMCP: The configured server.
    """
    mcp: FastMCP = FastMCP(name=f"proto-tools ({device})", instructions=instructions_for(device))

    @mcp.tool(title="Workspace and credentials", annotations=_read_only("Workspace and credentials"))
    def workspace_info() -> dict[str, Any]:
        """Show where calls go and what is attached: the Modal workspace and environment, or this machine.

        Call this first if anything seems misconfigured, or to answer where tools will run. It
        reports whether credentials are present at all and how many apps are deployed.
        """
        return impl.workspace_info(device)

    @mcp.tool(title="List tools", annotations=_read_only("List tools"))
    def list_tools(deployed_only: bool = True, category: str | None = None) -> dict[str, Any]:
        """List bioinformatics tools, flagged by whether they can run for you.

        Defaults to what your workspace actually serves. Pass deployed_only=False for the full
        catalogue, which also shows what you could deploy.

        Entries arrive under `tools`, the same as search_tools, each carrying the key to run under
        `tool_key`. An entry with `outages` has an upstream service down; its `note` says which
        mode is affected.
        """
        return impl.catalogue(deployed_only=deployed_only, category=category, device=device)

    @mcp.tool(title="Search tools", annotations=_read_only("Search tools"))
    def search_tools(query: str, deployed_only: bool = True, limit: int = 10) -> dict[str, Any]:
        """Find tools by keyword, matching the tool key and its description.

        Returns the best `limit` matches under `tools`, each with the `score` out of 100 it ranked
        on, plus `n_total` for how many matched in all. Run what you find by its `tool_key`, exactly
        as given: a display name is not a key.
        """
        return impl.search_tools(query, deployed_only=deployed_only, limit=limit, device=device)

    @mcp.tool(title="Tool schema", annotations=_read_only("Tool schema"))
    def get_tool_schema(tool_key: str) -> dict[str, Any]:
        """Get the input, config and output schemas for a tool.

        Step 1 of the workflow: get_tool_schema, then get_tool_example when available, then one
        run_tool call. `has_example` says whether step 2 has anything to return. `outages` lists
        upstream outages: run_tool refuses a call that reaches one (`affects` names the mode, or
        null for the whole tool), so choose a setting that avoids it or tell the user it is down.
        """
        return impl.get_tool_schema(tool_key)

    @mcp.tool(title="Tool example input", annotations=_read_only("Tool example input"))
    def get_tool_example(tool_key: str) -> dict[str, Any] | None:
        """Get a known-good example input for a tool, or null if it declares none.

        Step 2 of the workflow: get_tool_schema, then get_tool_example when available, then one
        run_tool call. Use its structure as the template for the request rather than as
        illustration. `run_tool(use_example=true)` runs it unchanged.
        """
        return impl.get_tool_example(tool_key)

    @mcp.tool(title="Tool provenance", annotations=_read_only("Tool provenance"))
    def get_tool_info(tool_key: str) -> dict[str, Any]:
        """Where a tool comes from: who built the model, how to cite it, and the code that runs it.

        Deliberately not part of get_tool_schema. That call answers "what arguments does this take"
        and happens before every run; this one answers "what is this and who should be credited",
        which is wanted when reporting a result rather than producing one.

        `source` is this project's implementation -- the wrapper, environment, and example. `links`
        are the model's own: its authors' repository, paper and weights. `license` and
        `weights_access` say on what terms the weights may be used and how they are obtained.
        `outages` lists declared upstream outages, as get_tool_schema does.
        """
        return impl.get_tool_info(tool_key)

    @mcp.tool(
        title="Run a tool",
        annotations=ToolAnnotations(
            title="Run a tool", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
        ),
    )
    async def run_tool(
        ctx: Context,
        tool_key: str,
        inputs: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        use_example: bool = False,
        output_dir: str | None = None,
        run_on: str | None = None,
    ) -> dict[str, Any]:
        """Run a tool and return the result.

        Required workflow: Call get_tool_schema(tool_key) and get_tool_example(tool_key) before
        constructing inputs. Tool inputs vary and may use nested biological entity structures; never
        guess their shape. Pass `use_example=true` to run the canonical example unchanged.

        Blocks while it runs. A first call after a few minutes idle pays a container start and a
        model load, and some tools legitimately run for many minutes.

        run_on overrides the backend for this call alone: "local" runs it on this machine, "modal"
        dispatches it to the deployment. Omit it to use the session's own backend. Some tools are
        answered in this process whatever the backend, because they need no GPU and no environment,
        or cannot be deployed at all; `ran_on` in the result reports where the call actually ran.

        Structure inputs take a file path or an http(s) URL in place of inlined content, so a file
        already on disk, such as another tool's output, never has to be read into the call. Large
        output fields are written under output_dir (default ./proto_tools_outputs) and returned as
        paths.
        """
        if run_on is None:
            target = device
        else:
            try:
                target = resolve_device(run_on)
            except DeviceUnavailableError as exc:
                # A bad backend name is the caller's to correct, so it comes back as a result
                # rather than a protocol error, the same way an unknown tool key does.
                return {"ok": False, "error": str(exc), "valid_run_on": ["local", "modal", "proto"]}

        loop = asyncio.get_running_loop()
        # A task scheduled from another thread falls outside this request's context, where the
        # progress token lives, and silently reports nothing.
        messages: asyncio.Queue[str | None] = asyncio.Queue()

        async def pump() -> None:
            # Records carry no notion of how much is left, so the count is all there is to send.
            # A client reads it as "still going", which is the question being asked.
            for step in itertools.count(1):
                message = await messages.get()
                if message is None:
                    return
                # One unreportable message is not worth failing a tool call that is otherwise fine.
                with contextlib.suppress(Exception):
                    await ctx.report_progress(progress=step, message=message)

        pumping = asyncio.create_task(pump())
        # Named first, or the opening message is generic and a tool that reports nothing at all
        # stays that way throughout.
        messages.put_nowait(f"Running {tool_key}")

        def forward(record: dict[str, Any]) -> None:
            # On the tailer thread: a caller who disconnected leaves a closed loop, and raising
            # here would take down progress for a run that is otherwise fine.
            message = record.get("m")
            if not message:
                return
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(messages.put_nowait, str(message))

        try:
            # Off the loop, or the notifications above would sit in the queue until the tool
            # returned and arrive at once, which is the silence this exists to remove.
            return await asyncio.to_thread(
                impl.run_tool,
                tool_key,
                inputs,
                config,
                output_dir,
                use_example,
                device=target,
                on_record=forward,
            )
        finally:
            messages.put_nowait(None)
            await pumping

    if device == "modal":

        @mcp.tool(
            title="Deploy a tool",
            annotations=ToolAnnotations(
                title="Deploy a tool", readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=True
            ),
        )
        async def deploy_tool(tool_key: str, ctx: Context, environment: str | None = None) -> dict[str, Any]:
            """Build a tool's Modal app in your own workspace, so run_tool can dispatch to it.

            Takes several minutes and costs a build on your Modal account: the image is built and
            the tool executed once, on a GPU where it needs one. The user is asked to confirm
            before anything starts; declining deploys nothing and costs nothing. A tool only
            needs this once; calling it again replaces the deployed app with the current version.

            Deploys into the environment workspace_info reports. Pass `environment` only to deploy
            somewhere else, after the user has agreed to that environment.
            """
            from proto_tools.modal.app import resolve_environment

            app = impl.app_for_tool(tool_key)
            if app is None:
                return {"ok": False, "error": f"{tool_key!r} is not a tool this deployment serves."}
            environment = resolve_environment(environment)

            answer = await ctx.elicit(
                f"Deploy {app} to Modal environment {environment!r}?\n\n"
                f"This builds a container image and then executes {tool_key} once. Both are "
                f"billed to your own Modal account, and both occur before any result is "
                f"returned. It may take several minutes. Declining incurs no cost.",
                response_type=_Approval,
            )
            # Declined, cancelled, or a client that cannot ask at all: none of them approve.
            # Read defensively so an unexpected payload shape reads as refusal, never consent.
            approved = isinstance(answer, AcceptedElicitation) and bool(getattr(answer.data, "approve", False))
            if not approved:
                return {"ok": False, "app": app, "error": "declined; nothing was deployed"}

            async def report(phase: str) -> None:
                await ctx.report_progress(progress=0, message=phase)

            return await impl.deploy_tool(tool_key, environment, report)

    return mcp


HELP = """proto-tools-mcp — run the proto-tools MCP server over stdio.

Exposes your deployed tools to MCP-compatible agents (e.g. Claude Code). The
server speaks the MCP protocol on stdin/stdout, so run it directly only to
register it with an agent, not interactively:

    claude mcp add proto-tools --scope user -- proto-tools-mcp

For a client configured through JSON, such as Claude Desktop or Cursor:

    {"mcpServers": {"proto-tools": {"command": "proto-tools-mcp"}}}

The console script is preferred over `python -m proto_tools.mcp`: pip pins the
interpreter it was installed into, so a client that does not inherit your shell's
environment still starts the right Python.

Tools run on your own Modal workspace, which needs credentials (`modal token new`)
and the tool deployed (`proto-tools deploy --apps <name>`).

Options:
  -h, --help    Show this message and exit.
"""


def main(argv: list[str] | None = None) -> None:
    """Run the server over stdio, or print help and exit for ``-h``/``--help``.

    ``--help`` is handled here so probing the entry point prints guidance
    instead of launching a server that blocks on stdin — a stdio server gives
    no feedback, so a curious ``--help`` would otherwise look like a hang.

    The banner is suppressed: stdio transport uses stdout for the protocol
    itself, so anything decorative printed there risks corrupting it.
    """
    args = sys.argv[1:] if argv is None else argv
    if "-h" in args or "--help" in args:
        print(HELP)
        return

    requested = None
    if "--device" in args:
        index = args.index("--device")
        requested = args[index + 1] if index + 1 < len(args) else ""
    try:
        device = resolve_device(requested)
    except DeviceUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    build_server(device).run(show_banner=False)
