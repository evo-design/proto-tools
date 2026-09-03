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

- `deploy_tool` requires the name of the Modal environment to deploy into. Users
  create one while setting up their account, and the documented name is
  `proto-env`. `workspace_info` reports the one in use, which is where a deploy
  should go; ask the user before deploying anywhere else, since a workspace can
  hold several.

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

INSTRUCTIONS = {
    "modal": _MODAL_INSTRUCTIONS + "\n" + _KEY_CONVENTION,
    "proto": _PROTO_INSTRUCTIONS + "\n" + _KEY_CONVENTION,
    "local": _LOCAL_INSTRUCTIONS + "\n" + _KEY_CONVENTION,
}


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

    @mcp.tool
    def workspace_info() -> dict[str, Any]:
        """Show which Modal workspace and environment calls go to, and how many apps are deployed.

        Call this first if anything seems misconfigured — it reports whether
        Modal credentials are present at all.
        """
        return impl.workspace_info(device)

    @mcp.tool
    def list_tools(deployed_only: bool = True, category: str | None = None) -> dict[str, Any]:
        """List available bioinformatics tools, with what each one is for.

        Each entry carries its category, a one-line summary, and whether it
        needs a GPU, which is usually enough to choose without fetching a
        schema for every candidate. Pass a category to narrow the list.

        Defaults to only those actually deployed in this workspace. Pass
        deployed_only=False to see the full catalogue, including tools the
        user would have to deploy first.

        Entries arrive under `tools`, the same as search_tools, each with the
        key to run under `tool_key`.
        """
        found = impl.list_tools(deployed_only=deployed_only, category=category, device=device)
        return {"tools": found, "n_total": len(found)}

    @mcp.tool
    def search_tools(query: str, deployed_only: bool = True, limit: int = 10) -> dict[str, Any]:
        """Find tools by keyword, matching the tool key, category and summary.

        Useful for questions like "what can fold a protein" or "which tools
        score sequences". Returns the best `limit` matches under `tools`, each
        with the `score` out of 100 it ranked on, plus `n_total` for how many
        matched in all — raise `limit` only if the total says it is worth it.

        Run what you find by its `tool_key`, exactly as given. A display name
        is not a key, and inferring one produces a tool that does not exist.
        """
        return impl.search_tools(query, deployed_only=deployed_only, limit=limit, device=device)

    @mcp.tool
    def get_tool_schema(tool_key: str) -> dict[str, Any]:
        """Get the input, config and output schemas for a tool.

        Call before run_tool unless you already know the shape — arguments are
        validated strictly and unknown fields are rejected.
        """
        return impl.get_tool_schema(tool_key)

    @mcp.tool
    def get_tool_example(tool_key: str) -> dict[str, Any] | None:
        """Get a known-good example input for a tool, or null if it declares none.

        Shows the shape; bulky values such as structure coordinates are elided.
        To actually run the example, call run_tool with use_example=True.
        """
        return impl.get_tool_example(tool_key)

    @mcp.tool
    def get_tool_info(tool_key: str) -> dict[str, Any]:
        """Where a tool comes from: who built the model, how to cite it, and the code that runs it.

        Use this when reporting results, so the method is attributed and a reader can follow it
        back to the paper, the weights, or the implementation. Separate from get_tool_schema, which
        answers what arguments a tool takes and is called before every run.
        """
        return impl.get_tool_info(tool_key)

    @mcp.tool
    async def run_tool(
        ctx: Context,
        tool_key: str,
        inputs: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        output_dir: str | None = None,
        use_example: bool = False,
        run_on: str | None = None,
    ) -> dict[str, Any]:
        """Run a tool on this session's backend and return the result.

        Blocks until it finishes. Most tools return in seconds once warm, but
        the first call after a few minutes idle pays a container start and
        model load, and a few tools (binder design, diffusion) legitimately run
        for many minutes. Check the tool description before calling.

        run_on overrides the backend for this call alone: "local" runs it on
        this machine, "modal" dispatches it to the deployment. Useful for
        sending a small CPU tool to the local machine without paying a
        container start, or one GPU tool to Modal from a local session. Omit it
        to use the session's own backend.

        Some tools are answered in this process whatever the backend, because
        they need no GPU and no environment, or cannot be deployed at all. The
        `ran_on` field in the result reports where the call actually ran.

        Structure inputs take a file path or an http(s) URL in place of inlined
        content — {"query_structure": "/path/to/file.pdb"} — so a file already on
        disk, such as another tool's output, never has to be read into the call.
        Other bulky inputs, such as MSAs, take their content rather than a path.

        Pass use_example=True to run the tool's canonical example input without
        supplying it — useful for structure tools whose inputs are very large.

        Large fields are written under output_dir (default ./proto_tools_outputs)
        and returned as paths.
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

        @mcp.tool
        async def deploy_tool(tool_key: str, environment: str, ctx: Context) -> dict[str, Any]:
            """Deploy the Modal app serving a tool, after the user approves the expenditure.

            Intended for a tool that list_tools reports as not deployed.

            A deployment builds a container image and then executes the tool once, on a
            GPU where the tool requires one. Both are billed to the user's own Modal
            account, and both occur before any result is returned to the caller. The
            user is asked to confirm beforehand; declining deploys nothing and incurs
            no cost.

            The operation may take several minutes. Progress is reported as the build
            advances through its phases.

            The environment argument is required. Naming the target Modal environment
            explicitly, rather than inheriting whichever is ambient, prevents an
            accidental deployment to production.
            """
            app = impl.app_for_tool(tool_key)
            if app is None:
                return {"ok": False, "error": f"{tool_key!r} is not a tool this deployment serves."}

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
