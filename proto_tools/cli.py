"""Command-line entry point for proto-tools discovery, docs, and runs.

Reachable as the ``proto-tools`` shell command after ``pip install``, or as
``python -m proto_tools`` without one.

The verbs ``list``, ``search``, ``schema``, ``example``, ``info``, ``run`` and
``workspace`` mirror the MCP server's tools and call the same functions in
:mod:`proto_tools.mcp.tools`, so an agent gets identical answers whichever
surface it uses. Their JSON output is the MCP payload, and a payload with
``ok: false`` exits 1. The remaining verbs are developer docs views over
``ToolRegistry``; they default to human-readable text and accept ``--json``
where they return structured data.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from proto_tools.mcp import tools as impl
from proto_tools.mcp.device import DeviceUnavailableError, resolve_device
from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)


# =============================================================================
# Output helpers
# =============================================================================


def _dump_json(value: Any) -> str:
    """Render any value as pretty-printed JSON."""
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, list) and value and isinstance(value[0], BaseModel):
        return json.dumps([v.model_dump() for v in value], indent=2, default=str)
    if isinstance(value, dict) and value and isinstance(next(iter(value.values()), None), list):
        # catalog(): {category: [ToolSpec]}
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, list):
                out[k] = [item.model_dump() if isinstance(item, BaseModel) else item for item in v]
            else:
                out[k] = v
        return json.dumps(out, indent=2, default=str)
    return json.dumps(value, indent=2, default=str)


def _summary_line(
    key: str, category: str, uses_gpu: bool, description: str, outage_scopes: list[str | None] | None = None
) -> str:
    """One-line text summary of a tool for list-style output.

    ``outage_scopes`` holds each declared outage's ``affects``: ``None`` marks the whole tool offline,
    a label marks only that mode degraded.
    """
    gpu = " (GPU)" if uses_gpu else ""
    scopes = outage_scopes or []
    if None in scopes:
        status = " (OFFLINE)"
    elif scopes:
        status = f" (DEGRADED: {', '.join(s for s in scopes if s)})"
    else:
        status = ""
    return f"{key:40s}  [{category}]{gpu}{status}  {description}"


def _entry_summary(entry: dict[str, Any]) -> str:
    """One-line text summary of an MCP list/search entry."""
    scopes = [o["affects"] for o in entry.get("outages", [])]
    return _summary_line(entry["tool_key"], entry["category"], entry["uses_gpu"], entry["summary"], scopes)


def _spec_summary(spec: ToolSpec) -> str:
    """One-line text summary of a ToolSpec for list-style output."""
    scopes = [o.affects for o in spec.outages]
    return _summary_line(spec.key, spec.category, spec.uses_gpu, spec.description, scopes)


def _emit(payload: dict[str, Any] | None) -> int:
    """Print an MCP payload as JSON, exiting 1 when it reports a failure."""
    print(_dump_json(payload))
    return 1 if payload and payload.get("ok") is False else 0


def _json_arg(value: str | None) -> dict[str, Any] | None:
    """Parse a ``--inputs``/``--config`` value: inline JSON, or ``@path`` to a JSON file."""
    if value is None:
        return None
    text = Path(value[1:]).read_text() if value.startswith("@") else value
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


# =============================================================================
# Verbs mirroring the MCP tools
# =============================================================================


def _cmd_list(args: argparse.Namespace) -> int:
    """``proto-tools list [--category C] [--device D] [--all]``: MCP ``list_tools``."""
    payload = impl.catalogue(deployed_only=not args.all, category=args.category, device=resolve_device(args.device))
    if args.json or payload.get("ok") is False:
        return _emit(payload)
    for entry in payload["tools"]:
        print(_entry_summary(entry))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """``proto-tools search <query> [--limit N] [--device D] [--all]``: MCP ``search_tools``."""
    payload = impl.search_tools(
        args.query, deployed_only=not args.all, limit=args.limit, device=resolve_device(args.device)
    )
    if args.json:
        return _emit(payload)
    for entry in payload["tools"]:
        print(f"{entry['score']:3d}  {_entry_summary(entry)}")
    if "hint" in payload:
        print(payload["hint"], file=sys.stderr)
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    """``proto-tools schema <tool>``: MCP ``get_tool_schema``."""
    return _emit(impl.get_tool_schema(args.tool))


def _cmd_example(args: argparse.Namespace) -> int:
    """``proto-tools example <tool> [--as-python]``: MCP ``get_tool_example``."""
    payload = impl.get_tool_example(args.tool)
    if payload is None:
        print(f"No example input defined for '{args.tool}'.", file=sys.stderr)
        return 1
    example = ToolRegistry.get_example_input(args.tool) if args.as_python and payload.get("ok") is not False else None
    if example is None:
        return _emit(payload)
    print(_render_example_as_python(ToolRegistry.get(args.tool), example), end="")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """``proto-tools info <tool>``: MCP ``get_tool_info``."""
    return _emit(impl.get_tool_info(args.tool))


def _cmd_run(args: argparse.Namespace) -> int:
    """``proto-tools run <tool> [--inputs ...] [--config ...] [--example]``: MCP ``run_tool``."""
    return _emit(
        impl.run_tool(
            args.tool,
            inputs=_json_arg(args.inputs),
            config=_json_arg(args.config),
            output_dir=args.output_dir,
            use_example=args.example,
            device=resolve_device(args.device),
        )
    )


def _cmd_workspace(args: argparse.Namespace) -> int:
    """``proto-tools workspace [--device D]``: MCP ``workspace_info``."""
    return _emit(impl.workspace_info(resolve_device(args.device)))


# =============================================================================
# Developer docs verbs
# =============================================================================


def _cmd_categories(args: argparse.Namespace) -> int:
    """``proto-tools categories``."""
    cats = ToolRegistry.list_categories()
    if args.json:
        print(_dump_json(cats))
    else:
        for c in cats:
            print(c)
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    """``proto-tools catalog``."""
    cat = ToolRegistry.catalog()
    if args.json:
        print(_dump_json(cat))
    else:
        for category, specs in cat.items():
            print(f"\n## {category}")
            for spec in specs:
                print(f"  {_spec_summary(spec)}")
    return 0


def _cmd_docs(args: argparse.Namespace) -> int:
    """``proto-tools docs <tool> [--no-toolkit-notes] [--no-license]``."""
    entry = ToolRegistry.get_tool_docs(
        args.tool,
        include_toolkit_notes=not args.no_toolkit_notes,
        include_license=not args.no_license,
    )
    if entry is None:
        print(f"No README entry found for tool '{args.tool}'.", file=sys.stderr)
        return 1

    if args.json:
        print(_dump_json(entry))
        return 0

    print(f"## {entry.label} (`{entry.key}`)\n")
    print(entry.intro)
    if entry.applications:
        print("\n### Applications\n")
        print(entry.applications)
    if entry.usage_tips:
        print("\n### Usage Tips\n")
        print(entry.usage_tips)
    if entry.toolkit_notes:
        print("\n### Toolkit Notes\n")
        print(entry.toolkit_notes)
    if entry.license:
        print("\n### License\n")
        print(_dump_json(entry.license))
    return 0


def _cmd_readme(args: argparse.Namespace) -> int:
    """``proto-tools readme <tool>``."""
    print(ToolRegistry.get_readme(args.tool))
    return 0


def _cmd_section(args: argparse.Namespace) -> int:
    """``proto-tools section <tool> <heading>``."""
    body = ToolRegistry.get_readme_section(args.tool, args.heading)
    if body is None:
        print(f"Section '{args.heading}' not found in README for '{args.tool}'.", file=sys.stderr)
        return 1
    print(body)
    return 0


def _cmd_sections(args: argparse.Namespace) -> int:
    """``proto-tools sections <tool>``."""
    sections = ToolRegistry.get_readme_sections(args.tool)
    if args.json:
        print(_dump_json(sections))
    else:
        print(f"# {sections.title}\n")
        print("## Overview\n")
        print(sections.overview)
        print("\n## Background\n")
        print(sections.background)
        if sections.toolkit_notes:
            print("\n## Toolkit Notes\n")
            print(sections.toolkit_notes)
        print(f"\n(tools registered: {', '.join(t.key for t in sections.tools)})")
    return 0


def _cmd_model_doc(args: argparse.Namespace, kind: str) -> int:
    """Shared handler for ``input`` / ``config`` / ``output`` verbs."""
    getter = {
        "input": ToolRegistry.get_input_doc,
        "config": ToolRegistry.get_config_doc,
        "output": ToolRegistry.get_output_doc,
    }[kind]
    doc = getter(args.tool)
    if args.json:
        print(_dump_json(doc))
        return 0

    print(f"{kind.capitalize()}: {doc.name}\n")
    if doc.docstring:
        print(doc.docstring)
        print()
    for f in doc.fields:
        marker = "required" if f.required else f"default={f.default!r}"
        print(f"  {f.name:24s}  {f.type_str:30s}  ({marker})")
        # Prefer the full docstring text; fall back to the terse field description.
        body = f.doc or f.description
        if body:
            print(f"  {'':24s}  {body.replace(chr(10), chr(10) + ' ' * 28)}")

    if doc.metric_specs:
        scope = f" (per {doc.metrics_per_item_field} item)" if doc.metrics_per_item_field else ""
        print(f"\nMetrics{scope}:")
        for m in doc.metric_specs:
            lo = m.min if m.min is not None else "-inf"
            hi = m.max if m.max is not None else "inf"
            bits = [m.type_str or "?", f"range [{lo}, {hi}]"]
            if m.unit:
                bits.append(m.unit)
            if m.availability:
                bits.append(m.availability)
            if m.better_values_are:
                bits.append(f"better={m.better_values_are}")
            star = "  *primary" if m.is_primary else ""
            print(f"  {m.name:24s}  {', '.join(bits)}{star}")
            if m.description:
                print(f"  {'':24s}  {m.description}")
    return 0


def _signature_payload(spec: ToolSpec) -> dict[str, Any]:
    """Collect the symbol names and import modules that make up a tool's call surface."""
    return {
        "key": spec.key,
        "run_function": spec.function.__name__,
        "input_class": spec.input_model.__name__,
        "config_class": spec.config_model.__name__,
        "output_class": spec.output_model.__name__,
        "modules": {
            "run_function": spec.function.__module__,
            "input_class": spec.input_model.__module__,
            "config_class": spec.config_model.__module__,
            "output_class": spec.output_model.__module__,
        },
        "required_input_fields": [name for name, field in spec.input_model.model_fields.items() if field.is_required()],
    }


def _render_signature(spec: ToolSpec) -> str:
    """Render a tool's call surface: imports, symbol names, and required input fields.

    Everything here is fixed-size, so surveying tools costs the same whether the tool
    takes a peptide or a 524,288 bp window. ``example --as-python`` carries real values and
    scales with them; this does not, which is what makes it the cheap discovery path.

    ``Output`` is named but not imported: callers never construct one, so importing it
    would leave an unused name in pasted code, and for 24 of the registered tools it
    lives in a different module from the ``Input`` anyway.

    Args:
        spec (ToolSpec): The resolved tool.

    Returns:
        str: Python source sketching the call, with ``...`` in place of argument values.
    """
    run_fn = spec.function.__name__
    input_cls = spec.input_model.__name__
    config_cls = spec.config_model.__name__

    by_module: dict[str, set[str]] = {}
    for symbol, module in (
        (input_cls, spec.input_model.__module__),
        (config_cls, spec.config_model.__module__),
        (run_fn, spec.function.__module__),
    ):
        by_module.setdefault(module, set()).add(symbol)
    imports = "\n".join(
        f"from {module} import (\n" + "".join(f"    {s},\n" for s in sorted(symbols)) + ")"
        for module, symbols in sorted(by_module.items())
    )

    required = [name for name, field in spec.input_model.model_fields.items() if field.is_required()]
    args = ", ".join(f"{name}=..." for name in required)
    call = (
        f"result = {run_fn}(\n"
        f"    {input_cls}({args}),\n"
        f"    {config_cls}(),  # optional, omit for defaults\n"
        f")  # -> {spec.output_model.__name__}"
    )

    hint = f"\n\n# Field-level docs: proto-tools input|config|output {spec.key}"
    return f"{imports}\n\n{call}{hint}\n"


def _cmd_signature(args: argparse.Namespace) -> int:
    """``proto-tools signature <tool> [--json]``."""
    from proto_tools.utils.tool_docs import _normalize_tool_key

    spec = ToolRegistry.get(_normalize_tool_key(args.tool))
    if args.json:
        print(_dump_json(_signature_payload(spec)))
        return 0
    print(_render_signature(spec), end="")
    return 0


def _render_example_as_python(spec: ToolSpec, example: BaseModel) -> str:
    """Render a runnable call to ``spec``, with the symbol names spelled out.

    Model and run-function names are derived from the toolkit, not the registry key, so
    they are not reliably guessable from the key an agent resolved the tool by. Emitting
    them removes the guess. Imports come from each symbol's defining module rather than
    the toolkit package, so they do not depend on ``__init__`` re-exports.

    Args:
        spec (ToolSpec): The resolved tool.
        example (BaseModel): The tool's example input instance.

    Returns:
        str: Python source that constructs the input and calls the run function.
    """
    run_fn = spec.function.__name__
    input_cls = spec.input_model.__name__
    module = spec.input_model.__module__

    symbols = sorted({input_cls, run_fn})
    if spec.function.__module__ == module:
        imports = f"from {module} import (\n" + "".join(f"    {s},\n" for s in symbols) + ")"
    else:
        imports = f"from {module} import {input_cls}\nfrom {spec.function.__module__} import {run_fn}"

    fields = example.model_dump(mode="json", exclude_defaults=True) or example.model_dump(mode="json")
    kwargs = "".join(f"        {name}={value!r},\n" for name, value in fields.items())
    call = f"result = {run_fn}(\n    {input_cls}(\n{kwargs}    ),\n)"

    config_cls = getattr(spec.config_model, "__name__", None)
    hint = f"\n\n# Optional: pass {config_cls}(...) as the second argument to override defaults." if config_cls else ""
    return f"{imports}\n\n{call}{hint}\n"


def _cmd_notebook(args: argparse.Namespace) -> int:
    """``proto-tools notebook <tool>``."""
    rendered = ToolRegistry.get_example_notebook(args.tool)
    if rendered is None:
        print(f"No example notebook found for '{args.tool}'.", file=sys.stderr)
        return 1
    print(rendered, end="")
    return 0


def _cmd_agent_context(_args: argparse.Namespace) -> int:
    """``proto-tools agent-context`` — usage primer for coding agents."""
    primer = Path(__file__).parent / "agent_context.md"
    print(primer.read_text(), end="")
    return 0


# =============================================================================
# Argparse wiring
# =============================================================================


def _cmd_eject_standalone(args: argparse.Namespace) -> int:
    """Copy a tool's standalone env-def dir into the working tree for overriding."""
    dest = ToolInstance.eject_standalone(args.tool, Path(args.dir))
    # dest.name is the normalized toolkit (folder name), which may differ from
    # args.tool (e.g. a tool key); the override var is keyed on the folder name.
    var = f"PROTO_{dest.name.upper().replace('-', '_')}_STANDALONE_DIR"
    print(f"Copied {dest.name} standalone env definition to {dest}")
    print("Edit setup.sh (and the other files) there, then point proto-tools at it:")
    print(f"  export {var}={dest}")
    return 0


def _cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy tools into a Modal workspace you own."""
    from proto_tools.modal.deploy import main as deploy_main

    return deploy_main(args.rest, prog="proto-tools deploy")


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Run the MCP server over stdio."""
    try:
        from proto_tools.mcp import main as mcp_main
    except ImportError as exc:
        # fastmcp ships in the 'mcp' extra, so a plain install reaches here.
        print(f"error: the MCP server needs the 'mcp' extra ({exc}).", file=sys.stderr)
        print('Install it with: pip install "proto-tools[mcp]"', file=sys.stderr)
        return 2

    mcp_main(args.rest)
    return 0


_CREDENTIAL_REMEDY = (
    "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (these work in a container, CI job, or agent "
    "sandbox), point MODAL_CONFIG_PATH at a token file, or run `modal token new` on a machine "
    "you work on directly."
)


def _modal_auth_check() -> tuple[str, str | None]:
    """Return how Modal authenticated, and the remedy when it did not.

    Calls the server rather than stopping at ``from_env``, which only checks that a token is
    present. A revoked or mistyped token passes that check and fails every real call, so the
    two are reported apart: they need different fixes.
    """
    from proto_tools.utils import modal_status as status

    try:
        import modal
        from modal.exception import AuthError
    except ImportError as exc:
        return f"unavailable — {exc}", "Reinstall proto-tools: Modal is a required dependency."

    try:
        client = modal.Client.from_env()
    except Exception as exc:
        checked = f"checked {'/'.join(status.TOKEN_VARS)}, {status.config_path()} ({status.config_state()})"
        return f"not found — {checked}", f"{_CREDENTIAL_REMEDY} Modal reported: {type(exc).__name__}: {exc}"

    mechanism = status.auth_mechanism() or "a source the SDK resolved itself"
    try:
        client.hello()
    except AuthError as exc:
        return (
            f"rejected — a credential was found via {mechanism}, and Modal refused it",
            f"The token exists but is not valid: {exc}. It may have been revoked, or belong to "
            f"another workspace. {_CREDENTIAL_REMEDY}",
        )
    except Exception as exc:
        # Reaching Modal at all is a separate failure from the credential being wrong.
        return (
            f"unverified — found via {mechanism}, but Modal was unreachable",
            f"Could not reach Modal to check the credential: {type(exc).__name__}: {exc}",
        )

    return f"OK via {mechanism}", None


def _proto_home_check() -> tuple[str, str | None]:
    """Return the PROTO_HOME line, and the remedy when it cannot be written to.

    A directory that does not exist yet is not a fault: it is created on first use. What
    matters is whether the nearest existing ancestor can be written to.
    """
    from proto_tools.utils.proto_home import get_proto_home

    home = get_proto_home()
    existing = next((p for p in (home, *home.parents) if p.exists()), home)
    if not os.access(existing, os.W_OK):
        return (
            f"{home}   writable: no ({existing} is read-only)",
            f"Grant write access to {existing}, or point PROTO_HOME at a writable directory.",
        )
    suffix = "writable: yes" if existing == home else "writable: yes (created on first use)"
    return f"{home}   {suffix}", None


def _temp_space_check() -> tuple[str, str | None]:
    """Return the temp-space line, and the remedy when a build directory cannot be made.

    Tool environments build under the temp directory, so a sandbox that confines it stops
    every local build before it starts.
    """
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "probe").write_text("probe")
    except OSError as exc:
        return (
            f"cannot write to {tempfile.gettempdir()}",
            f"Point TMPDIR at a writable directory: {type(exc).__name__}: {exc}",
        )
    return f"OK ({tempfile.gettempdir()})", None


def _mcp_extra_check() -> tuple[str, str | None]:
    """Return the MCP extra line. Its absence is reported, not treated as a fault."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        import fastmcp  # noqa: F401

        return f"installed (fastmcp {version('fastmcp')})", None
    except (ImportError, PackageNotFoundError):
        return 'not installed — `pip install "proto-tools[mcp]"` to run the MCP server', None


def _workspace_lines() -> dict[str, str]:
    """Describe the workspace calls land in. Only meaningful once Modal has authenticated."""
    import modal

    from proto_tools.modal.app import resolve_environment
    from proto_tools.modal.manifest import APP_BUCKETS
    from proto_tools.utils.modal_status import deployed_apps

    deployed = sorted(deployed_apps())
    lines = {
        "workspace": getattr(modal.config, "_profile", None) or "(unknown)",
        "environment": resolve_environment(),
        "apps deployed": f"{len(deployed)} of {len(APP_BUCKETS)}",
    }
    if deployed:
        lines["apps deployed"] += f"   ({', '.join(deployed)})"
    return lines


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report whether this environment can run tools, and name the fix for whatever cannot."""
    lines: dict[str, str] = {}
    remedies: list[str] = []

    for label, check in (
        ("modal auth", _modal_auth_check),
        ("PROTO_HOME", _proto_home_check),
        ("temp space", _temp_space_check),
        ("mcp extra", _mcp_extra_check),
    ):
        lines[label], remedy = check()
        if remedy:
            remedies.append(f"{label}: {remedy}")

    # Only ask where calls land once the credential is known good; anything else would
    # report a workspace this machine cannot actually reach.
    if not any(r.startswith("modal auth:") for r in remedies):
        lines.update(_workspace_lines())

    if args.json:
        print(_dump_json({"checks": lines, "remedies": remedies}))
        return 1 if remedies else 0

    order = ["modal auth", "workspace", "environment", "apps deployed", "PROTO_HOME", "temp space", "mcp extra"]
    for label in order:
        if label in lines:
            print(f"{label:<16}: {lines[label]}")
    sys.stdout.flush()  # keep the remedies below the report when the two streams are captured
    for remedy in remedies:
        print(f"\n{remedy}", file=sys.stderr)
    return 1 if remedies else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proto-tools",
        description="Discover, inspect and run proto-tools registered tools. "
        "list, search, schema, example, info, run and workspace mirror the MCP "
        "server's tools and print the same payloads.",
        epilog="Coding agents: run `proto-tools agent-context` first for a usage primer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    def add_device(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--device",
            choices=("local", "modal", "proto"),
            default="local",
            help="Backend to answer for: this machine (default), your Modal workspace, or Proto.",
        )

    p_agent = sub.add_parser(
        "agent-context",
        help="Print a usage primer for coding agents (start here).",
    )
    p_agent.set_defaults(func=_cmd_agent_context)

    p_list = sub.add_parser("list", help="List tools that can run on the device (MCP list_tools).")
    p_list.add_argument("--category", help="Filter to a category, e.g. 'masked_models'.")
    p_list.add_argument("--all", action="store_true", help="Include tools the device cannot run yet.")
    add_device(p_list)
    p_list.add_argument("--json", action="store_true", help="Emit the MCP payload instead of text.")
    p_list.set_defaults(func=_cmd_list)

    p_search = sub.add_parser("search", help="Find tools by keyword, best match first (MCP search_tools).")
    p_search.add_argument("query", help='Natural-language query, e.g. "fold a protein".')
    p_search.add_argument("--limit", type=int, default=10, help="Maximum results (default: 10).")
    p_search.add_argument("--all", action="store_true", help="Include tools the device cannot run yet.")
    add_device(p_search)
    p_search.add_argument("--json", action="store_true", help="Emit the MCP payload instead of text.")
    p_search.set_defaults(func=_cmd_search)

    p_schema = sub.add_parser("schema", help="Input, config and output JSON Schemas (MCP get_tool_schema).")
    p_schema.add_argument("tool")
    p_schema.set_defaults(func=_cmd_schema)

    p_example = sub.add_parser("example", help="The tool's example input (MCP get_tool_example).")
    p_example.add_argument("tool")
    p_example.add_argument(
        "--as-python",
        action="store_true",
        help="Emit a runnable snippet with the correct import and symbol names instead of JSON.",
    )
    p_example.set_defaults(func=_cmd_example)

    p_info = sub.add_parser(
        "info", help="Provenance: links, citation, DOI, license, weights access (MCP get_tool_info)."
    )
    p_info.add_argument("tool")
    p_info.set_defaults(func=_cmd_info)

    p_run = sub.add_parser("run", help="Run a tool and print its result (MCP run_tool).")
    p_run.add_argument("tool")
    p_run.add_argument("--inputs", help="Input fields as a JSON object, or @path to a JSON file.")
    p_run.add_argument("--config", help="Config fields as a JSON object, or @path to a JSON file.")
    p_run.add_argument("--example", action="store_true", help="Run the tool's example input.")
    p_run.add_argument("--output-dir", help="Where oversized result fields are written.")
    add_device(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_workspace = sub.add_parser("workspace", help="Where calls land on the device (MCP workspace_info).")
    add_device(p_workspace)
    p_workspace.set_defaults(func=_cmd_workspace)

    p_cat_list = sub.add_parser("categories", help="List all categories.")
    p_cat_list.add_argument("--json", action="store_true")
    p_cat_list.set_defaults(func=_cmd_categories)

    p_catalog = sub.add_parser("catalog", help="Tools grouped by category.")
    p_catalog.add_argument("--json", action="store_true")
    p_catalog.set_defaults(func=_cmd_catalog)

    p_docs = sub.add_parser(
        "docs",
        help="Per-tool docs (intro + applications + usage tips + toolkit notes + license).",
    )
    p_docs.add_argument("tool", help="Tool identifier (registry key, run-function name, etc.).")
    p_docs.add_argument(
        "--no-toolkit-notes",
        action="store_true",
        help="Omit the toolkit-wide Toolkit Notes from the output.",
    )
    p_docs.add_argument(
        "--no-license",
        action="store_true",
        help="Omit the parsed license.yaml from the output.",
    )
    p_docs.add_argument("--json", action="store_true")
    p_docs.set_defaults(func=_cmd_docs)

    p_eject = sub.add_parser(
        "eject-standalone",
        help="Copy a tool's standalone env-def dir into your working tree to edit and override it.",
    )
    p_eject.add_argument("tool", help="Tool identifier (toolkit name, registry key, run-function name, ...).")
    p_eject.add_argument(
        "--dir",
        default="./proto_standalone",
        help="Destination root; the copy lands in <dir>/<toolkit>/ (default: ./proto_standalone).",
    )
    p_eject.set_defaults(func=_cmd_eject_standalone)

    p_readme = sub.add_parser("readme", help="Full README text for the tool's toolkit.")
    p_readme.add_argument("tool")
    p_readme.set_defaults(func=_cmd_readme)

    p_section = sub.add_parser("section", help="One named H2 section from the README.")
    p_section.add_argument("tool")
    p_section.add_argument("heading", help='Exact heading text, e.g. "Background".')
    p_section.set_defaults(func=_cmd_section)

    p_sections = sub.add_parser("sections", help="Structured view of the whole README.")
    p_sections.add_argument("tool")
    p_sections.add_argument("--json", action="store_true")
    p_sections.set_defaults(func=_cmd_sections)

    for kind in ("input", "config", "output"):
        p = sub.add_parser(kind, help=f"Pydantic {kind}-model docs.")
        p.add_argument("tool")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=lambda a, k=kind: _cmd_model_doc(a, k))

    p_signature = sub.add_parser(
        "signature",
        help="Imports, symbol names, and required input fields for the tool's call. No example payload.",
    )
    p_signature.add_argument("tool")
    p_signature.add_argument("--json", action="store_true", help="Emit the symbol names as JSON.")
    p_signature.set_defaults(func=_cmd_signature)

    p_notebook = sub.add_parser(
        "notebook",
        help="Toolkit example notebook rendered as markdown + fenced code (outputs stripped).",
    )
    p_notebook.add_argument("tool")
    p_notebook.set_defaults(func=_cmd_notebook)

    p_deploy = sub.add_parser(
        "deploy",
        help="Deploy tools into your own Modal workspace, and smoke-test them.",
    )
    p_deploy.set_defaults(func=_cmd_deploy)

    p_mcp = sub.add_parser("mcp", help="Run the MCP server over stdio (needs the 'mcp' extra).")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_doctor = sub.add_parser(
        "doctor",
        help="Check this environment can reach Modal and build tools; exits non-zero with a remedy.",
    )
    p_doctor.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


_PASSTHROUGH = {"deploy": _cmd_deploy, "mcp": _cmd_mcp}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args_in = sys.argv[1:] if argv is None else argv
    # These own their whole option surface, which argparse would try to claim first.
    if args_in and args_in[0] in _PASSTHROUGH:
        return _PASSTHROUGH[args_in[0]](argparse.Namespace(rest=args_in[1:]))

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyError as exc:
        print(f"error: tool not registered: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError, DeviceUnavailableError) as exc:
        # Identifier-resolution failures, malformed --inputs, unreadable @files, a device
        # without its credentials, an existing eject destination.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
