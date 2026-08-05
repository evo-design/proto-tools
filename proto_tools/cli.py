"""Command-line entry point for proto-tools discovery + structured docs.

Reachable as the ``proto-tools`` shell command after ``pip install``, or as
``python -m proto_tools`` without one. Every verb maps one-to-one to a
``ToolRegistry`` classmethod so the CLI surface stays in sync with the
in-process API.

Defaults to human-readable text output (so a developer can pipe a tool's
docs into ``less`` without parsing JSON). Every verb that returns structured
data accepts ``--json`` for machine-readable output suitable for agents or
MCP servers calling the CLI via subprocess.
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


def _spec_summary(spec: ToolSpec) -> str:
    """One-line text summary of a ToolSpec for list-style output."""
    gpu = " (GPU)" if spec.uses_gpu else ""
    return f"{spec.key:40s}  [{spec.category}]{gpu}  {spec.description}"


# =============================================================================
# Verb handlers
# =============================================================================


def _cmd_list(args: argparse.Namespace) -> int:
    """``proto-tools list [--category C] [--gpu] [--cpu]``."""
    if args.category:
        specs = ToolRegistry.list_by_category(args.category)
    elif args.gpu:
        specs = sorted(ToolRegistry.list_gpu_tools(), key=lambda s: s.key)
    elif args.cpu:
        specs = sorted(ToolRegistry.list_cpu_tools(), key=lambda s: s.key)
    else:
        specs = sorted(ToolRegistry.list_all(), key=lambda s: s.key)

    if args.json:
        print(_dump_json(specs))
    else:
        for spec in specs:
            print(_spec_summary(spec))
    return 0


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


def _cmd_schema(args: argparse.Namespace) -> int:
    """``proto-tools schema <tool> [--input|--config|--output]``."""
    if args.input:
        payload = ToolRegistry.get_input_schema(args.tool)
    elif args.config:
        payload = ToolRegistry.get_config_schema(args.tool)
    elif args.output:
        payload = ToolRegistry.get_output_schema(args.tool)
    else:
        payload = ToolRegistry.get_schemas(args.tool)
    print(json.dumps(payload, indent=2, default=str))
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
    takes a peptide or a 524,288 bp window. ``example-input`` carries real values and
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


def _cmd_example_input(args: argparse.Namespace) -> int:
    """``proto-tools example-input <tool> [--as-python]``."""
    example = ToolRegistry.get_example_input(args.tool)
    if example is None:
        print(f"No example input defined for '{args.tool}'.", file=sys.stderr)
        return 1
    if args.as_python:
        print(_render_example_as_python(ToolRegistry.get(args.tool), example), end="")
        return 0
    print(_dump_json(example))
    return 0


def _cmd_example(args: argparse.Namespace) -> int:
    """``proto-tools example <tool>``."""
    rendered = ToolRegistry.get_example_notebook(args.tool)
    if rendered is None:
        print(f"No example notebook found for '{args.tool}'.", file=sys.stderr)
        return 1
    print(rendered, end="")
    return 0


def _cmd_citation(args: argparse.Namespace) -> int:
    """``proto-tools citation <tool>``."""
    cite = ToolRegistry.get_citation(args.tool)
    if cite is None:
        print(f"No citation registered for '{args.tool}'.", file=sys.stderr)
        return 1
    print(cite)
    return 0


def _cmd_links(args: argparse.Namespace) -> int:
    """``proto-tools links <tool>``."""
    links = ToolRegistry.get_links(args.tool)
    if links is None:
        print(f"No links registered for '{args.tool}'.", file=sys.stderr)
        return 1
    if args.json:
        print(_dump_json(links))
    else:
        for k, value in links.items():
            # links.yaml values may be str or list at runtime; widen for the list branch.
            v: object = value
            display = ", ".join(v) if isinstance(v, list) else str(v)
            print(f"{k:16s}  {display}")
    return 0


def _cmd_license(args: argparse.Namespace) -> int:
    """``proto-tools license <tool>``."""
    lic = ToolRegistry.get_license(args.tool)
    if lic is None:
        print(f"No license registered for '{args.tool}'.", file=sys.stderr)
        return 1
    print(_dump_json(lic))
    return 0


def _cmd_access(args: argparse.Namespace) -> int:
    """``proto-tools access <tool>`` — open | hf-gated | request."""
    print(ToolRegistry.get_weights_access(args.tool))
    return 0


def _cmd_doi(args: argparse.Namespace) -> int:
    """``proto-tools doi <tool>``."""
    doi = ToolRegistry.get_doi(args.tool)
    if doi is None:
        print(f"No DOI registered for '{args.tool}'.", file=sys.stderr)
        return 1
    print(doi)
    return 0


def _cmd_url(args: argparse.Namespace) -> int:
    """``proto-tools url <tool>``."""
    url = ToolRegistry.get_docs_url(args.tool)
    if url is None:
        print(f"No docs URL resolvable for '{args.tool}'.", file=sys.stderr)
        return 1
    print(url)
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
        description="Discover and inspect proto-tools registered tools. "
        "Every verb maps to a `ToolRegistry` classmethod; pass --json on "
        "verbs that return structured data for machine-readable output.",
        epilog="Coding agents: run `proto-tools agent-context` first for a usage primer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p_agent = sub.add_parser(
        "agent-context",
        help="Print a usage primer for coding agents (start here).",
    )
    p_agent.set_defaults(func=_cmd_agent_context)

    p_list = sub.add_parser("list", help="List registered tools.")
    filt = p_list.add_mutually_exclusive_group()
    filt.add_argument("--category", help="Filter to a category, e.g. 'masked_models'.")
    filt.add_argument("--gpu", action="store_true", help="Only tools that require a GPU.")
    filt.add_argument("--cpu", action="store_true", help="Only tools that do not require a GPU.")
    p_list.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    p_list.set_defaults(func=_cmd_list)

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

    p_schema = sub.add_parser("schema", help="JSON Schema(s) for the tool.")
    p_schema.add_argument("tool")
    p_schema_g = p_schema.add_mutually_exclusive_group()
    p_schema_g.add_argument("--input", action="store_true")
    p_schema_g.add_argument("--config", action="store_true")
    p_schema_g.add_argument("--output", action="store_true")
    p_schema.set_defaults(func=_cmd_schema)

    p_signature = sub.add_parser(
        "signature",
        help="Imports, symbol names, and required input fields for the tool's call. No example payload.",
    )
    p_signature.add_argument("tool")
    p_signature.add_argument("--json", action="store_true", help="Emit the symbol names as JSON.")
    p_signature.set_defaults(func=_cmd_signature)

    p_example_input = sub.add_parser("example-input", help="A minimal valid Input for the tool.")
    p_example_input.add_argument("tool")
    p_example_input.add_argument(
        "--as-python",
        action="store_true",
        help="Emit a runnable snippet with the correct import and symbol names instead of JSON.",
    )
    p_example_input.set_defaults(func=_cmd_example_input)

    p_example = sub.add_parser(
        "example",
        help="Toolkit example notebook rendered as markdown + fenced code (outputs stripped).",
    )
    p_example.add_argument("tool")
    p_example.set_defaults(func=_cmd_example)

    p_cite = sub.add_parser("citation", help="BibTeX citation, if registered.")
    p_cite.add_argument("tool")
    p_cite.set_defaults(func=_cmd_citation)

    p_links = sub.add_parser("links", help="GitHub / HuggingFace / etc. links from links.yaml.")
    p_links.add_argument("tool")
    p_links.add_argument("--json", action="store_true")
    p_links.set_defaults(func=_cmd_links)

    p_license = sub.add_parser("license", help="Parsed license.yaml.")
    p_license.add_argument("tool")
    p_license.set_defaults(func=_cmd_license)

    p_access = sub.add_parser(
        "access",
        help="Model-weights access: open | hf-gated | request.",
    )
    p_access.add_argument("tool")
    p_access.set_defaults(func=_cmd_access)

    p_doi = sub.add_parser("doi", help="DOI for the tool's primary citation, if any.")
    p_doi.add_argument("tool")
    p_doi.set_defaults(func=_cmd_doi)

    p_url = sub.add_parser("url", help="Public docs URL for the tool's page.")
    p_url.add_argument("tool")
    p_url.set_defaults(func=_cmd_url)

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
    except ValueError as exc:
        # Identifier-resolution failures, ambiguous toolkit names, etc.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: tool not registered: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
