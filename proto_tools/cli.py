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
import sys
from typing import Any

from pydantic import BaseModel

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec

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
        if sections.qc_pending:
            print("(NOTE: README still has the QC-pending TODO callout.)")
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
        if f.description:
            print(f"  {'':24s}  {f.description}")

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


def _cmd_example_input(args: argparse.Namespace) -> int:
    """``proto-tools example-input <tool>``."""
    example = ToolRegistry.get_example_input(args.tool)
    if example is None:
        print(f"No example input defined for '{args.tool}'.", file=sys.stderr)
        return 1
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


# =============================================================================
# Argparse wiring
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proto-tools",
        description="Discover and inspect proto-tools registered tools. "
        "Every verb maps to a `ToolRegistry` classmethod; pass --json on "
        "verbs that return structured data for machine-readable output.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

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

    p_example_input = sub.add_parser("example-input", help="A minimal valid Input for the tool.")
    p_example_input.add_argument("tool")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
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


if __name__ == "__main__":
    sys.exit(main())
