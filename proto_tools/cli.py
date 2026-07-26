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
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils import remote_execution as remote_exec
from proto_tools.utils.tool_instance import ToolInstance

logger = logging.getLogger(__name__)


# =============================================================================
# Output helpers
# =============================================================================


def _dump_json(value: Any) -> str:
    """Render any value as pretty-printed JSON."""
    if isinstance(value, BaseModel):
        return str(value.model_dump_json(indent=2))
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


def _cmd_agent_context(_args: argparse.Namespace) -> int:
    """``proto-tools agent-context`` — usage primer for coding agents."""
    primer = Path(__file__).parent / "agent_context.md"
    print(primer.read_text(), end="")
    return 0


def _remote_nodes_from_args(args: argparse.Namespace) -> list[remote_exec.RemoteNode]:
    """Load and optionally filter remote nodes for CLI commands."""
    nodes = remote_exec.load_remote_nodes_json(args.profile)
    node_names = getattr(args, "node", None)
    if not node_names:
        return nodes
    requested = set(node_names)
    filtered = [node for node in nodes if node.name in requested]
    missing = sorted(requested - {node.name for node in filtered})
    if missing:
        raise ValueError(f"Remote node(s) not found in {args.profile}: {', '.join(missing)}")
    return filtered


def _existing_remote_command_node_from_args(args: argparse.Namespace) -> remote_exec.ExistingRemoteCommandNode:
    """Load one existing-command node for direct wrapper execution."""
    nodes = remote_exec.load_existing_remote_command_nodes_json(args.profile)
    return remote_exec.select_existing_remote_command_node(nodes, args.node)


def _raise_for_remote_profile_errors(nodes: list[remote_exec.RemoteNode]) -> None:
    """Raise before SSH commands when a remote profile has static errors."""
    profile_errors = [issue for issue in remote_exec.validate_remote_nodes(nodes) if issue.level == "error"]
    if profile_errors:
        first = profile_errors[0]
        raise ValueError(f"Invalid remote profile: {first.node_name or '-'} {first.field}: {first.message}")


def _validated_remote_nodes_from_args(args: argparse.Namespace) -> list[remote_exec.RemoteNode]:
    """Load remote nodes and reject static profile errors."""
    nodes = _remote_nodes_from_args(args)
    _raise_for_remote_profile_errors(nodes)
    return nodes


def _parse_remote_scaffold_node(raw: str) -> tuple[str, str | None, str]:
    """Parse ``NAME[@HOST]=HOME`` for profile scaffolding."""
    if "=" not in raw:
        raise ValueError(f"Remote profile node must use NAME[@HOST]=HOME: {raw!r}")
    left, home_dir = raw.split("=", 1)
    if not left or not home_dir:
        raise ValueError(f"Remote profile node must use NAME[@HOST]=HOME: {raw!r}")
    if "@" in left:
        name, host = left.split("@", 1)
    else:
        name, host = left, None
    if not name or host == "":
        raise ValueError(f"Remote profile node must use NAME[@HOST]=HOME: {raw!r}")
    return name, host, home_dir


def _print_remote_mapping(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    """Print per-node remote command results."""
    if getattr(args, "json", False):
        print(_dump_json(payload))
        return 0
    for name, value in payload.items():
        rendered = json.dumps(value, sort_keys=True) if isinstance(value, dict | list) else str(value)
        print(f"{name}\t{rendered}")
    return 0


def _remote_timeout_kwargs(args: argparse.Namespace) -> dict[str, float]:
    """Return optional timeout kwargs without disturbing narrow test fakes."""
    timeout_sec = getattr(args, "timeout_sec", None)
    return {} if timeout_sec is None else {"timeout_sec": timeout_sec}


def _cmd_remote_profile_scaffold(args: argparse.Namespace) -> int:
    """``proto-tools remote profile scaffold``."""
    if not args.direct and not args.slurm:
        raise ValueError("remote profile scaffold requires at least one --direct or --slurm node")
    nodes: list[remote_exec.RemoteNode] = []
    for raw in args.direct:
        name, host, home_dir = _parse_remote_scaffold_node(raw)
        nodes.append(
            remote_exec.scaffold_remote_node(
                name,
                home_dir,
                host=host,
                scheduler="direct",
                repo_name=args.repo_name,
                work_name=args.work_name_template or args.work_name,
                bootstrap_python=args.bootstrap_python,
                weight=args.direct_weight,
                max_concurrent_buckets=args.direct_max_concurrent_buckets,
                worker_device=args.direct_worker_device,
            )
        )
    for raw in args.slurm:
        name, host, home_dir = _parse_remote_scaffold_node(raw)
        nodes.append(
            remote_exec.scaffold_remote_node(
                name,
                home_dir,
                host=host,
                scheduler="slurm",
                repo_name=args.repo_name,
                work_name=args.work_name_template or args.work_name,
                bootstrap_python=args.bootstrap_python,
                weight=args.slurm_weight,
                max_concurrent_buckets=args.slurm_max_concurrent_buckets,
                max_active_slurm_jobs=args.slurm_max_active_jobs,
                worker_device=args.slurm_worker_device,
                sbatch_args=tuple(args.slurm_sbatch_arg),
            )
        )

    issues = remote_exec.validate_remote_nodes(nodes)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        first = errors[0]
        raise ValueError(f"generated remote profile is invalid: {first.node_name or '-'} {first.field}: {first.message}")

    remote_exec.write_remote_nodes_json(nodes, args.output, overwrite=args.overwrite)
    summary = {
        "output": str(args.output),
        "nodes": [node.to_dict() for node in nodes],
        "issues": [issue.to_dict() for issue in issues],
    }
    if args.json:
        print(_dump_json(summary))
    else:
        print(f"profile\t{args.output}")
        for node in nodes:
            print(f"node\t{node.name}\t{node.host}\t{node.scheduler}\twork_dir={node.work_dir}")
        for issue in issues:
            print(f"{issue.level}\t{issue.node_name or '-'}\t{issue.field}\t{issue.message}")
    return 0


def _cmd_remote_profile_list(args: argparse.Namespace) -> int:
    """``proto-tools remote profile list``."""
    nodes = _remote_nodes_from_args(args)
    if args.json:
        print(_dump_json([node.to_dict() for node in nodes]))
        return 0
    for node in nodes:
        print(
            f"{node.name}\t{node.host}\t{node.scheduler}\t"
            f"weight={node.weight}\tmax_concurrent_buckets={node.max_concurrent_buckets}"
        )
    return 0


def _cmd_remote_profile_validate(args: argparse.Namespace) -> int:
    """``proto-tools remote profile validate``."""
    nodes = _remote_nodes_from_args(args)
    issues = remote_exec.validate_remote_nodes(nodes)
    has_errors = any(issue.level == "error" for issue in issues)
    payload = {
        "ok": not has_errors,
        "node_count": len(nodes),
        "issues": [issue.to_dict() for issue in issues],
    }
    if args.json:
        print(_dump_json(payload))
    elif not issues:
        print("ok")
    else:
        for issue in issues:
            print(f"{issue.level}\t{issue.node_name or '-'}\t{issue.field}\t{issue.message}")
    return 1 if has_errors else 0


def _cmd_remote_manager_start(args: argparse.Namespace) -> int:
    """``proto-tools remote manager-start``."""
    nodes = _validated_remote_nodes_from_args(args)
    client = remote_exec.RemoteManagerClient()
    timeout_kwargs = _remote_timeout_kwargs(args)
    payload = {
        node.name: client.start_node(
            node,
            poll_interval=args.poll_interval,
            slurm_group_size=args.slurm_group_size,
            **timeout_kwargs,
        )
        for node in nodes
    }
    return _print_remote_mapping(args, payload)


def _cmd_remote_manager_status(args: argparse.Namespace) -> int:
    """``proto-tools remote manager-status``."""
    nodes = _validated_remote_nodes_from_args(args)
    client = remote_exec.RemoteManagerClient()
    timeout_kwargs = _remote_timeout_kwargs(args)
    payload = {node.name: client.status_node(node, **timeout_kwargs) for node in nodes}
    return _print_remote_mapping(args, payload)


def _cmd_remote_manager_preflight(args: argparse.Namespace) -> int:
    """``proto-tools remote manager-preflight``."""
    nodes = _validated_remote_nodes_from_args(args)
    client = remote_exec.RemoteManagerClient()
    timeout_kwargs = _remote_timeout_kwargs(args)
    payload = {node.name: client.preflight_node(node, **timeout_kwargs) for node in nodes}
    return _print_remote_mapping(args, payload)


def _cmd_remote_manager_diagnostics(args: argparse.Namespace) -> int:
    """``proto-tools remote manager-diagnostics``."""
    nodes = _validated_remote_nodes_from_args(args)
    client = remote_exec.RemoteManagerClient()
    timeout_kwargs = _remote_timeout_kwargs(args)
    payload = {node.name: client.diagnostics_node(node, **timeout_kwargs) for node in nodes}
    return _print_remote_mapping(args, payload)


def _cmd_remote_smoke(args: argparse.Namespace) -> int:
    """``proto-tools remote smoke``."""
    report = remote_exec.smoke_remote_nodes(
        _remote_nodes_from_args(args),
        local_results_dir=args.local_results_dir,
        timeout_sec=args.timeout_sec,
    )
    if args.json:
        print(_dump_json(report.to_dict()))
    else:
        print("ok" if report.ok else "failed")
        for issue in report.profile_issues:
            print(f"{issue.level}\t{issue.node_name or '-'}\t{issue.field}\t{issue.message}")
        for name, payload in report.preflight.items():
            if not payload.get("ok"):
                print(f"preflight-failed\t{name}\t{json.dumps(payload, sort_keys=True)}")
        for name, status in report.manager_status.items():
            if status.startswith("check-failed:"):
                print(f"manager-status-failed\t{name}\t{status}")
        for name, payload in report.diagnostics.items():
            if isinstance(payload, dict) and payload.get("ok") is False:
                print(f"diagnostics-failed\t{name}\t{json.dumps(payload, sort_keys=True)}")
    return 0 if report.ok else 1


def _cmd_remote_existing_plan(args: argparse.Namespace) -> int:
    """``proto-tools remote existing plan``."""
    node = _existing_remote_command_node_from_args(args)
    plan = remote_exec.plan_existing_remote_command(
        node,
        local_input_path=args.input,
        run_id=args.run_id,
        local_collect_dir=args.local_collect_dir,
    )
    payload = plan.to_dict()
    if args.json:
        print(_dump_json(payload))
    else:
        for step, command in payload["commands"].items():
            print(f"existing-command\t{plan.node_name}\t{step}\t{shlex.join(command)}")
    return 0


def _cmd_remote_existing_run(args: argparse.Namespace) -> int:
    """``proto-tools remote existing run``."""
    node = _existing_remote_command_node_from_args(args)
    result = remote_exec.run_existing_remote_command(
        node,
        local_input_path=args.input,
        run_id=args.run_id,
        local_collect_dir=args.local_collect_dir,
        timeout_sec=args.timeout_sec,
    )
    if args.json:
        print(_dump_json(result.to_dict()))
    else:
        print(f"completed\t{result.plan.node_name}\t{result.plan.run_id}")
        for path in result.collected_files:
            print(f"collected\t{path}")
    return 0


def _cmd_remote_existing_launch(args: argparse.Namespace) -> int:
    """``proto-tools remote existing launch``."""
    nodes = remote_exec.load_existing_remote_batch_nodes_json(args.profile)
    result = remote_exec.launch_existing_remote_command_batch(
        nodes,
        input_fasta=args.input,
        software=args.software,
        run_dir=args.run_dir,
        bucket_size=args.bucket_size,
        slurm_group_size=args.slurm_group_size,
        wait_poll_interval=args.wait_poll_interval,
        timeout_sec=args.timeout_sec,
        remote_check_timeout_sec=args.remote_check_timeout_sec,
    )
    if args.json:
        print(_dump_json(result.to_dict()))
    else:
        print(f"completed\t{result.run_id}\t{result.task_count}")
        for node_name, count in result.assignment.items():
            print(f"assigned\t{node_name}\t{count}")
        print(f"summary\t{result.summary}")
    return 0


def _cmd_remote_compute(args: argparse.Namespace) -> int:
    """``proto-tools remote compute``."""
    config = remote_exec.resolve_remote_compute_config(args.software, registry_path=args.registry)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise ValueError(f"Remote compute input FASTA not found: {input_path}")
    run_dir = Path(args.run_dir).expanduser().resolve()
    nodes = remote_exec.load_existing_remote_batch_nodes_json(config.profile_path)
    result = remote_exec.launch_existing_remote_command_batch(
        nodes,
        input_fasta=input_path,
        software=config.software,
        run_dir=run_dir,
        bucket_size=config.bucket_size if args.bucket_size is None else args.bucket_size,
        slurm_group_size=config.slurm_group_size if args.slurm_group_size is None else args.slurm_group_size,
        wait_poll_interval=(
            config.wait_poll_interval if args.wait_poll_interval is None else args.wait_poll_interval
        ),
        timeout_sec=config.timeout_sec if args.timeout_sec is None else args.timeout_sec,
        remote_check_timeout_sec=(
            config.remote_check_timeout_sec
            if args.remote_check_timeout_sec is None
            else args.remote_check_timeout_sec
        ),
    )
    payload = {
        "software": config.software,
        "registry": str(config.registry_path),
        "profile": str(config.profile_path),
        "receipt": str(run_dir / "compute.json"),
        **result.to_dict(),
    }
    remote_exec.write_remote_compute_receipt(payload, run_dir / "compute.json")
    if args.json:
        print(_dump_json(payload))
    else:
        print(f"completed\t{result.run_id}\t{result.task_count}")
        print(f"software\t{config.software}")
        print(f"profile\t{config.profile_path}")
        print(f"receipt\t{run_dir / 'compute.json'}")
        for node_name, count in result.assignment.items():
            print(f"assigned\t{node_name}\t{count}")
        print(f"summary\t{result.summary}")
    return 0


def _cmd_remote_deploy_render(args: argparse.Namespace) -> int:
    """``proto-tools remote deploy render``."""
    nodes = _remote_nodes_from_args(args)
    report = remote_exec.build_deploy_render_report(
        nodes,
        args.local_repo_dir,
        allow_dirty_checkout=args.allow_dirty_checkout,
    )
    if args.json:
        print(_dump_json(report.to_dict()))
    else:
        for blocker in report.blockers:
            print(f"blocker\t{blocker}")
        for warning in report.warnings:
            print(f"warning\t{warning}")
        for action in report.actions:
            print(f"action\t{action}")
        for node in report.nodes:
            for command in node.commands:
                print(f"deploy-command\t{node.node_name}\t{shlex.join(list(command))}")
    return 0 if report.ok else 1


def _cmd_remote_deploy_apply(args: argparse.Namespace) -> int:
    """``proto-tools remote deploy apply``."""
    nodes = _remote_nodes_from_args(args)
    report = remote_exec.build_deploy_render_report(
        nodes,
        args.local_repo_dir,
        allow_dirty_checkout=args.allow_dirty_checkout,
    )
    blockers = list(report.blockers)
    shared_roots = set(remote_exec.shared_remote_install_roots(nodes))
    if args.allow_shared_install_root:
        blockers = [blocker for blocker in blockers if blocker not in shared_roots]
    if blockers:
        shared_root_prefix = (
            "shared install roots require --allow-shared-install-root; "
            if any(blocker in shared_roots for blocker in blockers)
            else ""
        )
        raise ValueError(
            "remote deploy apply preflight failed: " + shared_root_prefix + "; ".join(blockers)
        )
    client = remote_exec.RemoteDeploymentClient()
    payload = {}
    for node in nodes:
        results = client.deploy_node(node, args.local_repo_dir)
        payload[node.name] = {
            "command_count": len(results),
            "returncodes": [getattr(result, "returncode", 0) for result in results],
        }
    if args.json:
        print(_dump_json(payload))
    else:
        for name, summary in payload.items():
            print(f"deployed\t{name}\tcommand_count={summary['command_count']}")
    return 0


def _cmd_remote_rsync_once(args: argparse.Namespace) -> int:
    """``proto-tools remote rsync-once``."""
    nodes = _remote_nodes_from_args(args)
    for cmd in remote_exec.render_rsync_pull_commands(nodes, args.local_results_dir):
        subprocess.run(cmd, check=True)  # noqa: S603
    return 0


def _cmd_remote_rsync_start(args: argparse.Namespace) -> int:
    """``proto-tools remote rsync-start``."""
    stdout = remote_exec.LocalRsyncPullClient().start(
        _remote_nodes_from_args(args),
        args.local_results_dir,
        pid_path=args.pid_file,
        log_path=args.log_file,
        interval_sec=args.interval_sec,
    )
    print(stdout)
    return 0


def _cmd_remote_rsync_status(args: argparse.Namespace) -> int:
    """``proto-tools remote rsync-status``."""
    print(remote_exec.LocalRsyncPullClient().status(args.pid_file))
    return 0


def _cmd_remote_rsync_stop(args: argparse.Namespace) -> int:
    """``proto-tools remote rsync-stop``."""
    print(remote_exec.LocalRsyncPullClient().stop(args.pid_file))
    return 0


def _cmd_remote_run_plan(args: argparse.Namespace) -> int:
    """``proto-tools remote run plan``."""
    nodes = _validated_remote_nodes_from_args(args)
    inputs = json.loads(Path(args.input_json).read_text())
    config = None if args.config_json is None else json.loads(Path(args.config_json).read_text())
    local_results_dir = Path(args.local_results_dir).resolve()
    diagnostics: dict[str, Any] | None = None
    initial_node_loads: dict[str, float] | None = None
    if args.use_diagnostics_backlog:
        diagnostics, initial_node_loads = remote_exec.remote_run_diagnostics_backlog_loads(
            nodes,
            timeout_sec=args.remote_check_timeout_sec,
        )
    planner_kwargs: dict[str, Any] = {}
    if initial_node_loads is not None:
        planner_kwargs["initial_node_loads"] = initial_node_loads
    plan = remote_exec.RemoteDispatchPlanner(nodes, args.bucket_size, **planner_kwargs).build_plan(
        args.tool,
        inputs,
        config,
        run_id=args.run_id,
    )
    manifest = remote_exec.RemoteRunManifest.from_plan(
        plan,
        nodes=nodes,
        bucket_size=args.bucket_size,
        local_results_dir=local_results_dir,
    )
    remote_exec.write_remote_run_manifest(manifest, args.manifest, overwrite=args.overwrite)
    summary = {
        "run_id": manifest.run_id,
        "manifest": str(args.manifest),
        "bucket_count": len(manifest.requests),
        "local_results_dir": manifest.local_results_dir,
    }
    if diagnostics is not None:
        summary["diagnostics"] = diagnostics
    if initial_node_loads is not None:
        summary["initial_node_loads"] = initial_node_loads
    if args.json:
        print(_dump_json(summary))
    else:
        print(f"manifest\t{args.manifest}")
        print(f"run_id\t{manifest.run_id}")
        print(f"bucket_count\t{len(manifest.requests)}")
        if initial_node_loads is not None:
            print(f"initial_node_loads\t{json.dumps(initial_node_loads, sort_keys=True)}")
    return 0


def _cmd_remote_run_launch(args: argparse.Namespace) -> int:
    """``proto-tools remote run launch``."""
    nodes = _remote_nodes_from_args(args)
    inputs = json.loads(Path(args.input_json).read_text())
    config = None if args.config_json is None else json.loads(Path(args.config_json).read_text())
    result = remote_exec.launch_remote_run(
        args.tool,
        nodes,
        inputs,
        config,
        bucket_size=args.bucket_size,
        local_results_dir=args.local_results_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        run_id=args.run_id,
        overwrite=args.overwrite,
        manager_poll_interval=args.manager_poll_interval,
        slurm_group_size=args.slurm_group_size,
        wait_poll_interval=args.wait_poll_interval,
        timeout_sec=args.timeout_sec,
        rsync_mode=args.rsync_mode,
        rsync_pid_path=args.rsync_pid_file,
        rsync_log_path=args.rsync_log_file,
        rsync_interval_sec=args.rsync_interval_sec,
        use_diagnostics_backlog=args.use_diagnostics_backlog,
        overwrite_output=args.overwrite_output,
        remote_check_timeout_sec=args.remote_check_timeout_sec,
    )
    if args.json:
        print(_dump_json(result.to_dict()))
    else:
        print(f"completed\t{result.run_id}")
        print(f"manifest\t{result.manifest_path}")
        print(f"output\t{result.output_path}")
        print(f"bucket_count\t{result.bucket_count}")
    return 0


def _cmd_remote_run_resume(args: argparse.Namespace) -> int:
    """``proto-tools remote run resume``."""
    manifest = remote_exec.load_remote_run_manifest(args.manifest)
    result = remote_exec.resume_remote_run_manifest(
        manifest,
        manifest_path=args.manifest,
        output_path=args.output,
        manager_poll_interval=args.manager_poll_interval,
        slurm_group_size=args.slurm_group_size,
        wait_poll_interval=args.wait_poll_interval,
        timeout_sec=args.timeout_sec,
        rsync_mode=args.rsync_mode,
        rsync_pid_path=args.rsync_pid_file,
        rsync_log_path=args.rsync_log_file,
        rsync_interval_sec=args.rsync_interval_sec,
        overwrite_output=args.overwrite_output,
        remote_check_timeout_sec=args.remote_check_timeout_sec,
    )
    if args.json:
        print(_dump_json(result.to_dict()))
    else:
        print(f"completed\t{result.run_id}")
        print(f"manifest\t{result.manifest_path}")
        print(f"output\t{result.output_path}")
        print(f"bucket_count\t{result.bucket_count}")
    return 0


def _cmd_remote_run_submit(args: argparse.Namespace) -> int:
    """``proto-tools remote run submit``."""
    manifest = remote_exec.load_remote_run_manifest(args.manifest)
    staged = remote_exec.submit_remote_run_manifest(manifest)
    summary = {"run_id": manifest.run_id, "bucket_count": len(staged)}
    if args.json:
        print(_dump_json(summary))
    else:
        print(f"submitted\t{manifest.run_id}\tbucket_count={len(staged)}")
    return 0


def _cmd_remote_run_wait(args: argparse.Namespace) -> int:
    """``proto-tools remote run wait``."""
    manifest = remote_exec.load_remote_run_manifest(args.manifest)
    remote_exec.wait_remote_run_manifest(
        manifest,
        poll_interval=args.poll_interval,
        timeout_sec=args.timeout_sec,
        pull_results=not args.no_pull,
        rsync_pid_path=args.rsync_pid_file,
        rsync_log_path=args.rsync_log_file,
    )
    summary = {"run_id": manifest.run_id, "status": "completed"}
    if args.json:
        print(_dump_json(summary))
    else:
        print(f"completed\t{manifest.run_id}")
    return 0


def _cmd_remote_run_collect(args: argparse.Namespace) -> int:
    """``proto-tools remote run collect``."""
    remote_exec.ensure_remote_run_output_path_available(args.output, overwrite=args.overwrite_output)
    manifest = remote_exec.load_remote_run_manifest(args.manifest)
    output = remote_exec.collect_remote_run_manifest(manifest)
    output_path = Path(args.output)
    remote_exec.write_remote_run_output(output, output_path, overwrite=args.overwrite_output)
    summary = {"run_id": manifest.run_id, "output": str(output_path)}
    if args.json:
        print(_dump_json(summary))
    else:
        print(f"output\t{output_path}")
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

    p_remote = sub.add_parser("remote", help="Manage user-owned SSH remote execution nodes.")
    remote_sub = p_remote.add_subparsers(dest="remote_verb", required=True)

    def add_profile_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", required=True, help="JSON profile with a top-level nodes list.")
        p.add_argument("--node", action="append", help="Limit to one node name; repeat for multiple nodes.")

    def add_existing_command_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", required=True, help="Existing-command JSON profile with a top-level nodes list.")
        p.add_argument("--node", required=True, help="Existing-command node name to run on.")
        p.add_argument("--input", required=True, help="Local input file to upload.")
        p.add_argument("--run-id", required=True, help="Safe run identifier used in the remote work directory.")
        p.add_argument("--local-collect-dir", required=True, help="Local directory for collected remote outputs.")

    p_remote_compute = remote_sub.add_parser(
        "compute",
        help="Distribute, supervise, and collect a maintained remote protein runtime.",
    )
    p_remote_compute.add_argument("software", help="Maintained software name or alias, for example af3.")
    p_remote_compute.add_argument("input", help="Input FASTA; each record is one task.")
    p_remote_compute.add_argument("run_dir", help="New local run directory, or the same directory to resume.")
    p_remote_compute.add_argument("--registry", help="Override the $hand-compute runtime registry.")
    p_remote_compute.add_argument("--bucket-size", type=int)
    p_remote_compute.add_argument("--slurm-group-size", type=int)
    p_remote_compute.add_argument("--wait-poll-interval", type=float)
    p_remote_compute.add_argument("--timeout-sec", type=float)
    p_remote_compute.add_argument("--remote-check-timeout-sec", type=float)
    p_remote_compute.add_argument("--json", action="store_true")
    p_remote_compute.set_defaults(func=_cmd_remote_compute)

    p_remote_run = remote_sub.add_parser("run", help="Plan, submit, wait for, and collect remote bucket runs.")
    run_sub = p_remote_run.add_subparsers(dest="run_verb", required=True)

    p_remote_run_plan = run_sub.add_parser("plan", help="Create a manifest-backed remote run plan.")
    p_remote_run_plan.add_argument("tool", help="Tool key to run remotely.")
    add_profile_args(p_remote_run_plan)
    p_remote_run_plan.add_argument("--input-json", required=True, help="JSON input payload for the tool.")
    p_remote_run_plan.add_argument("--config-json", help="JSON config payload for the tool.")
    p_remote_run_plan.add_argument("--bucket-size", type=int, required=True)
    p_remote_run_plan.add_argument("--local-results-dir", required=True)
    p_remote_run_plan.add_argument("--manifest", required=True)
    p_remote_run_plan.add_argument("--run-id")
    p_remote_run_plan.add_argument("--overwrite", action="store_true")
    p_remote_run_plan.add_argument(
        "--use-diagnostics-backlog",
        action="store_true",
        help="Seed planning load from current queue/running/submitted manager diagnostics.",
    )
    p_remote_run_plan.add_argument(
        "--remote-check-timeout-sec",
        type=float,
        help="Per-SSH timeout for diagnostics when --use-diagnostics-backlog is set.",
    )
    p_remote_run_plan.add_argument("--json", action="store_true")
    p_remote_run_plan.set_defaults(func=_cmd_remote_run_plan)

    p_remote_run_launch = run_sub.add_parser(
        "launch",
        help="Preflight managers, submit a manifest run, wait, and collect output.",
    )
    p_remote_run_launch.add_argument("tool", help="Tool key to run remotely.")
    add_profile_args(p_remote_run_launch)
    p_remote_run_launch.add_argument("--input-json", required=True, help="JSON input payload for the tool.")
    p_remote_run_launch.add_argument("--config-json", help="JSON config payload for the tool.")
    p_remote_run_launch.add_argument("--bucket-size", type=int, required=True)
    p_remote_run_launch.add_argument("--local-results-dir", required=True)
    p_remote_run_launch.add_argument("--manifest", required=True)
    p_remote_run_launch.add_argument("--output", required=True)
    p_remote_run_launch.add_argument("--run-id")
    p_remote_run_launch.add_argument("--overwrite", action="store_true")
    p_remote_run_launch.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing merged output JSON.",
    )
    p_remote_run_launch.add_argument("--manager-poll-interval", type=float, default=5.0)
    p_remote_run_launch.add_argument("--slurm-group-size", type=int, default=10)
    p_remote_run_launch.add_argument("--wait-poll-interval", type=float, default=5.0)
    p_remote_run_launch.add_argument("--timeout-sec", type=float)
    p_remote_run_launch.add_argument(
        "--remote-check-timeout-sec",
        type=float,
        help="Per-SSH timeout for manager preflight, start, and diagnostics checks.",
    )
    p_remote_run_launch.add_argument("--rsync-mode", choices=("inline", "background"), default="inline")
    p_remote_run_launch.add_argument("--rsync-pid-file")
    p_remote_run_launch.add_argument("--rsync-log-file")
    p_remote_run_launch.add_argument("--rsync-interval-sec", type=int, default=10)
    p_remote_run_launch.add_argument(
        "--use-diagnostics-backlog",
        action="store_true",
        help="Seed planning load from current queue/running/submitted manager diagnostics.",
    )
    p_remote_run_launch.add_argument("--json", action="store_true")
    p_remote_run_launch.set_defaults(func=_cmd_remote_run_launch)

    p_remote_run_resume = run_sub.add_parser(
        "resume",
        help="Resume an existing manifest run without replanning or rewriting the manifest.",
    )
    p_remote_run_resume.add_argument("--manifest", required=True)
    p_remote_run_resume.add_argument("--output", required=True)
    p_remote_run_resume.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing merged output JSON.",
    )
    p_remote_run_resume.add_argument("--manager-poll-interval", type=float, default=5.0)
    p_remote_run_resume.add_argument("--slurm-group-size", type=int, default=10)
    p_remote_run_resume.add_argument("--wait-poll-interval", type=float, default=5.0)
    p_remote_run_resume.add_argument("--timeout-sec", type=float)
    p_remote_run_resume.add_argument(
        "--remote-check-timeout-sec",
        type=float,
        help="Per-SSH timeout for manager preflight and start checks.",
    )
    p_remote_run_resume.add_argument("--rsync-mode", choices=("inline", "background"), default="inline")
    p_remote_run_resume.add_argument("--rsync-pid-file")
    p_remote_run_resume.add_argument("--rsync-log-file")
    p_remote_run_resume.add_argument("--rsync-interval-sec", type=int, default=10)
    p_remote_run_resume.add_argument("--json", action="store_true")
    p_remote_run_resume.set_defaults(func=_cmd_remote_run_resume)

    p_remote_run_submit = run_sub.add_parser("submit", help="Idempotently stage queued bucket requests from a manifest.")
    p_remote_run_submit.add_argument("--manifest", required=True)
    p_remote_run_submit.add_argument("--json", action="store_true")
    p_remote_run_submit.set_defaults(func=_cmd_remote_run_submit)

    p_remote_run_wait = run_sub.add_parser("wait", help="Wait for manifest results to appear locally.")
    p_remote_run_wait.add_argument("--manifest", required=True)
    p_remote_run_wait.add_argument("--poll-interval", type=float, default=5.0)
    p_remote_run_wait.add_argument("--timeout-sec", type=float)
    p_remote_run_wait.add_argument("--no-pull", action="store_true", help="Do not run rsync pull rounds while waiting.")
    p_remote_run_wait.add_argument("--rsync-pid-file", help="Background rsync pid file to monitor while waiting.")
    p_remote_run_wait.add_argument("--rsync-log-file", help="Background rsync log file to include when the pull loop stops.")
    p_remote_run_wait.add_argument("--json", action="store_true")
    p_remote_run_wait.set_defaults(func=_cmd_remote_run_wait)

    p_remote_run_collect = run_sub.add_parser("collect", help="Merge completed manifest results into one output JSON.")
    p_remote_run_collect.add_argument("--manifest", required=True)
    p_remote_run_collect.add_argument("--output", required=True)
    p_remote_run_collect.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing an existing merged output JSON.",
    )
    p_remote_run_collect.add_argument("--json", action="store_true")
    p_remote_run_collect.set_defaults(func=_cmd_remote_run_collect)

    p_remote_existing = remote_sub.add_parser("existing", help="Run already-installed remote command profiles.")
    existing_sub = p_remote_existing.add_subparsers(dest="existing_verb", required=True)

    p_remote_existing_plan = existing_sub.add_parser("plan", help="Render stage/execute/collect commands.")
    add_existing_command_args(p_remote_existing_plan)
    p_remote_existing_plan.add_argument("--json", action="store_true")
    p_remote_existing_plan.set_defaults(func=_cmd_remote_existing_plan)

    p_remote_existing_run = existing_sub.add_parser("run", help="Run one existing remote command and collect outputs.")
    add_existing_command_args(p_remote_existing_run)
    p_remote_existing_run.add_argument("--timeout-sec", type=float)
    p_remote_existing_run.add_argument("--json", action="store_true")
    p_remote_existing_run.set_defaults(func=_cmd_remote_existing_run)

    p_remote_existing_launch = existing_sub.add_parser(
        "launch",
        help="Distribute an existing-command FASTA batch through remote managers.",
    )
    p_remote_existing_launch.add_argument("software", help="Maintained software name, for example alphafold3.")
    p_remote_existing_launch.add_argument("--profile", required=True, help="Combined manager/command profile JSON.")
    p_remote_existing_launch.add_argument("--input", required=True, help="Input FASTA; each record is one task.")
    p_remote_existing_launch.add_argument(
        "--run-dir",
        required=True,
        help="New run directory, or an exact matching manifest to resume in place.",
    )
    p_remote_existing_launch.add_argument("--bucket-size", type=int, default=1)
    p_remote_existing_launch.add_argument("--slurm-group-size", type=int, default=10)
    p_remote_existing_launch.add_argument("--wait-poll-interval", type=float, default=5.0)
    p_remote_existing_launch.add_argument("--timeout-sec", type=float)
    p_remote_existing_launch.add_argument("--remote-check-timeout-sec", type=float)
    p_remote_existing_launch.add_argument("--json", action="store_true")
    p_remote_existing_launch.set_defaults(func=_cmd_remote_existing_launch)

    p_remote_profile = remote_sub.add_parser("profile", help="Inspect remote node profiles.")
    profile_sub = p_remote_profile.add_subparsers(dest="profile_verb", required=True)

    p_remote_profile_scaffold = profile_sub.add_parser("scaffold", help="Write a remote node profile JSON.")
    p_remote_profile_scaffold.add_argument("--output", required=True, help="Path to write the profile JSON.")
    p_remote_profile_scaffold.add_argument("--direct", action="append", default=[], metavar="NAME[@HOST]=HOME")
    p_remote_profile_scaffold.add_argument("--slurm", action="append", default=[], metavar="NAME[@HOST]=HOME")
    p_remote_profile_scaffold.add_argument("--repo-name", default="proto-tools")
    p_remote_profile_scaffold.add_argument("--work-name", default="proto_remote")
    p_remote_profile_scaffold.add_argument("--work-name-template")
    p_remote_profile_scaffold.add_argument("--bootstrap-python", default="python3")
    p_remote_profile_scaffold.add_argument("--direct-worker-device", default="cuda:0")
    p_remote_profile_scaffold.add_argument("--slurm-worker-device", default="cuda")
    p_remote_profile_scaffold.add_argument("--direct-weight", type=float, default=1.0)
    p_remote_profile_scaffold.add_argument("--slurm-weight", type=float, default=1.0)
    p_remote_profile_scaffold.add_argument("--direct-max-concurrent-buckets", type=int, default=1)
    p_remote_profile_scaffold.add_argument("--slurm-max-concurrent-buckets", type=int, default=1)
    p_remote_profile_scaffold.add_argument(
        "--slurm-max-active-jobs",
        type=int,
        default=1,
        help="Maximum submitted Slurm jobs one node manager may keep active.",
    )
    p_remote_profile_scaffold.add_argument("--slurm-sbatch-arg", action="append", default=[])
    p_remote_profile_scaffold.add_argument("--overwrite", action="store_true")
    p_remote_profile_scaffold.add_argument("--json", action="store_true")
    p_remote_profile_scaffold.set_defaults(func=_cmd_remote_profile_scaffold)

    p_remote_profile_list = profile_sub.add_parser("list", help="List configured remote nodes.")
    add_profile_args(p_remote_profile_list)
    p_remote_profile_list.add_argument("--json", action="store_true")
    p_remote_profile_list.set_defaults(func=_cmd_remote_profile_list)
    p_remote_profile_validate = profile_sub.add_parser("validate", help="Run static checks on a remote node profile.")
    add_profile_args(p_remote_profile_validate)
    p_remote_profile_validate.add_argument("--json", action="store_true")
    p_remote_profile_validate.set_defaults(func=_cmd_remote_profile_validate)

    p_remote_smoke = remote_sub.add_parser("smoke", help="Run no-submit remote profile and manager checks.")
    add_profile_args(p_remote_smoke)
    p_remote_smoke.add_argument("--local-results-dir", default="remote_results")
    p_remote_smoke.add_argument("--timeout-sec", type=float)
    p_remote_smoke.add_argument("--json", action="store_true")
    p_remote_smoke.set_defaults(func=_cmd_remote_smoke)

    p_remote_deploy = remote_sub.add_parser("deploy", help="Render or apply remote manager deployment.")
    deploy_sub = p_remote_deploy.add_subparsers(dest="deploy_verb", required=True)

    p_remote_deploy_render = deploy_sub.add_parser("render", help="Render deployment commands without running them.")
    add_profile_args(p_remote_deploy_render)
    p_remote_deploy_render.add_argument("--local-repo-dir", required=True, help="Local proto-tools checkout to upload.")
    p_remote_deploy_render.add_argument(
        "--allow-dirty-checkout",
        action="store_true",
        help="Mark dirty local checkout state as a warning instead of a blocker.",
    )
    p_remote_deploy_render.add_argument("--json", action="store_true")
    p_remote_deploy_render.set_defaults(func=_cmd_remote_deploy_render)

    p_remote_deploy_apply = deploy_sub.add_parser("apply", help="Run deployment commands on remote nodes.")
    add_profile_args(p_remote_deploy_apply)
    p_remote_deploy_apply.add_argument("--local-repo-dir", required=True, help="Local proto-tools checkout to upload.")
    p_remote_deploy_apply.add_argument(
        "--allow-shared-install-root",
        action="store_true",
        help="Allow selected nodes to write the same repo_dir or venv_dir.",
    )
    p_remote_deploy_apply.add_argument(
        "--allow-dirty-checkout",
        action="store_true",
        help="Allow uploading a checkout with tracked or untracked local changes.",
    )
    p_remote_deploy_apply.add_argument("--json", action="store_true")
    p_remote_deploy_apply.set_defaults(func=_cmd_remote_deploy_apply)

    p_remote_manager = remote_sub.add_parser("manager", help="Start and inspect node-side managers.")
    manager_sub = p_remote_manager.add_subparsers(dest="manager_verb", required=True)
    p_remote_manager_start = manager_sub.add_parser("start", help="Start manager loops on remote nodes.")
    add_profile_args(p_remote_manager_start)
    p_remote_manager_start.add_argument("--poll-interval", type=float, default=5.0)
    p_remote_manager_start.add_argument("--slurm-group-size", type=int, default=10)
    p_remote_manager_start.add_argument("--timeout-sec", type=float, help="Per-SSH timeout for this manager command.")
    p_remote_manager_start.add_argument("--json", action="store_true")
    p_remote_manager_start.set_defaults(func=_cmd_remote_manager_start)

    for verb, handler, help_text in (
        ("status", _cmd_remote_manager_status, "Show manager pid state."),
        ("preflight", _cmd_remote_manager_preflight, "Run remote environment checks."),
        ("diagnostics", _cmd_remote_manager_diagnostics, "Show manager queue/log diagnostics."),
    ):
        p = manager_sub.add_parser(verb, help=help_text)
        add_profile_args(p)
        p.add_argument("--timeout-sec", type=float, help="Per-SSH timeout for this manager command.")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=handler)

    p_remote_rsync = remote_sub.add_parser("rsync", help="Manage local result pull loops.")
    rsync_sub = p_remote_rsync.add_subparsers(dest="rsync_mode", required=True)
    p_remote_rsync_pull = rsync_sub.add_parser("pull", help="Pull remote result roots to a local mirror.")
    pull_sub = p_remote_rsync_pull.add_subparsers(dest="rsync_pull_verb", required=True)

    p_remote_rsync_once = pull_sub.add_parser("once", help="Run one rsync pull round for all selected nodes.")
    add_profile_args(p_remote_rsync_once)
    p_remote_rsync_once.add_argument("--local-results-dir", required=True)
    p_remote_rsync_once.set_defaults(func=_cmd_remote_rsync_once)

    p_remote_rsync_start = pull_sub.add_parser("start", help="Start a local background rsync pull loop.")
    add_profile_args(p_remote_rsync_start)
    p_remote_rsync_start.add_argument("--local-results-dir", required=True)
    p_remote_rsync_start.add_argument("--pid-file", required=True)
    p_remote_rsync_start.add_argument("--log-file", required=True)
    p_remote_rsync_start.add_argument("--interval-sec", type=int, default=10)
    p_remote_rsync_start.set_defaults(func=_cmd_remote_rsync_start)

    p_remote_rsync_status = pull_sub.add_parser("status", help="Show local rsync pull loop status.")
    p_remote_rsync_status.add_argument("--pid-file", required=True)
    p_remote_rsync_status.set_defaults(func=_cmd_remote_rsync_status)

    p_remote_rsync_stop = pull_sub.add_parser("stop", help="Stop a local rsync pull loop.")
    p_remote_rsync_stop.add_argument("--pid-file", required=True)
    p_remote_rsync_stop.set_defaults(func=_cmd_remote_rsync_stop)

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
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
