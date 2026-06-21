"""Notebook display helpers for tool documentation.

Thin Jupyter-display wrappers over the text-extraction API in
``proto_tools.utils.tool_docs``. Each ``display_*`` function pulls structured
content via the public extractors, renders it through ``IPython.display``, and
returns ``None`` (the display is the side-effect).

Use ``proto_tools.utils.tool_docs`` directly for programmatic access — agents,
MCP servers, and any non-notebook consumer should not import from this module.
"""

from __future__ import annotations

import logging
import typing

from pydantic import BaseModel

from proto_tools.utils.tool_docs import (
    ModelDoc,
    get_model_doc,
    get_readme_section,
    get_readme_sections,
    resolve_toolkit_dir,
    toolkit_specs,
)

logger = logging.getLogger(__name__)

# Live docs site — only used to build the "View in Proto Docs" link.
_DOCS_BASE_URL = "https://bio-pro.mintlify.app/tools"


def _docs_url_path(tool: str) -> str:
    """Return the docs-site URL path segment for a tool."""
    p = resolve_toolkit_dir(tool)
    return f"{p.parent.name.replace('_', '-')}/{p.name.replace('_', '-')}"


def _esc(text: str) -> str:
    """Escape ``|`` so a value stays inside one markdown table cell."""
    return text.replace("|", "\\|")


def _code(text: str) -> str:
    """Render inline code via HTML ``<code>`` so pipes survive table cells.

    Markdown code spans ignore backslash escapes, so ``|`` inside one breaks
    the table. HTML ``<code>`` with entity-encoded pipes renders correctly in
    both Jupyter and GitHub.
    """
    html = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "&#124;")
    return f"<code>{html}</code>"


def _field_rows(doc: ModelDoc) -> str:
    """Render a model's fields as markdown table body rows (one row per field)."""
    rows: list[str] = []
    for f in doc.fields:
        if f.required:
            default = "required"
        elif f.default is not None:
            default = _code(repr(f.default))
        else:
            default = _code("None")
        desc = _esc((f.description or "").replace("\n", " ")).strip()
        rows.append(f"| {_code(f.name)} | {_code(f.type_str)} | {default} | {desc} |")
    return "\n".join(rows)


def _field_table(title: str, doc: ModelDoc) -> str:
    """Render one model's fields as a titled markdown table."""
    if not doc.fields:
        return f"{title}\n\n*No fields.*"
    header = f"{title}\n\n| Field | Type | Default | Description |\n|-------|------|---------|-------------|"
    return header + "\n" + _field_rows(doc)


def _model_table(doc: ModelDoc, kind: str) -> str:
    """Render a ``ModelDoc`` as a markdown table for notebook display."""
    if not doc.fields:
        return f"*No {kind} fields.*"
    return _field_table(f"**{kind.capitalize()}** — `{doc.name}`", doc)


def _is_entity(cls: type) -> bool:
    """True for shared entity models (Structure, Ligands, …); rendered as leaf types."""
    return "proto_tools.entities" in getattr(cls, "__module__", "")


def _is_metrics(cls: type) -> bool:
    """True for Metrics subclasses; rendered from ``metric_spec``, not raw fields."""
    return hasattr(cls, "metric_spec")


def _nested_submodels(model_class: type[BaseModel]) -> list[type[BaseModel]]:
    """Return nested Pydantic submodels referenced by a model's fields, in field order.

    Walks each field annotation (including inside ``list``/``dict``/``Union``) and
    collects referenced models, skipping shared entities and Metrics classes — those
    are rendered specially or left as leaf types.
    """
    found: list[type[BaseModel]] = []

    def walk(annotation: object) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if not _is_entity(annotation) and not _is_metrics(annotation) and annotation not in found:
                found.append(annotation)
            return
        for arg in typing.get_args(annotation):
            walk(arg)

    for info in model_class.model_fields.values():
        walk(info.annotation)
    return found


def _metric_table(metrics_class: type, per_item_field: str | None) -> str:
    """Render a Metrics subclass's ``metric_spec`` as a markdown table, or ``""`` if none."""
    specs = get_model_doc(metrics_class, metrics_class=metrics_class).metric_specs or []
    if not specs:
        return ""
    rows: list[str] = []
    for m in specs:
        if m.min is not None and m.max is not None:
            rng = _code(f"[{m.min:g}, {m.max:g}]")
        elif m.min is not None:
            rng = _code(f">= {m.min:g}")
        elif m.max is not None:
            rng = _code(f"<= {m.max:g}")
        else:
            rng = ""
        name = _code(m.name) + (" **(primary)**" if m.is_primary else "")
        unit = _code(m.unit) if m.unit else ""
        desc = _esc((m.description or "").replace("\n", " ")).strip()
        rows.append(f"| {name} | {_code(m.type_str or '')} | {rng} | {unit} | {desc} |")
    scope = f" (one set per `{per_item_field}` item)" if per_item_field else ""
    header = (
        f"**Metrics** — `{metrics_class.__name__}`{scope}\n\n"
        "| Metric | Type | Range | Unit | Description |\n"
        "|--------|------|-------|------|-------------|"
    )
    return header + "\n" + "\n".join(rows)


def _render_output(spec: object) -> str:
    """Render a tool's output model: top-level table, nested submodels, then metrics.

    Recurses nested Pydantic submodels (stopping at shared entities and Metrics
    classes), then appends the metric-spec table for the tool's Metrics model.
    """
    blocks: list[str] = []
    seen: set[type] = set()

    def walk(cls: type, title: str) -> None:
        if cls in seen:
            return
        seen.add(cls)
        blocks.append(_field_table(title, get_model_doc(cls)))
        for sub in _nested_submodels(cls):
            walk(sub, f"**`{sub.__name__}`**")

    walk(spec.output_model, f"**Output** — `{spec.output_model.__name__}`")
    if spec.metrics_model is not None:
        metric_md = _metric_table(spec.metrics_model, spec.iterable_output_field)
        if metric_md:
            blocks.append(metric_md)
    return "\n\n".join(blocks)


def display_overview(tool: str) -> None:
    """Render the tool title and the ``## Overview`` paragraph.

    Args:
        tool (str): Tool identifier — full path (``"structure-prediction/esmfold"``),
            tool name (``"esmfold"``), or run function (``"run_esmfold"``).
    """
    from IPython.display import Markdown, display

    try:
        sections = get_readme_sections(tool)
    except (ValueError, OSError) as exc:
        logger.warning("Unable to read README for '%s': %s", tool, exc)
        return

    title = f"# {sections.title}" if sections.title else ""
    text = f"{title}\n\n{sections.overview}".strip() if title else sections.overview
    display(Markdown(text))  # type: ignore[no-untyped-call]


def display_docs_section(tool: str, section: str) -> None:
    """Render one named H2 section from the tool's README.

    Args:
        tool (str): Tool identifier — full path, tool name, or run function name.
        section (str): Section heading to extract (e.g. ``"Background"``).
    """
    from IPython.display import Markdown, display

    try:
        body = get_readme_section(tool, section)
    except (ValueError, OSError) as exc:
        logger.warning("Unable to read README for '%s': %s", tool, exc)
        return

    content = body or f"*Section `{section}` not found in README.*"
    display(Markdown(content))  # type: ignore[no-untyped-call]


def display_available_tools(tool: str) -> None:
    """List the run functions registered for the toolkit, with their descriptions.

    Args:
        tool (str): Tool identifier — full path, tool name, or run function name.
    """
    from IPython.display import Markdown, display

    try:
        specs = toolkit_specs(tool)
    except ValueError as exc:
        logger.warning("%s", exc)
        return

    if not specs:
        display(Markdown("*No tools registered for this toolkit.*"))  # type: ignore[no-untyped-call]
        return

    lines = [f"- **`{s.function.__name__}()`** — {s.description}" for s in specs]
    display(Markdown("\n".join(lines)))  # type: ignore[no-untyped-call]


def display_doc_link(tool: str, label: str = "VIEW IN PROTO DOCS") -> None:
    """Display a shield-style badge linking to the tool's page on the live docs site.

    Renders an inline SVG badge — works offline and never hits the network.

    Args:
        tool (str): Tool identifier — full path, tool name, or run function name.
        label (str): Badge text. Defaults to ``"VIEW IN PROTO DOCS"``.
    """
    from IPython.display import HTML, display

    try:
        url_path = _docs_url_path(tool)
    except ValueError as exc:
        logger.warning("%s", exc)
        return

    url = f"{_DOCS_BASE_URL}/{url_path}"
    text = label.upper()
    text_width = len(text) * 7.5 + 20
    total_width = 30 + text_width
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" height="28" width="{total_width}">'
        f'<rect rx="4" width="{total_width}" height="28" fill="#046e7a"/>'
        '<rect rx="0" x="0" width="30" height="28" fill="#046e7a"/>'
        '<text x="15" y="18" fill="white" font-size="14" text-anchor="middle">'
        "\U0001f4d6</text>"
        f'<text x="{30 + text_width / 2}" y="18" fill="white" '
        'font-family="Verdana,sans-serif" font-size="10" font-weight="bold" '
        f'text-anchor="middle" letter-spacing="1">{text}</text>'
        "</svg>"
    )
    display(HTML(f'<a href="{url}" target="_blank">{svg}</a>'))  # type: ignore[no-untyped-call]


def display_api_reference(tool: str, model: str, function_name: str | None = None) -> None:
    """Render one Input / Config / Output Pydantic model as a markdown table.

    Args:
        tool (str): Tool identifier — full path, tool name, or run function name.
        model (str): One of ``"input"``, ``"config"``, ``"output"``.
        function_name (str | None): Run function name for multi-function toolkits
            (e.g. ``"run_proteinmpnn_sample"``). Optional for single-function toolkits.
    """
    from IPython.display import Markdown, display

    try:
        specs = toolkit_specs(tool)
    except ValueError as exc:
        logger.warning("%s", exc)
        return

    if function_name:
        target = next((s for s in specs if s.function.__name__ == function_name), None)
        if target is None:
            display(Markdown(f"*Function `{function_name}` not found in toolkit.*"))  # type: ignore[no-untyped-call]
            return
    elif len(specs) == 1:
        target = specs[0]
    elif len(specs) > 1:
        names = ", ".join(f"`{s.function.__name__}`" for s in specs)
        msg = f"*Multi-function toolkit; pass `function_name` (one of: {names}).*"
        display(Markdown(msg))  # type: ignore[no-untyped-call]
        return
    else:
        display(Markdown("*No tools registered for this toolkit.*"))  # type: ignore[no-untyped-call]
        return

    model_key = model.lower()
    model_attr = {"input": "input_model", "config": "config_model", "output": "output_model"}.get(model_key)
    if model_attr is None:
        msg = f"*Unknown model `{model}` (use 'input', 'config', or 'output').*"
        display(Markdown(msg))  # type: ignore[no-untyped-call]
        return

    # The output model nests submodel and metric tables; input/config stay flat.
    if model_key == "output":
        markdown = _render_output(target)
    else:
        markdown = _model_table(get_model_doc(getattr(target, model_attr)), model_key)
    display(Markdown(markdown))  # type: ignore[no-untyped-call]
