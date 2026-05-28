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


def _model_table(doc: ModelDoc, kind: str) -> str:
    """Render a ``ModelDoc`` as a markdown table for notebook display."""
    if not doc.fields:
        return f"*No {kind} fields.*"

    def esc(text: str) -> str:
        """Escape ``|`` so a value stays inside one markdown table cell."""
        return text.replace("|", "\\|")

    rows: list[str] = []
    for f in doc.fields:
        if f.required:
            default = "required"
        elif f.default is not None:
            default = f"`{esc(repr(f.default))}`"
        else:
            default = "`None`"
        desc = esc((f.description or "").replace("\n", " ")).strip()
        rows.append(f"| `{f.name}` | `{esc(f.type_str)}` | {default} | {desc} |")
    header = (
        f"**{kind.capitalize()}** — `{doc.name}`\n\n"
        "| Field | Type | Default | Description |\n"
        "|-------|------|---------|-------------|"
    )
    return header + "\n" + "\n".join(rows)


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

    model_attr = {"input": "input_model", "config": "config_model", "output": "output_model"}.get(model.lower())
    if model_attr is None:
        msg = f"*Unknown model `{model}` (use 'input', 'config', or 'output').*"
        display(Markdown(msg))  # type: ignore[no-untyped-call]
        return

    doc = get_model_doc(getattr(target, model_attr))
    display(Markdown(_model_table(doc, model.lower())))  # type: ignore[no-untyped-call]
