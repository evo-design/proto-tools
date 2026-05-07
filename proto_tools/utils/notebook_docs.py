"""Display tool documentation sections in notebooks from local sources.

Pulls content directly from each toolkit's ``README.md`` and from the
``ToolRegistry`` (Pydantic Input/Config/Output schemas). No network access.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Live docs site — only used to build the "View in Proto Docs" link.
_DOCS_BASE_URL = "https://bio-pro.mintlify.app/tools"


def _toolkit_dir(tool: str) -> Path:
    """Resolve a tool identifier to its on-disk toolkit directory.

    Accepts any of: docs-style path (``"structure-prediction/esmfold"``), tool
    directory name (``"esmfold"``), run function name (``"run_esmfold"``), or
    registry key (``"esmfold-prediction"``).

    Args:
        tool (str): Tool identifier in any supported format.

    Returns:
        Path: Absolute path to ``proto_tools/tools/{category}/{toolkit}``.

    Raises:
        ValueError: If no toolkit matches.
    """
    from proto_tools.tools.tool_registry import ToolRegistry

    if "/" in tool:
        category, toolkit = tool.split("/", 1)
        cat_us = category.replace("-", "_")
        tk_us = toolkit.replace("-", "_")
        for spec in ToolRegistry.list_all():
            p = spec.source_file.parent
            if p.name == tk_us and p.parent.name == cat_us:
                return p
        raise ValueError(f"Could not resolve toolkit dir for '{tool}'")

    func_prefix = tool if tool.startswith("run_") else f"run_{tool}"
    tool_normalized = tool.replace("-", "_").removeprefix("run_")

    for spec in ToolRegistry.list_all():
        p = spec.source_file.parent
        if (
            spec.key == tool
            or spec.function.__name__ == func_prefix
            or spec.function.__name__.startswith(func_prefix)
            or p.name == tool_normalized
        ):
            return p

    raise ValueError(f"Could not resolve toolkit for '{tool}'")


def _docs_url_path(tool: str) -> str:
    """Return the docs-site URL path segment for a tool."""
    p = _toolkit_dir(tool)
    return f"{p.parent.name.replace('_', '-')}/{p.name.replace('_', '-')}"


def _read_readme(tool: str) -> str:
    """Read the toolkit's README.md from disk."""
    return (_toolkit_dir(tool) / "README.md").read_text()


def _toolkit_specs(tool: str) -> list[Any]:
    """Return the registry specs for tools whose source files live in this toolkit dir."""
    from proto_tools.tools.tool_registry import ToolRegistry

    target = _toolkit_dir(tool)
    return sorted(
        (s for s in ToolRegistry.list_all() if s.source_file.parent == target),
        key=lambda s: s.key,
    )


_TODO_CALLOUT_RE = re.compile(
    r"^>\s*\[!NOTE\]\s*\n>\s*\*\*TODO:\*\*\s*This README still needs to be reviewed[^\n]*\n+",
    re.MULTILINE,
)


def _strip_review_callout(md: str) -> str:
    """Strip the temporary ``> [!NOTE] **TODO: review**`` callout from a README."""
    return _TODO_CALLOUT_RE.sub("", md)


def _extract_section(md: str, heading: str) -> str:
    """Extract a markdown section by heading text at any level.

    Matches the first heading containing the given text and captures everything
    until the next heading at the same or higher level.

    Args:
        md (str): Full markdown content.
        heading (str): Section heading text (e.g. ``"Background"``).

    Returns:
        str: Section content including its heading, or empty string if not found.
    """
    m = re.search(rf"^(#{{1,6}})\s+{re.escape(heading)}\s*$", md, re.MULTILINE)
    if not m:
        return ""
    level = len(m.group(1))
    start = m.start()
    end_match = re.search(rf"^#{{{1},{level}}}\s+", md[m.end() :], re.MULTILINE)
    end = m.end() + end_match.start() if end_match else len(md)
    return md[start:end].strip()


def _extract_title(md: str) -> str:
    """Return the first H1 line in the README, or empty string."""
    m = re.search(r"^#\s+.+$", md, re.MULTILINE)
    return m.group() if m else ""


def _format_type(annotation: Any) -> str:
    """Stringify a Python type annotation for table display."""
    return re.sub(r"<class '([^']+)'>", r"\1", str(annotation).replace("typing.", ""))


def _format_default(field_info: Any) -> str:
    """Stringify a Pydantic v2 ``FieldInfo`` default for table display."""
    if field_info.is_required():
        return "required"
    default = field_info.get_default(call_default_factory=True)
    if default is None:
        return "`None`"
    return f"`{default!r}`"


def _render_model_table(model_class: type[BaseModel], kind: str) -> str:
    """Render a Pydantic model's fields as a markdown table.

    Args:
        model_class (type[BaseModel]): Pydantic ``BaseModel`` subclass.
        kind (str): One of ``"input"``, ``"config"``, ``"output"``.

    Returns:
        str: Markdown table of name / type / default / description.
    """
    from proto_tools.utils.tool_io import _OUTPUT_METADATA_FIELDS

    exclude = _OUTPUT_METADATA_FIELDS if kind == "output" else set()
    rows: list[str] = []
    for fname, finfo in model_class.model_fields.items():
        if fname in exclude:
            continue
        type_str = _format_type(finfo.annotation)
        default_str = _format_default(finfo)
        desc = (finfo.description or "").replace("\n", " ").replace("|", "\\|").strip()
        rows.append(f"| `{fname}` | `{type_str}` | {default_str} | {desc} |")

    if not rows:
        return f"*No {kind} fields.*"

    header = (
        f"**{kind.capitalize()}** — `{model_class.__name__}`\n\n"
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
        md = _strip_review_callout(_read_readme(tool))
    except (ValueError, OSError) as exc:
        logger.warning("Unable to read README for '%s': %s", tool, exc)
        return

    title = _extract_title(md)
    body = _extract_section(md, "Overview")
    body = re.sub(r"^##\s+Overview\s*\n*", "", body)
    text = f"{title}\n\n{body}".strip() if title else body
    display(Markdown(text))  # type: ignore[no-untyped-call]


def display_docs_section(tool: str, section: str) -> None:
    """Render one named section from the tool's README.

    Args:
        tool (str): Tool identifier — full path, tool name, or run function name.
        section (str): Section heading to extract (e.g. ``"Background"``).
    """
    from IPython.display import Markdown, display

    try:
        md = _strip_review_callout(_read_readme(tool))
    except (ValueError, OSError) as exc:
        logger.warning("Unable to read README for '%s': %s", tool, exc)
        return

    content = _extract_section(md, section)
    if not content:
        content = f"*Section `{section}` not found in README.*"
    display(Markdown(content))  # type: ignore[no-untyped-call]


def display_available_tools(tool: str) -> None:
    """List the run functions registered for the toolkit, with their descriptions.

    Args:
        tool (str): Tool identifier — full path, tool name, or run function name.
    """
    from IPython.display import Markdown, display

    try:
        specs = _toolkit_specs(tool)
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
        specs = _toolkit_specs(tool)
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

    display(Markdown(_render_model_table(getattr(target, model_attr), model.lower())))  # type: ignore[no-untyped-call]
