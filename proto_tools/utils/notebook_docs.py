"""Fetch and display documentation sections from the Proto docs site in notebooks."""

from __future__ import annotations

import logging
import re
import urllib.request

logger = logging.getLogger(__name__)

_DOCS_BASE_URL = "https://bio-pro.mintlify.app/tools"


def _resolve_tool_path(tool: str) -> str:
    """Resolve a tool identifier to its docs URL path.

    Accepts any of: full path (``"structure-prediction/esmfold"``), tool directory
    name (``"esmfold"``), run function name (``"run_esmfold"``), or registry key
    (``"esmfold-prediction"``).

    Args:
        tool (str): Tool identifier in any supported format.

    Returns:
        str: URL path segment like ``"structure-prediction/esmfold"``.
    """
    if "/" in tool:
        return tool

    from proto_tools.tools.tool_registry import ToolRegistry

    # Try matching by function name, tool dir name, or registry key
    func_prefix = tool if tool.startswith("run_") else f"run_{tool}"
    tool_normalized = tool.replace("-", "_").removeprefix("run_")

    for spec in ToolRegistry.list_all():
        src = str(spec.source_file)
        m = re.search(r"proto_tools/tools/([^/]+)/([^/]+)/", src)
        if not m:
            continue
        cat, tool_dir = m.group(1), m.group(2)

        if (
            spec.function.__name__ == func_prefix
            or spec.function.__name__.startswith(func_prefix)
            or tool_dir == tool_normalized
            or spec.key == tool
        ):
            return f"{cat.replace('_', '-')}/{tool_dir.replace('_', '-')}"

    return tool


def _fetch_markdown(tool_path: str) -> str:
    """Fetch raw markdown for a tool docs page.

    Args:
        tool_path (str): Path segment after ``/tools/``, e.g. ``"structure-prediction/esmfold"``.

    Returns:
        str: Raw markdown content.
    """
    url = f"{_DOCS_BASE_URL}/{tool_path}.md"
    resp: bytes = urllib.request.urlopen(url).read()  # noqa: S310
    return resp.decode()


def _extract_section(md: str, heading: str) -> str:
    """Extract a markdown section by heading text at any level.

    Matches the first heading containing the given text (e.g. ``# ESMFold``,
    ``## Background``) and captures everything until the next heading at the
    same or higher level.

    Args:
        md (str): Full markdown content.
        heading (str): Section heading text (e.g. ``"Background"`` or ``"ESMFold"``).

    Returns:
        str: Section content including the heading, or the full markdown if not found.
    """
    heading_match = re.search(rf"^(#{{1,6}}) {re.escape(heading)}", md, re.MULTILINE)
    if not heading_match:
        return md
    level = len(heading_match.group(1))
    start = heading_match.start()
    # Stop at next heading at same or higher level (1..level # chars)
    end_match = re.search(rf"^#{{{1},{level}}} ", md[heading_match.end() :], re.MULTILINE)
    end = heading_match.end() + end_match.start() if end_match else len(md)
    return md[start:end].strip()


def _extract_overview(md: str) -> str:
    """Extract the page title and description blockquote.

    Args:
        md (str): Full page markdown.

    Returns:
        str: Title heading + description blockquote.
    """
    title_match = re.search(r"^(# .+)$", md, re.MULTILINE)
    if not title_match:
        return md

    title = title_match.group(1)
    rest = md[title_match.end() :]
    # The description is the first blockquote after the title (skipping the llms.txt notice)
    bq_match = re.search(r"^(> (?!##).+)$", rest, re.MULTILINE)
    if bq_match:
        return f"{title}\n\n{bq_match.group(1)}"
    return title


def _strip_html(text: str) -> str:
    """Remove HTML/SVG tags and Mintlify MDX components, keeping inner text.

    Args:
        text (str): Raw markdown with HTML.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"<svg[^>]*>.*?</svg>", "", text, flags=re.DOTALL)
    text = re.sub(r"<a[^>]*>.*?</a>", "", text, flags=re.DOTALL)
    text = re.sub(r"<div[^>]*>|</div>", "", text)
    text = re.sub(r"<span[^>]*>|</span>", "", text)
    text = re.sub(r"<img[^>]*/?>", "", text)
    text = re.sub(r"<input[^>]*/?>", "", text)
    text = re.sub(r"<label[^>]*>.*?</label>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?Expandable[^>]*>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_function_block(api_section: str, function_name: str | None) -> str:
    """Extract the API block for a specific run function.

    Args:
        api_section (str): The full ``## API Docs`` section.
        function_name (str | None): e.g. ``"run_esmfold"``; None for single-function tools.

    Returns:
        str: The markdown block for that function's Input/Config/Output.
    """
    if function_name is None:
        return api_section

    parts = re.split(r"(?=### `run_\w+\(\)`)", api_section)
    for part in parts:
        if f"### `{function_name}()`" in part:
            return part

    return api_section


def _paramfield_to_row(tag_attrs: str, description: str) -> str:
    """Convert ``<ParamField>``/``<ResponseField>`` attributes to a markdown table row.

    Args:
        tag_attrs (str): The tag's attribute string.
        description (str): Field description text.

    Returns:
        str: Formatted table row.
    """
    name = re.search(r'(?:path|name)="([^"]*)"', tag_attrs)
    type_ = re.search(r'type="([^"]*)"', tag_attrs)
    default = re.search(r'default="([^"]*)"', tag_attrs)
    required = "required" in tag_attrs

    name_str = f"`{name.group(1)}`" if name else ""
    type_str = f"`{type_.group(1)}`" if type_ else ""
    default_str = f"`{default.group(1)}`" if default else "required" if required else ""
    return f"| {name_str} | {type_str} | {default_str} | {description} |"


def _convert_model_section(section_html: str) -> str:
    """Convert one api-model-section div to a markdown table.

    Args:
        section_html (str): HTML content of one model section.

    Returns:
        str: Clean markdown table.
    """
    badge_match = re.search(r"api-(input|config|output)-badge", section_html)
    badge = badge_match.group(1).capitalize() if badge_match else "Model"

    class_match = re.search(r"####\s+(\w+)", section_html)
    class_name = class_match.group(1) if class_match else badge

    # Extract fields: tag attrs + description text until next field or closing tag
    field_pattern = (
        r"<(?:ParamField|ResponseField)\s+([^>]+)>\s*(.*?)"
        r"(?=<(?:ParamField|ResponseField)|</div>\s*(?:<div|$)|\Z)"
    )
    rows = []
    for m in re.finditer(field_pattern, section_html, re.DOTALL):
        tag_attrs = m.group(1)
        desc_raw = m.group(2)
        # Strip nested tags, collapse whitespace
        desc = re.sub(r"<[^>]+>", "", desc_raw).strip()
        desc = re.sub(r"\s+", " ", desc)
        rows.append(_paramfield_to_row(tag_attrs, desc))

    header = (
        f"**{badge}** — `{class_name}`\n\n"
        f"| Field | Type | Default | Description |\n"
        f"|-------|------|---------|-------------|"
    )
    return header + "\n" + "\n".join(rows)


def _extract_api_model(md: str, model: str, function_name: str | None = None) -> str:
    """Extract a single Input/Config/Output model from the API Docs section.

    Args:
        md (str): Full page markdown.
        model (str): One of ``"input"``, ``"config"``, or ``"output"``.
        function_name (str | None): Run function name for multi-function tools.

    Returns:
        str: Cleaned markdown table for that model.
    """
    api_section = _extract_section(md, "API Docs")
    block = _extract_function_block(api_section, function_name)

    # Split on model section divs and find the matching one
    sections = re.split(r'(?=<div class="api-model-section )', block)
    for section in sections:
        if f"api-{model}-section" in section:
            return _convert_model_section(section)

    return f"*{model.capitalize()} section not found.*"


def display_overview(tool: str) -> None:
    """Fetch a docs page and render the tool title and description.

    Args:
        tool (str): Tool identifier — full path (``"structure-prediction/esmfold"``),
            tool name (``"esmfold"``), or run function (``"run_esmfold"``).
    """
    from IPython.display import Markdown, display

    tool_path = _resolve_tool_path(tool)
    try:
        md = _fetch_markdown(tool_path)
    except Exception:
        logger.warning("Unable to fetch docs for '%s'", tool_path)
        return
    display(Markdown(_extract_overview(md)))  # type: ignore[no-untyped-call]


def display_docs_section(tool: str, section: str) -> None:
    """Fetch a docs page and render one section as notebook markdown.

    Args:
        tool (str): Tool identifier — full path (``"structure-prediction/esmfold"``),
            tool name (``"esmfold"``), or run function (``"run_esmfold"``).
        section (str): Section heading to extract (e.g. ``"Background"``).
    """
    from IPython.display import Markdown, display

    tool_path = _resolve_tool_path(tool)
    try:
        md = _fetch_markdown(tool_path)
    except Exception:
        logger.warning("Unable to fetch docs for '%s'", tool_path)
        return
    content = _extract_section(md, section)
    content = _strip_html(content)
    display(Markdown(content))  # type: ignore[no-untyped-call]


def display_api_reference(tool: str, model: str, function_name: str | None = None) -> None:
    """Fetch a docs page and render one API model (Input/Config/Output) as a markdown table.

    Args:
        tool (str): Tool identifier — full path (``"structure-prediction/esmfold"``),
            tool name (``"esmfold"``), or run function (``"run_esmfold"``).
        model (str): One of ``"input"``, ``"config"``, or ``"output"``.
        function_name (str | None): Run function name for multi-function tools
            (e.g. ``"run_proteinmpnn_sample"``). Defaults to None for single-function tools.
    """
    from IPython.display import Markdown, display

    tool_path = _resolve_tool_path(tool)
    try:
        md = _fetch_markdown(tool_path)
    except Exception:
        logger.warning("Unable to fetch docs for '%s'", tool_path)
        return
    display(Markdown(_extract_api_model(md, model, function_name)))  # type: ignore[no-untyped-call]
