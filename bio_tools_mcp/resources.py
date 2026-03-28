"""bio_tools_mcp/resources.py

MCP resources — tool documentation, schemas, and citations as URI templates."""

from __future__ import annotations

import json

from bio_programming_tools.tools.tool_registry import ToolRegistry

from bio_tools_mcp.server import mcp


# ---------------------------------------------------------------------------
# Tool documentation resource
# ---------------------------------------------------------------------------


@mcp.resource(
    "bio-tools://tools/{key}",
    description="Full metadata and description for a registered tool.",
    mime_type="text/markdown",
)
def tool_doc(key: str) -> str:
    """Get tool documentation by key."""

    spec = ToolRegistry.get(key)
    lines = [
        f"# {spec.label}",
        "",
        f"**Key:** `{spec.key}`",
        f"**Category:** {spec.category}",
        f"**GPU required:** {'Yes' if spec.uses_gpu else 'No'}",
        f"**Device count:** {spec.device_count}",
        f"**Source file:** `{spec.source_file}`",
        "",
        "## Description",
        "",
        spec.description,
    ]

    # Add schemas
    schemas = ToolRegistry.get_schemas(key)
    for schema_name, schema in schemas.items():
        lines.extend([
            "",
            f"## {schema_name.title()} Schema",
            "",
            "```json",
            json.dumps(schema, indent=2),
            "```",
        ])

    # Add example input if available
    example = ToolRegistry.get_example_input(key)
    if example is not None:
        lines.extend([
            "",
            "## Example Input",
            "",
            "```json",
            json.dumps(example.model_dump(), indent=2, default=str),
            "```",
        ])

    # Add citation if available
    citation = ToolRegistry.get_citation(key)
    if citation:
        lines.extend([
            "",
            "## Citation",
            "",
            "```bibtex",
            citation,
            "```",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema resource
# ---------------------------------------------------------------------------


@mcp.resource(
    "bio-tools://schemas/{key}",
    description="Input, config, and output JSON schemas for a tool.",
    mime_type="application/json",
)
def tool_schemas(key: str) -> str:
    """Get full JSON schemas for a tool."""

    schemas = ToolRegistry.get_schemas(key)
    return json.dumps(schemas, indent=2)


# ---------------------------------------------------------------------------
# Citation resource
# ---------------------------------------------------------------------------


@mcp.resource(
    "bio-tools://citations/{key}",
    description="BibTeX citation for a tool.",
    mime_type="text/plain",
)
def tool_citation(key: str) -> str:
    """Get BibTeX citation for a tool."""

    citation = ToolRegistry.get_citation(key)
    if citation is None:
        raise ValueError(
            f"No citation found for tool '{key}'. "
            f"Not all tools have cite.bib files."
        )
    return citation


# ---------------------------------------------------------------------------
# Example notebook resource
# ---------------------------------------------------------------------------


@mcp.resource(
    "bio-tools://examples/{key}",
    description="Path to the example Jupyter notebook for a tool, plus its code cells.",
    mime_type="text/markdown",
)
def tool_example(key: str) -> str:
    """Get example notebook content for a tool.

    Args:
        key (str): Tool registry key (e.g., ``"esmfold-prediction"``).

    Returns the notebook path and extracted code/markdown cells.
    """

    spec = ToolRegistry.get(key)
    tool_dir = spec.source_file.parent
    examples_dir = tool_dir / "examples"
    notebook_path = examples_dir / "example.ipynb"

    if not notebook_path.is_file():
        raise ValueError(
            f"No example notebook found for tool '{key}'. "
            f"Expected at: {notebook_path}"
        )

    notebook = json.loads(notebook_path.read_text())
    lines = [
        f"# Example: {spec.label}",
        "",
        f"**Notebook path:** `{notebook_path}`",
        "",
    ]

    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        if cell_type == "code":
            lines.extend(["```python", source, "```", ""])
        elif cell_type == "markdown":
            lines.extend([source, ""])

    return "\n".join(lines)
