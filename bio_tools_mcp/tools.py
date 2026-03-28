"""bio_tools_mcp/tools.py

MCP tools — discovery, search, schemas, and citations."""

from __future__ import annotations

from typing import Any

from bio_programming_tools.tools.tool_registry import ToolRegistry

from bio_tools_mcp.server import mcp


def _spec_to_dict(spec) -> dict[str, Any]:
    """Convert a ToolSpec to a dict with source_file included."""
    data = spec.model_dump(exclude={"config_model"})
    data["source_file"] = str(spec.source_file)
    return data


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "List all registered bioinformatics tools with key, label, description, "
        "category, GPU requirement, and device count. Optionally filter by category."
    ),
    annotations={"readOnlyHint": True},
)
def list_tools(category: str | None = None) -> list[dict[str, Any]]:
    """List available tools. Optionally filter by category.

    Args:
        category (str | None): Filter by category (e.g. 'gene_annotation', 'structure_prediction').
    """

    specs = ToolRegistry.list_all()
    results = []
    for spec in specs:
        if category and spec.category != category:
            continue
        results.append(_spec_to_dict(spec))
    return results


@mcp.tool(
    description="List all tool categories with tool counts.",
    annotations={"readOnlyHint": True},
)
def list_categories() -> dict[str, list[str]]:
    """List all categories and the tool keys in each."""

    categories: dict[str, list[str]] = {}
    for spec in ToolRegistry.list_all():
        categories.setdefault(spec.category, []).append(spec.key)
    return {k: sorted(v) for k, v in sorted(categories.items())}


@mcp.tool(
    description="List tools that require a GPU.",
    annotations={"readOnlyHint": True},
)
def list_gpu_tools() -> list[dict[str, Any]]:
    """List all tools that require GPU execution."""

    return [_spec_to_dict(spec) for spec in ToolRegistry.list_gpu_tools()]


@mcp.tool(
    description="List tools that run on CPU only (no GPU required).",
    annotations={"readOnlyHint": True},
)
def list_cpu_tools() -> list[dict[str, Any]]:
    """List all CPU-only tools."""

    return [_spec_to_dict(spec) for spec in ToolRegistry.list_cpu_tools()]


# ---------------------------------------------------------------------------
# Schema tools
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Get the full JSON Schemas (input, config, output) for a tool. "
        "Returns property names, types, defaults, and descriptions."
    ),
    annotations={"readOnlyHint": True},
)
def get_tool_schema(key: str) -> dict[str, Any]:
    """Get input, config, and output JSON schemas for a tool.

    Args:
        key (str): Tool registry key (e.g. 'blast-search', 'esmfold-prediction').
    """

    return ToolRegistry.get_schemas(key)


@mcp.tool(
    description="Get an example minimal input for a tool, if available.",
    annotations={"readOnlyHint": True},
)
def get_tool_example(key: str) -> dict[str, Any] | None:
    """Get an example input instance for a tool.

    Args:
        key (str): Tool registry key (e.g. 'blast-search').
    """

    example = ToolRegistry.get_example_input(key)
    if example is None:
        return None
    return example.model_dump()


# ---------------------------------------------------------------------------
# Citation tools
# ---------------------------------------------------------------------------


@mcp.tool(
    description="Get the BibTeX citation for a tool.",
    annotations={"readOnlyHint": True},
)
def get_tool_citation(key: str) -> str | None:
    """Get BibTeX citation for a tool.

    Args:
        key (str): Tool registry key (e.g. 'evo2-sample').
    """

    return ToolRegistry.get_citation(key)


@mcp.tool(
    description="List all available BibTeX citations as {tool_key: bibtex}.",
    annotations={"readOnlyHint": True},
)
def list_citations() -> dict[str, str]:
    """Get all tool citations."""

    return ToolRegistry.list_citations()


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------


@mcp.tool(
    description=(
        "Search tools by keyword across key, label, description, and category. "
        "Returns matching tools ranked by relevance."
    ),
    annotations={"readOnlyHint": True},
)
def search_tools(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search tools by keyword.

    Args:
        query (str): Search query (keywords, tool name, or category).
        max_results (int): Maximum results to return.
    """

    query_lower = query.lower()
    query_terms = query_lower.split()

    scored = []
    for spec in ToolRegistry.list_all():
        score = 0
        key_lower = spec.key.lower()
        label_lower = spec.label.lower()
        desc_lower = spec.description.lower()
        cat_lower = spec.category.lower()

        for term in query_terms:
            # Exact key match is highest signal
            if term == key_lower:
                score += 10
            elif term in key_lower:
                score += 5
            # Label match
            if term in label_lower:
                score += 3
            # Category match
            if term in cat_lower:
                score += 3
            # Description match
            if term in desc_lower:
                score += 1

        if score > 0:
            scored.append((score, spec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [_spec_to_dict(spec) for _, spec in scored[:max_results]]
