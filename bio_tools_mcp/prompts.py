"""MCP prompts — reusable workflow templates for agents."""

from __future__ import annotations

from bio_tools_mcp.server import mcp


@mcp.prompt(
    description=(
        "Find the right bioinformatics tool for a task. Walks through "
        "category listing, keyword search, and schema inspection."
    ),
)
def find_tool(task: str) -> str:
    """Help an agent find the right tool for a biological task.

    Args:
        task: What the user wants to do (e.g. "predict protein structure",
              "score a DNA sequence", "find homologs").
    """
    return f"""\
I need to find a bioinformatics tool for this task: {task}

Follow these steps:
1. Call list_categories() to see all tool categories and their tool keys.
2. Identify which category is most relevant to the task.
3. Call search_tools("{task}") to find matching tools ranked by relevance.
4. For the top 1-2 candidates, call get_tool_schema(key) to inspect their \
input/config/output schemas.
5. Call get_tool_example(key) to see a minimal working input.
6. Recommend the best tool with a brief explanation of why it fits and \
show example Python code to call it."""


@mcp.prompt(
    description=(
        "Get a complete walkthrough of a specific tool: metadata, schemas, "
        "example input, and citation — all in one context dump."
    ),
)
def tool_walkthrough(tool_key: str) -> str:
    """Pull together everything an agent needs to use a specific tool.

    Args:
        tool_key: Tool registry key (e.g. 'esmfold-prediction', 'blast-search').
    """
    return f"""\
Give me a complete walkthrough of the tool "{tool_key}".

Gather this information:
1. Call get_tool_schema("{tool_key}") for the full input/config/output JSON schemas.
2. Call get_tool_example("{tool_key}") for a minimal working input.
3. Call get_tool_citation("{tool_key}") for the paper reference.

Then present:
- What the tool does and when to use it
- Required vs optional inputs (from the schema)
- Key config parameters and their defaults
- Example Python code using the tool
- The citation for reference"""
