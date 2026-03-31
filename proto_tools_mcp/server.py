"""
mcp_server/server.py

Exposes the tool registry (discovery, schemas, citations, search)
to any MCP-compatible AI client via the FastMCP SDK.

Usage:
    # stdio (Claude Desktop / Claude Code)
    python -m mcp_server

    # HTTP (remote access)
    python -m mcp_server --transport http --port 9200

    # Or via FastMCP CLI
    fastmcp run mcp_server/server.py:mcp
"""

from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP(
    name="proto-tools",
    instructions=(
        "Proto-tools: a library of bioinformatics tool wrappers organized "
        "by category. Use the discovery tools to browse available tools, search by "
        "keyword, inspect schemas, and read citations. This server is read-only — "
        "it helps you find and understand tools so you can call them from Python."
    ),
)

# Register tools, resources, and prompts by importing their modules.
import mcp_server.prompts  # noqa: F401, E402
import mcp_server.resources  # noqa: F401, E402
import mcp_server.tools  # noqa: F401, E402


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Proto-tools MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=9200, help="HTTP port (default: 9200)"
    )
    args = parser.parse_args()

    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
