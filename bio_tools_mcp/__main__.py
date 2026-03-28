"""bio_tools_mcp/__main__.py

Entry point: python -m bio_tools_mcp."""

try:
    from bio_tools_mcp.server import main
except ImportError as exc:
    if "fastmcp" in str(exc):
        raise SystemExit(
            "MCP server requires fastmcp. Install with: pip install -e '.[mcp]'"
        ) from None
    raise

main()
