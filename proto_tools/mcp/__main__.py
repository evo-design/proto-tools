"""Entry point: ``python -m proto_tools.mcp``."""

import sys

try:
    from proto_tools.mcp.server import main
except ImportError as exc:  # fastmcp ships in the 'mcp' extra, so a plain install reaches here.
    print(f"error: the MCP server needs the 'mcp' extra ({exc}).", file=sys.stderr)  # noqa: T201
    print('Install it with: pip install "proto-tools[mcp]"', file=sys.stderr)  # noqa: T201
    raise SystemExit(2) from exc

if __name__ == "__main__":
    main()
