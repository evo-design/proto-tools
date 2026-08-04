"""MCP server exposing proto-tools to AI agents, over stdio."""

from typing import Any

__all__ = ["build_server", "main"]


def __getattr__(name: str) -> Any:
    """Resolve the server entry points lazily, so ``fastmcp`` is only needed to run one."""
    if name in __all__:
        from proto_tools.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
