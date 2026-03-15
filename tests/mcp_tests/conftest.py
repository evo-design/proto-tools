"""Shared fixtures for MCP server tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def tool_registry():
    """Provide access to the populated ToolRegistry."""
    from bio_programming_tools.tools.tool_registry import ToolRegistry

    # Trigger tool registration by importing all tools
    import bio_programming_tools.tools  # noqa: F401

    return ToolRegistry


@pytest.fixture(scope="module")
def all_tool_keys(tool_registry):
    """List of all registered tool keys."""
    return [spec.key for spec in tool_registry.list_all()]


@pytest.fixture(scope="module")
def sample_tool_key(tool_registry):
    """A single tool key for targeted tests."""
    specs = tool_registry.list_all()
    assert len(specs) > 0, "No tools registered"
    return specs[0].key
