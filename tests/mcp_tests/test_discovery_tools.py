"""Tests for MCP discovery tools."""

from __future__ import annotations

from bio_tools_mcp.tools import (
    list_categories,
    list_cpu_tools,
    list_gpu_tools,
    list_tools,
)


# ── list_tools ──────────────────────────────────────────────────────────────


def test_list_tools_returns_all(tool_registry):
    result = list_tools()
    expected_count = tool_registry.count()
    assert len(result) == expected_count
    assert all(isinstance(t, dict) for t in result)


def test_list_tools_has_required_fields():
    result = list_tools()
    assert len(result) > 0
    required = {"key", "label", "category", "description", "uses_gpu", "device_count", "source_file"}
    for t in result:
        assert required.issubset(t.keys()), f"Missing fields in {t['key']}: {required - t.keys()}"


def test_list_tools_source_file_is_path():
    result = list_tools()
    for t in result:
        assert t["source_file"].endswith(".py"), (
            f"source_file for {t['key']} doesn't look like a .py path: {t['source_file']}"
        )


def test_list_tools_filter_by_category():
    all_tools = list_tools()
    categories = {t["category"] for t in all_tools}
    assert len(categories) > 1, "Expected multiple categories"

    for cat in categories:
        filtered = list_tools(category=cat)
        assert len(filtered) > 0
        assert all(t["category"] == cat for t in filtered)


def test_list_tools_invalid_category_returns_empty():
    result = list_tools(category="nonexistent_category_xyz")
    assert result == []


# ── list_categories ─────────────────────────────────────────────────────────


def test_list_categories_returns_dict():
    result = list_categories()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_list_categories_values_are_sorted():
    result = list_categories()
    for cat, keys in result.items():
        assert keys == sorted(keys), f"Keys not sorted in category {cat}"


def test_list_categories_covers_all_tools(tool_registry):
    result = list_categories()
    all_keys = {spec.key for spec in tool_registry.list_all()}
    cat_keys = {key for keys in result.values() for key in keys}
    assert cat_keys == all_keys


# ── GPU/CPU filtering ──────────────────────────────────────────────────────


def test_gpu_tools_all_require_gpu():
    result = list_gpu_tools()
    for t in result:
        assert t["uses_gpu"] is True, f"{t['key']} marked as GPU but uses_gpu=False"


def test_cpu_tools_none_require_gpu():
    result = list_cpu_tools()
    for t in result:
        assert t["uses_gpu"] is False, f"{t['key']} marked as CPU but uses_gpu=True"


def test_gpu_plus_cpu_equals_all(tool_registry):
    gpu = list_gpu_tools()
    cpu = list_cpu_tools()
    assert len(gpu) + len(cpu) == tool_registry.count()
