"""tests/tool_infra_tests/test_env_report.py.

Auto-discovered environment report smoke tests.

One test per standalone tool directory. Tools in the same directory share an
environment, so testing one proves the env works.
Only runs with ``pytest --env-report``; deselected during normal test runs.
"""

from pathlib import Path

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from proto_tools.utils.tool_instance import ToolInstance
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.pytest_helpers import (
    SKIP_CI_TOOLKITS,
    build_inputs_and_config,
    parse_min_gpu_count,
)

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "tools"


# ============================================================================
# Parametrization
# ============================================================================
def _tool_dir_id(tool_dir: Path) -> str:
    """Return a stable ``category/toolkit`` identifier for failure messages."""
    rel = tool_dir.relative_to(_TOOLS_DIR)
    return f"{rel.parts[0]}/{rel.parts[1]}"


def _build_tool_params() -> list:
    """Build pytest parametrize params from ToolRegistry.

    One test per standalone tool directory — tools sharing a directory share a
    standalone environment, so only the first tool (by key) per directory is
    tested.

    Marks applied per-param:
    - ``include_in_env_report(tool=..., category=...)`` for report collection
    - ``uses_gpu`` / ``uses_gpu(n)`` based on device_count
    """
    params = []
    seen_dirs: set = set()

    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        if not spec.has_standalone_env:
            continue
        if spec.example_input is None:
            continue

        # One test per standalone tool directory.
        tool_dir = spec.source_file.parent
        if tool_dir in seen_dirs:
            continue
        seen_dirs.add(tool_dir)

        marks = [
            pytest.mark.include_in_env_report(
                tool=spec.key,
                category=spec.category,
            ),
        ]

        if spec.uses_gpu:
            gpu_count = parse_min_gpu_count(spec.device_count)
            marks.append(pytest.mark.uses_gpu(gpu_count))

        if spec.source_file.parent.name in SKIP_CI_TOOLKITS:
            marks.append(pytest.mark.skip_ci)

        params.append(pytest.param(spec, id=spec.key, marks=marks))

    return params


def test_env_report_params_cover_every_standalone_tool_dir() -> None:
    """Every registered standalone tool directory has one env-report smoke test."""
    params = _build_tool_params()
    covered_specs = [param.values[0] for param in params]
    covered_dirs = {spec.source_file.parent for spec in covered_specs}
    standalone_dirs = {spec.source_file.parent for spec in ToolRegistry.list_all() if spec.has_standalone_env}
    worker_dirs = set(ToolInstance._get_tool_dirs().values())

    unregistered_dirs = worker_dirs - standalone_dirs
    assert not unregistered_dirs, "standalone tool dir(s) with no registered ToolSpec: " + "; ".join(
        _tool_dir_id(tool_dir) for tool_dir in sorted(unregistered_dirs)
    )

    non_standalone = [spec.key for spec in covered_specs if not spec.has_standalone_env]
    assert not non_standalone, f"env-report selected non-standalone tools: {sorted(non_standalone)}"

    missing_dirs = standalone_dirs - covered_dirs
    if missing_dirs:
        specs_by_dir = {
            tool_dir: [spec for spec in ToolRegistry.list_all() if spec.source_file.parent == tool_dir]
            for tool_dir in sorted(missing_dirs)
        }
        details = [
            f"{_tool_dir_id(tool_dir)} ({', '.join(spec.key for spec in specs)})"
            for tool_dir, specs in specs_by_dir.items()
        ]
        pytest.fail("env-report is missing standalone tool dir(s): " + "; ".join(details))

    extra_dirs = covered_dirs - standalone_dirs
    assert not extra_dirs, f"env-report selected non-standalone dir(s): {sorted(map(str, extra_dirs))}"


# ============================================================================
# Test
# ============================================================================
@pytest.mark.parametrize("spec", _build_tool_params())
def test_tool_env_report(spec: ToolSpec, tmp_path):
    """Smoke-test a single tool: build env, run example_input, verify success."""
    inputs, config = build_inputs_and_config(spec, tmp_path, {"verbose": True})

    result = spec.function(inputs, config)
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
    assert_metrics_in_spec(result)
