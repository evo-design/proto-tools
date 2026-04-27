"""tests/tool_infra_tests/test_env_report.py.

Auto-discovered environment report smoke tests.

One test per tool directory (standalone environment). Tools in the same
directory share an environment, so testing one proves the env works.
Only runs with ``pytest --env-report``; deselected during normal test runs.
"""

import pytest

from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec
from tests.tool_infra_tests._metric_helpers import assert_metrics_in_spec
from tests.tool_infra_tests.pytest_helpers import (
    EXCLUDED_CATEGORIES,
    SKIP_CI_TOOLKITS,
    build_inputs_and_config,
    parse_min_gpu_count,
)


# ============================================================================
# Parametrization
# ============================================================================
def _build_tool_params() -> list:
    """Build pytest parametrize params from ToolRegistry.

    One test per tool directory — tools sharing a directory share a standalone
    environment, so only the first tool (by key) per directory is tested.

    Marks applied per-param:
    - ``include_in_env_report(tool=..., category=...)`` for report collection
    - ``uses_gpu`` / ``uses_gpu(n)`` based on device_count
    """
    params = []
    seen_dirs: set = set()

    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        if spec.category in EXCLUDED_CATEGORIES:
            continue
        if spec.example_input is None:
            continue

        # One test per tool directory (= one per standalone env)
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
