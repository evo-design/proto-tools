"""
tests/tool_infra_tests/test_env_report.py

Auto-discovered environment report smoke tests.

One test per tool directory (standalone environment). Tools in the same
directory share an environment, so testing one proves the env works.
Only runs with ``pytest --env-report``; deselected during normal test runs.
"""
from __future__ import annotations

import pytest

from proto_tools.tools.structure_prediction.shared_data_models import (
    MSAStructurePredictionConfig,
)
from proto_tools.tools.tool_registry import ToolRegistry, ToolSpec

# ============================================================================
# Configuration
# ============================================================================
_EXCLUDED_CATEGORIES = {"database_retrieval"}
_CHIMERA_ONLY_KEYS = {"alphafold3-prediction"}


def _parse_min_gpu_count(device_count: str) -> int:
    """Parse minimum GPU count from a device_count spec (e.g. '1', '2', '1-2', '>=1', '>1')."""
    import re

    m = re.match(r">=(\d+)", device_count)
    if m:
        return int(m.group(1))
    m = re.match(r">(\d+)", device_count)
    if m:
        return int(m.group(1)) + 1
    m = re.match(r"(\d+)", device_count)
    if m:
        return int(m.group(1))
    return 1


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
    - ``only_chimera`` for tools that only run on the Chimera cluster
    """
    params = []
    seen_dirs: set = set()

    for spec in sorted(ToolRegistry.list_all(), key=lambda s: s.key):
        if spec.category in _EXCLUDED_CATEGORIES:
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
            gpu_count = _parse_min_gpu_count(spec.device_count)
            marks.append(pytest.mark.uses_gpu(gpu_count))

        if spec.key in _CHIMERA_ONLY_KEYS:
            marks.append(pytest.mark.only_chimera)

        params.append(pytest.param(spec, id=spec.key, marks=marks))

    return params


# ============================================================================
# Test
# ============================================================================
@pytest.mark.parametrize("spec", _build_tool_params())
def test_tool_env_report(spec: ToolSpec, tmp_path):
    """Smoke-test a single tool: build env, run example_input, verify success."""
    inputs = spec.example_input()

    # Build config with env-report-safe overrides
    config_kwargs = {"verbose": True}
    if issubclass(spec.config_model, MSAStructurePredictionConfig):
        config_kwargs["use_msa"] = False

    # blast-create-db writes output files relative to the input fasta;
    # redirect to a temp dir so database files don't pollute the repo.
    if spec.key == "blast-create-db":
        config_kwargs["out_prefix"] = str(tmp_path / "blast_db")

    # bioemu-sample always requires MSAs — load fixture A3M so preprocess
    # skips the remote ColabFold API call.
    if spec.key == "bioemu-sample":
        from proto_tools.tools.sequence_alignment.msas import MSA

        a3m_path = spec.source_file.parent / "examples" / "example.a3m"
        fixture_msa = MSA(str(a3m_path))
        sequence = inputs.complexes[0].chains[0].sequence
        inputs = inputs.model_copy(update={"msas": {sequence: fixture_msa}})

    config = spec.config_model(**config_kwargs)

    result = spec.function(inputs, config)
    assert result.success, f"Tool {spec.key} failed: {result.errors}"
