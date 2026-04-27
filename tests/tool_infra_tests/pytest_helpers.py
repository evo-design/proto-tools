"""Shared test helpers for tool infrastructure tests."""

import re
from pathlib import Path
from typing import Any

from proto_tools.tools.tool_registry import ToolSpec

EXCLUDED_CATEGORIES: frozenset[str] = frozenset({"database_retrieval"})

# Toolkits (directory names under ``proto_tools/tools/.../{toolkit}/``) whose
# parametrized smoke tests should be bypassed on CI runners. Every tool in
# these toolkits gets ``skip_ci`` applied at parametrize time.
SKIP_CI_TOOLKITS: frozenset[str] = frozenset({"mmseqs2_homology_search"})


def parse_min_gpu_count(device_count: str) -> int:
    """Parse minimum GPU count from a device_count spec (e.g. '1', '2', '1-2', '>=1', '>1')."""
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


def build_inputs_and_config(
    spec: ToolSpec,
    tmp_path: Path,
    extra_config_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Build example inputs and a minimal-cost config for a tool.

    Uses ``Config.minimal()`` to apply tool-specific cheap-mode defaults
    (e.g. MSA generation disabled for structure predictors). Tool-specific
    test plumbing overrides (like output path redirection) are applied here.

    Args:
        spec (ToolSpec): The tool specification from the registry.
        tmp_path (Path): Temporary directory for file outputs.
        extra_config_kwargs (dict[str, Any] | None): Additional keyword arguments
            to pass to the config constructor (e.g. {"seed": 42}, {"verbose": True}).

    Returns:
        tuple[Any, Any]: (inputs, config) ready for spec.function().
    """
    inputs = spec.example_input()

    config_kwargs: dict[str, Any] = dict(extra_config_kwargs or {})

    if spec.key == "blast-create-db":
        config_kwargs.setdefault("out_prefix", str(tmp_path / "blast_db"))

    config = spec.config_model.minimal(**config_kwargs)
    return inputs, config
