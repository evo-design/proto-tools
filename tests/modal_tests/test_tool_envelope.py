"""The worker envelope returns results made only of JSON-native values."""

from __future__ import annotations

import json
from typing import Any

from proto_tools.modal.utils import apply_tool_envelope, strip_base_fields
from proto_tools.tools.sequence_scoring.alphagenome.alphagenome_score_variants import (
    AlphaGenomeScoreVariantsOutput,
)
from proto_tools.tools.sequence_scoring.alphagenome.shared_data_models import AlphaGenomeScoreOutput

_JSON_LEAF_TYPES = (str, int, float, bool, type(None))


def _non_json_paths(node: Any, path: str = "$") -> list[str]:
    """Return the path and type of every value in *node* that is not a JSON-native type."""
    if isinstance(node, dict):
        return [p for key, value in node.items() for p in _non_json_paths(value, f"{path}.{key}")]
    if isinstance(node, list):
        return [p for i, value in enumerate(node) for p in _non_json_paths(value, f"{path}[{i}]")]
    return [] if isinstance(node, _JSON_LEAF_TYPES) else [f"{path}: {type(node).__name__}"]


def _nested_output() -> AlphaGenomeScoreVariantsOutput:
    """A real output whose ``results`` hold ``BaseToolOutput`` children, each with its own ``timestamp``."""
    return AlphaGenomeScoreVariantsOutput(
        results=[AlphaGenomeScoreOutput(scores=[{"variant_id": "chr22:36201698:A>C", "raw_score": 0.5}])]
    )


def test_envelope_result_holds_only_json_native_values() -> None:
    result = apply_tool_envelope(_nested_output)

    assert _non_json_paths(result) == []
    json.dumps(result)


def test_envelope_result_matches_pydantic_json_mode() -> None:
    output = _nested_output()

    result = apply_tool_envelope(lambda: output)

    assert result == strip_base_fields(output.model_dump(mode="json"))
    assert AlphaGenomeScoreVariantsOutput.model_validate(result).results == output.results
