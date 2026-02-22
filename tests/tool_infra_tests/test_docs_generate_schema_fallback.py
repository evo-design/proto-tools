"""Tests for docs schema extraction fallback behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from bio_programming_tools.tools.gene_annotation.blast.blast_search import BlastSearchOutput
from bio_programming_tools.tools.gene_annotation.pyhmmer.shared_data_models import PyHmmerOutput
from bio_programming_tools.tools.rna_splicing.splice_transformer.splice_transformer import SpliceTransformerOutput


def _load_docs_generator_module():
    """Load docs/generate_docs.py as a Python module for direct testing."""
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "docs" / "generate_docs.py"
    spec = importlib.util.spec_from_file_location("tool_docs_generate", module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field_map(fields: List[Dict]) -> Dict[str, Dict]:
    return {field["name"]: field for field in fields}


def test_parse_handles_dataframe_and_ndarray_with_field_introspection_fallback():
    docs_gen = _load_docs_generator_module()

    class MixedOutput(BaseModel):
        table: Optional[pd.DataFrame] = None
        prediction: np.ndarray
        model_config = ConfigDict(arbitrary_types_allowed=True)

    parsed = docs_gen.parse_pydantic_fields(MixedOutput)
    fields = _field_map(parsed)

    assert fields["table"]["type"] == "DataFrame"
    assert fields["prediction"]["type"].startswith("ndarray")


def test_parse_handles_nested_union_and_collection_annotations():
    docs_gen = _load_docs_generator_module()

    class ComplexConfig(BaseModel):
        ids: List[Union[int, str]]
        score_grid: Dict[str, Tuple[int, float]]
        model_config = ConfigDict(arbitrary_types_allowed=True)

    parsed = docs_gen.parse_pydantic_fields(ComplexConfig)
    fields = _field_map(parsed)

    assert fields["ids"]["type"] == "List[integer | string]"
    assert fields["score_grid"]["type"] == "Dict[string, Tuple[integer, number]]"


def test_real_output_models_parse_without_schema_crash():
    docs_gen = _load_docs_generator_module()
    exclude = docs_gen.BASE_OUTPUT_METADATA_FIELDS

    blast_fields = _field_map(docs_gen.parse_pydantic_fields(BlastSearchOutput, exclude=exclude))
    pyhmmer_fields = _field_map(docs_gen.parse_pydantic_fields(PyHmmerOutput, exclude=exclude))
    splice_fields = _field_map(docs_gen.parse_pydantic_fields(SpliceTransformerOutput, exclude=exclude))

    assert blast_fields["results_df"]["type"] == "DataFrame"
    assert pyhmmer_fields["sequence_hits_df"]["type"] == "DataFrame"
    assert pyhmmer_fields["domain_hits_df"]["type"] == "DataFrame"
    assert splice_fields["prediction"]["type"].startswith("ndarray")


def test_output_section_is_preserved_when_strip_output_disabled():
    docs_gen = _load_docs_generator_module()

    readme_text = """## Inputs\nfoo\n\n## Output Specification\nbar\n\n## Quick Start\nbaz\n"""
    preserved = docs_gen._strip_hand_written_schemas(
        readme_text,
        strip_input=True,
        strip_config=True,
        strip_output=False,
        strip_important=True,
    )

    assert "## Output Specification" in preserved
    assert "## Inputs" not in preserved
