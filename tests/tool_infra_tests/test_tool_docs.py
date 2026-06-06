"""tests/tool_infra_tests/test_tool_docs.py.

Tests for the programmatic README + Pydantic-model extractors in
``proto_tools.utils.tool_docs`` plus the parallel ``ToolRegistry`` wrappers.

Coverage:

- Canonical structure assertions on the ESM2 README (the reference
  implementation for the template).
- A parametrized smoke test that walks every toolkit README, verifies it parses
  through ``get_readme_sections``, and checks that every registered tool in the
  toolkit shows up as a ``ToolReadmeEntry``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.tool_docs import (
    FieldDoc,
    MetricSpecDoc,
    ModelDoc,
    ReadmeSections,
    _normalize_tool_key,
    get_example_notebook,
    get_model_doc,
    get_readme,
    get_readme_section,
    get_readme_sections,
    get_tool_docs,
)

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "tools"
_EXCLUDED_DIRS = frozenset({"__pycache__", "infra", "utils", "testing"})


def _all_tool_readmes() -> list[Path]:
    """Find every ``category/toolkit/README.md`` on disk."""
    readmes: list[Path] = []
    for readme in sorted(_TOOLS_DIR.rglob("README.md")):
        rel = readme.relative_to(_TOOLS_DIR)
        if len(rel.parts) != 3:
            continue
        category, toolkit, _ = rel.parts
        if category in _EXCLUDED_DIRS or toolkit in _EXCLUDED_DIRS:
            continue
        readmes.append(readme)
    return readmes


def _toolkit_id(readme: Path) -> str:
    """``category/toolkit`` identifier for a README path, with dashes."""
    rel = readme.relative_to(_TOOLS_DIR)
    return f"{rel.parts[0].replace('_', '-')}/{rel.parts[1].replace('_', '-')}"


_TOOL_READMES = _all_tool_readmes()
_TOOL_README_IDS = [_toolkit_id(r) for r in _TOOL_READMES]


# ── ESM2 reference structure ────────────────────────────────────────────────


def test_esm2_readme_sections_structure() -> None:
    """ESM2 is the canonical example and must satisfy the full structure."""
    sections = get_readme_sections("esm2-embedding")

    assert isinstance(sections, ReadmeSections)
    assert sections.title == "ESM2"
    assert sections.overview, "Overview body is empty"
    assert sections.background, "Background body is empty"
    assert sections.toolkit_notes, "Toolkit Notes body is empty"

    registry_keys = sorted(spec.key for spec in ToolRegistry.list_all() if spec.source_file.parent.name == "esm2")
    parsed_keys = sorted(t.key for t in sections.tools)
    assert parsed_keys == registry_keys, f"parsed={parsed_keys} registry={registry_keys}"

    for entry in sections.tools:
        assert entry.intro, f"{entry.key}: intro is empty"
        assert entry.applications, f"{entry.key}: applications is empty"
        assert entry.usage_tips, f"{entry.key}: usage_tips is empty"


def test_get_tool_docs_includes_toolkit_notes_by_default() -> None:
    """``get_tool_docs`` attaches the toolkit-wide notes by default."""
    entry = get_tool_docs("esm2-embedding")
    assert entry is not None
    assert entry.key == "esm2-embedding"
    assert entry.label == "ESM2 Embeddings"
    assert entry.toolkit_notes, "Toolkit notes should be attached by default"


def test_get_tool_docs_omits_toolkit_notes_when_disabled() -> None:
    """``include_toolkit_notes=False`` leaves the field as None."""
    entry = get_tool_docs("esm2-embedding", include_toolkit_notes=False)
    assert entry is not None
    assert entry.toolkit_notes is None


def test_get_tool_docs_includes_license_by_default() -> None:
    """``get_tool_docs`` attaches the parsed license.yaml, incl. weights.access."""
    entry = get_tool_docs("esm3-embedding")
    assert entry is not None
    assert entry.license is not None
    assert entry.license["code"]["spdx"] == "Custom (Cambrian Open License Agreement)"
    assert entry.license["weights"]["access"] == "hf-gated"


def test_get_tool_docs_omits_license_when_disabled() -> None:
    """``include_license=False`` leaves the field as None."""
    entry = get_tool_docs("esm2-embedding", include_license=False)
    assert entry is not None
    assert entry.license is None


def test_get_tool_docs_returns_none_for_unknown_key_in_polished_readme() -> None:
    """Asking for a key not present in the README returns None, not an error."""
    # A bogus key resolves the toolkit but matches no H3, so the entry is absent.
    sections = get_readme_sections("esm2-embedding")
    keys_in_readme = {t.key for t in sections.tools}
    assert "esm2-nonexistent" not in keys_in_readme


def test_get_readme_section_round_trip() -> None:
    """``get_readme_section`` returns the same body as the structured field."""
    overview_via_section = get_readme_section("esm2-embedding", "Overview")
    overview_via_struct = get_readme_sections("esm2-embedding").overview
    assert overview_via_section == overview_via_struct


def test_get_readme_section_returns_none_when_missing() -> None:
    """A non-existent heading returns None."""
    assert get_readme_section("esm2-embedding", "Nonexistent Section") is None


_LEARNING_RESOURCES_URL = "https://hazyresearch.stanford.edu/blog/2024-03-14-evo"


def test_background_excludes_learning_resources_by_default() -> None:
    """The optional '### Learning Resources' subsection is hidden from agents by default."""
    background = get_readme_sections("evo1-sample").background
    assert background, "Background body is empty"
    assert "Learning Resources" not in background
    assert _LEARNING_RESOURCES_URL not in background

    section = get_readme_section("evo1-sample", "Background")
    assert section is not None
    assert "Learning Resources" not in section
    assert _LEARNING_RESOURCES_URL not in section


def test_background_includes_learning_resources_when_requested() -> None:
    """include_learning_resources=True keeps the subsection in the Background body."""
    background = get_readme_sections("evo1-sample", include_learning_resources=True).background
    assert "Learning Resources" in background
    assert _LEARNING_RESOURCES_URL in background

    section = get_readme_section("evo1-sample", "Background", include_learning_resources=True)
    assert section is not None
    assert "Learning Resources" in section
    assert _LEARNING_RESOURCES_URL in section


# ── Pydantic model docs ─────────────────────────────────────────────────────


def test_get_model_doc_returns_normalized_view() -> None:
    """``get_model_doc`` returns name, docstring, and field rows."""
    spec = ToolRegistry.get("esm2-embedding")
    doc = get_model_doc(spec.config_model)

    assert isinstance(doc, ModelDoc)
    assert doc.name == spec.config_model.__name__
    assert doc.docstring, "Config docstring should not be empty"

    names = {f.name for f in doc.fields}
    assert "model_checkpoint" in names
    assert "return_logits" in names
    assert "repr_layer" in names

    return_logits = next(f for f in doc.fields if f.name == "return_logits")
    assert isinstance(return_logits, FieldDoc)
    assert return_logits.required is False
    assert return_logits.default is False
    assert return_logits.description


def test_get_output_doc_nests_metric_specs() -> None:
    """``get_output_doc`` nests ``MetricSpecDoc`` (with primary + per-item field) by default."""
    doc = ToolRegistry.get_output_doc("chai1-prediction")
    assert doc.metric_specs is not None
    assert all(isinstance(m, MetricSpecDoc) for m in doc.metric_specs)
    assert doc.primary_metric == "avg_plddt"
    assert doc.metrics_per_item_field == "structures"

    avg_plddt = next(m for m in doc.metric_specs if m.name == "avg_plddt")
    assert (avg_plddt.type_str, avg_plddt.min, avg_plddt.max) == ("float", 0.0, 1.0)
    assert avg_plddt.availability == "always"
    assert avg_plddt.better_values_are == "higher"
    assert avg_plddt.is_primary is True
    assert sum(m.is_primary for m in doc.metric_specs) == 1

    # A non-primary metric also carries its direction (field propagates to every spec).
    avg_pae = next(m for m in doc.metric_specs if m.name == "avg_pae")
    assert avg_pae.better_values_are == "lower"
    assert avg_pae.is_primary is False

    # A tool with no registered metrics_class has no nested specs.
    assert ToolRegistry.get_output_doc("evo1-sample").metric_specs is None


def test_get_input_doc_excludes_output_metadata_fields() -> None:
    """Output-side metadata fields don't leak into input docs."""
    doc = ToolRegistry.get_input_doc("esm2-embedding")
    metadata_fields = {"tool_id", "execution_time", "success", "errors"}
    leaked = {f.name for f in doc.fields} & metadata_fields
    assert not leaked, f"Leaked output-metadata fields: {leaked}"


# ── Badge linkification ─────────────────────────────────────────────────────


def test_extracted_toolkit_notes_have_no_badge_html() -> None:
    """Badge <a><img></a> HTML is collapsed to markdown links in extracted text."""
    entry = get_tool_docs("esm2-embedding")
    assert entry is not None
    notes = entry.toolkit_notes or ""
    assert "<img" not in notes
    assert "shields.io" not in notes
    # The four guide links survive as markdown links.
    assert "[Tool Persistence guide](https://bio-pro.mintlify.app/tools/guides/tool-persistence)" in notes
    assert "[Cloud Inference guide](https://bio-pro.mintlify.app/tools/guides/cloud-inference)" in notes


def test_raw_get_readme_preserves_badge_html() -> None:
    """``get_readme`` is a faithful dump — badge HTML is left intact."""
    raw = get_readme("esm2-embedding")
    assert "shields.io" in raw
    assert "<img" in raw


# ── Flexible identifier resolution ──────────────────────────────────────────


def test_normalize_tool_key_accepts_registry_key() -> None:
    """The registry key itself round-trips through the normalizer."""
    assert _normalize_tool_key("esm2-embedding") == "esm2-embedding"


def test_normalize_tool_key_accepts_run_function_name() -> None:
    """Both ``run_xyz`` and ``xyz`` resolve to the matching registry key."""
    assert _normalize_tool_key("run_esm2_embeddings") == "esm2-embedding"
    assert _normalize_tool_key("esm2_embeddings") == "esm2-embedding"


def test_normalize_tool_key_rejects_multi_tool_toolkit() -> None:
    """A toolkit name with multiple tools is ambiguous and must raise."""
    with pytest.raises(ValueError, match="ambiguous"):
        _normalize_tool_key("esm2")


def test_normalize_tool_key_rejects_unknown_identifier() -> None:
    """Identifiers that don't resolve raise rather than returning silently."""
    with pytest.raises(ValueError, match="Could not resolve"):
        _normalize_tool_key("not-a-real-tool-name")


def test_get_tool_docs_accepts_run_function_name() -> None:
    """Per-tool extractor works when called with the run-function name."""
    entry = get_tool_docs("run_esm2_embeddings")
    assert entry is not None
    assert entry.key == "esm2-embedding"


def test_get_tool_docs_raises_on_ambiguous_toolkit_name() -> None:
    """Per-tool extractor surfaces the ambiguous-toolkit error."""
    with pytest.raises(ValueError, match="ambiguous"):
        get_tool_docs("esm2")


def test_registry_model_doc_methods_accept_run_function_name() -> None:
    """``get_input_doc`` / ``get_config_doc`` / ``get_output_doc`` accept run names."""
    via_key = ToolRegistry.get_input_doc("esm2-embedding")
    via_func = ToolRegistry.get_input_doc("run_esm2_embeddings")
    assert via_key == via_func


# ── Catalog / category lookup ───────────────────────────────────────────────


def test_list_categories_is_sorted_and_unique() -> None:
    """``list_categories`` returns sorted, deduplicated category names."""
    categories = ToolRegistry.list_categories()
    assert categories == sorted(set(categories))
    assert "masked_models" in categories


def test_list_by_category_returns_keyed_specs() -> None:
    """``list_by_category`` returns every tool whose ``spec.category`` matches."""
    specs = ToolRegistry.list_by_category("masked_models")
    keys = [s.key for s in specs]
    assert keys == sorted(keys), "Expected key-sorted output"
    assert {"esm2-embedding", "esm2-sample", "esm2-score", "esm2-gradient"}.issubset(keys)
    assert all(s.category == "masked_models" for s in specs)


def test_list_by_category_unknown_returns_empty() -> None:
    """Unknown category resolves to an empty list rather than raising."""
    assert ToolRegistry.list_by_category("not-a-real-category") == []


def test_catalog_groups_every_tool() -> None:
    """``catalog`` partitions every registered tool by category."""
    cat = ToolRegistry.catalog()
    total_in_catalog = sum(len(v) for v in cat.values())
    assert total_in_catalog == len(ToolRegistry.list_all())
    assert set(cat.keys()) == set(ToolRegistry.list_categories())


def test_registry_wrappers_delegate_correctly() -> None:
    """``ToolRegistry`` methods are thin wrappers over ``tool_docs``."""
    assert ToolRegistry.get_readme("esm2-embedding") == get_readme("esm2-embedding")
    assert ToolRegistry.get_readme_sections("esm2-embedding") == get_readme_sections("esm2-embedding")
    assert ToolRegistry.get_tool_docs("esm2-embedding") == get_tool_docs("esm2-embedding")


# ── Parametrized smoke test on all tool READMEs ─────────────────────────────


@pytest.mark.parametrize("readme", _TOOL_READMES, ids=_TOOL_README_IDS)
def test_tool_readme_parses_cleanly(readme: Path) -> None:
    """Every tool README round-trips through ``get_readme_sections``.

    Asserts the parser doesn't raise, the canonical H2s are populated, and
    every registered tool in the toolkit shows up as a ``ToolReadmeEntry``.
    """
    toolkit_dir = readme.parent
    sections = get_readme_sections(toolkit_dir.name)

    assert isinstance(sections, ReadmeSections)
    assert sections.title, "H1 is empty"
    assert sections.overview, "Overview body is empty"
    assert sections.background, "Background body is empty"

    registry_keys = sorted(spec.key for spec in ToolRegistry.list_all() if spec.source_file.parent == toolkit_dir)
    parsed_keys = sorted(t.key for t in sections.tools)
    assert parsed_keys == registry_keys, f"{_toolkit_id(readme)}: parsed={parsed_keys} registry={registry_keys}"
    for entry in sections.tools:
        assert entry.intro, f"{entry.key}: intro paragraph is empty"


# ── Example notebook extraction ─────────────────────────────────────────────


def test_get_example_notebook_renders_markdown_and_code_fences() -> None:
    """A polished toolkit's example.ipynb renders as markdown prose + fenced code."""
    rendered = get_example_notebook("esm2-embedding")
    assert isinstance(rendered, str)
    assert rendered.startswith("# example notebook:")
    assert "examples/example.ipynb" in rendered
    assert "```python" in rendered


def test_get_example_notebook_returns_none_when_missing() -> None:
    """Toolkits without examples/example.ipynb return None rather than raising."""
    assert get_example_notebook("mmseqs2-clustering") is None
