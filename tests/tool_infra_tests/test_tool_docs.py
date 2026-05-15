"""tests/tool_infra_tests/test_tool_docs.py.

Tests for the programmatic README + Pydantic-model extractors in
``proto_tools.utils.tool_docs`` plus the parallel ``ToolRegistry`` wrappers.

Coverage:

- Canonical structure assertions on the ESM2 README (the reference
  implementation for the new template; see evo-design/proto-tools#783).
- A parametrized smoke test that walks every README which has already
  been quality-checked (no QC-pending callout) and verifies it parses
  through ``get_readme_sections`` and that every registered tool in the
  toolkit shows up as a ``ToolReadmeEntry``.

READMEs that still carry the ``> [!NOTE] **TODO:** This README still
needs to be reviewed`` callout are skipped — those have not been
migrated to the new structured template yet (issue #743 tracks the
rewrite). The smoke test will pick them up automatically as each one
is migrated and the callout is removed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from proto_tools.tools.tool_registry import ToolRegistry
from proto_tools.utils.tool_docs import (
    FieldDoc,
    ModelDoc,
    ReadmeSections,
    _normalize_tool_key,
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


# TODO(#743): once every README is migrated, delete _QC_CALLOUT_RE, _polished_readmes,
# the _POLISHED* lists, and test_qc_pending_flag_reflects_callout; smoke-test all READMEs.
_QC_CALLOUT_RE = re.compile(
    r"^>\s*\[!NOTE\]\s*\n>\s*\*\*TODO:\*\*\s*This README still needs to be reviewed",
    re.MULTILINE,
)


def _polished_readmes() -> list[Path]:
    """Subset of READMEs that have been migrated (no QC-pending callout)."""
    return [r for r in _all_tool_readmes() if not _QC_CALLOUT_RE.search(r.read_text())]


def _toolkit_id(readme: Path) -> str:
    """``category/toolkit`` identifier for a README path, with dashes."""
    rel = readme.relative_to(_TOOLS_DIR)
    return f"{rel.parts[0].replace('_', '-')}/{rel.parts[1].replace('_', '-')}"


_POLISHED = _polished_readmes()
_POLISHED_IDS = [_toolkit_id(r) for r in _POLISHED]


# ── ESM2 reference structure ────────────────────────────────────────────────


def test_esm2_readme_sections_structure() -> None:
    """ESM2 is the canonical example and must satisfy the full structure."""
    sections = get_readme_sections("esm2-embedding")

    assert isinstance(sections, ReadmeSections)
    assert sections.title == "ESM2"
    assert sections.qc_pending is False
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


def test_qc_pending_flag_reflects_callout() -> None:
    """``qc_pending`` is True when the README still has the TODO callout."""
    assert get_readme_sections("esm2-embedding").qc_pending is False
    # Skip if no QC-pending READMEs remain (migration complete).
    pending = [r for r in _all_tool_readmes() if _QC_CALLOUT_RE.search(r.read_text())]
    if not pending:
        pytest.skip("No QC-pending READMEs remain; migration is complete")
    sample = pending[0]
    toolkit_id = _toolkit_id(sample)
    assert get_readme_sections(toolkit_id).qc_pending is True


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


# ── Parametrized smoke test on polished READMEs ─────────────────────────────


@pytest.mark.parametrize("readme", _POLISHED, ids=_POLISHED_IDS)
def test_polished_readme_parses_cleanly(readme: Path) -> None:
    """Every polished README round-trips through ``get_readme_sections``.

    Asserts the parser doesn't raise, the canonical H2s are populated, and
    every registered tool in the toolkit shows up as a ``ToolReadmeEntry``.
    """
    toolkit_dir = readme.parent
    sections = get_readme_sections(toolkit_dir.name)

    assert isinstance(sections, ReadmeSections)
    assert sections.title, "H1 is empty"
    assert sections.overview, "Overview body is empty"
    assert sections.background, "Background body is empty"
    assert sections.qc_pending is False

    registry_keys = sorted(spec.key for spec in ToolRegistry.list_all() if spec.source_file.parent == toolkit_dir)
    parsed_keys = sorted(t.key for t in sections.tools)
    assert parsed_keys == registry_keys, f"{_toolkit_id(readme)}: parsed={parsed_keys} registry={registry_keys}"
    for entry in sections.tools:
        assert entry.intro, f"{entry.key}: intro paragraph is empty"
