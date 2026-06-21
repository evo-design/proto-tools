"""Programmatic access to tool READMEs and Pydantic model docs.

Each toolkit ships a ``README.md`` with a canonical three-section structure
(``## Overview``, ``## Background``, ``## Tools``) plus an optional toolkit-wide
``## Toolkit Notes`` section. Inside ``## Tools``, every registered tool is its
own H3 subsection with an intro paragraph, ``#### Applications``, and
``#### Usage Tips``. The structure is enforced by the README consistency tests
(see ``tests/style_consistency_tests/test_readme_consistency.py``).

This module exposes that structure to agents, MCP servers, and any other
non-notebook consumer that needs typed access to the prose. Notebook display
helpers in ``proto_tools.utils.notebook_docs`` are thin wrappers over these
extractors.

Three surfaces:

1. **README extraction** — ``get_readme`` / ``get_readme_sections`` /
   ``get_readme_section`` / ``get_tool_docs`` return raw or structured
   slices of a toolkit's README. ``get_tool_docs`` also attaches the parsed
   ``license.yaml`` by default (the human-facing License callout above
   ``## Overview`` is not a parsed section; structured license metadata is
   the agent-facing source of truth).
2. **Pydantic model docs** — ``get_model_doc`` returns a normalized view of
   a model's class docstring plus per-field name / type / default /
   description / required flag.
3. **Example notebooks** — ``get_example_notebook`` returns the toolkit's
   ``examples/example.ipynb`` rendered as a flat string (markdown prose
   interleaved with fenced ``python`` code blocks; outputs stripped) so
   agents can read end-to-end usage demos without parsing nbformat JSON.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, get_args

from docstring_parser import DocstringStyle
from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, Field

from proto_tools.utils.tool_io import Directionality, Metrics

logger = logging.getLogger(__name__)


# =============================================================================
# Public data models
# =============================================================================


class ToolReadmeEntry(BaseModel):
    """One tool's H3 subsection inside a toolkit README's ``## Tools``."""

    key: str = Field(description="Registry key, e.g. 'esm2-embedding'.")
    label: str = Field(description="Display label parsed from the H3 heading.")
    intro: str = Field(description="Paragraph between the H3 and the first H4.")
    applications: str | None = Field(
        default=None,
        description="Body of the '#### Applications' H4, if present.",
    )
    usage_tips: str | None = Field(
        default=None,
        description="Body of the '#### Usage Tips' H4, if present.",
    )
    toolkit_notes: str | None = Field(
        default=None,
        description=(
            "Body of the toolkit-level '## Toolkit Notes' section. Populated "
            "by ``get_tool_docs`` when ``include_toolkit_notes=True`` so "
            "agents see toolkit-wide guidance alongside per-tool tips."
        ),
    )
    license: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Parsed license.yaml (code/weights SPDX, commercial_use, "
            "attribution_required, weights.access). Populated by "
            "``get_tool_docs`` when ``include_license=True`` so agents can "
            "check gating and usage terms in the same call."
        ),
    )


class ReadmeSections(BaseModel):
    """Structured view of a toolkit README."""

    title: str = Field(description="Plain text of the README's H1 heading.")
    overview: str = Field(description="Body of the '## Overview' section.")
    background: str = Field(
        description=(
            "Body of the '## Background' section. The optional "
            "'### Learning Resources' subsection is excluded unless "
            "include_learning_resources=True was passed."
        )
    )
    tools: list[ToolReadmeEntry] = Field(
        default_factory=list,
        description="H3 subsections inside the '## Tools' section, one per registered tool.",
    )
    toolkit_notes: str = Field(
        default="",
        description="Body of the '## Toolkit Notes' section. Empty when the section is not present.",
    )
    other_sections: dict[str, str] = Field(
        default_factory=dict,
        description="Any non-canonical H2 sections, keyed by heading text.",
    )


class FieldDoc(BaseModel):
    """One field of a Pydantic model."""

    name: str = Field(description="Attribute name.")
    type_str: str = Field(description="Stringified type annotation.")
    default: Any | None = Field(
        default=None,
        description="Default value, or None when the field is required.",
    )
    description: str | None = Field(
        default=None,
        description="Terse per-field description (from ``Field(description=...)``).",
    )
    doc: str | None = Field(
        default=None,
        description="Full per-field documentation from the class docstring's ``Attributes:`` section.",
    )
    required: bool = Field(description="Whether the field is required (no default).")


class MetricSpecDoc(BaseModel):
    """One metric's declarative spec, from a ``Metrics`` subclass ``metric_spec``."""

    name: str = Field(description="Metric key (e.g. ``avg_plddt``).")
    type_str: str | None = Field(
        default=None,
        description="Stringified value type (e.g. ``float``, ``list[list[float]]``).",
    )
    min: float | None = Field(default=None, description="Minimum valid value; None if unbounded below.")
    max: float | None = Field(default=None, description="Maximum valid value; None if unbounded above.")
    unit: str | None = Field(default=None, description="Unit string (e.g. ``Å``, ``REU``, ``bits``).")
    availability: str | None = Field(
        default=None,
        description="When the metric is present (e.g. ``always``, ``depends on model output``).",
    )
    description: str | None = Field(default=None, description="Human-readable description of the metric.")
    better_values_are: Directionality | None = Field(
        default=None,
        description="Optimization direction: higher, lower, in-range, or context-dependent.",
    )
    is_primary: bool = Field(
        default=False,
        description="True if this is the model's ``primary_metric`` (headline value).",
    )


class ModelDoc(BaseModel):
    """Normalized view of a Pydantic model's docs."""

    name: str = Field(description="Class name.")
    docstring: str = Field(description="Cleaned class docstring.")
    fields: list[FieldDoc] = Field(default_factory=list, description="Per-field docs.")
    metric_specs: list[MetricSpecDoc] | None = Field(
        default=None,
        description=(
            "For tool output models with an associated Metrics subclass: the per-metric "
            "spec table (type/range/unit/availability). None when no Metrics is associated. "
            "When ``metrics_per_item_field`` is set, these describe the metrics carried by "
            "each item of that output list (one set per item), not the call as a whole."
        ),
    )
    primary_metric: str | None = Field(
        default=None,
        description="Name of the associated Metrics subclass's primary (headline) metric, if any.",
    )
    metrics_per_item_field: str | None = Field(
        default=None,
        description=(
            "Name of the iterable output field whose every item carries the metrics in "
            "``metric_specs`` (e.g. ``structures``: one metric set per predicted structure). "
            "None means the tool is not iterable, so the metrics describe the single output."
        ),
    )


# =============================================================================
# README extraction
# =============================================================================


_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# Canonical H2s become typed fields on ReadmeSections; everything else lands in other_sections.
_CANONICAL_H2: frozenset[str] = frozenset({"Overview", "Background", "Tools", "Toolkit Notes"})


def resolve_toolkit_dir(tool: str) -> Path:
    """Resolve a tool identifier to its on-disk toolkit directory.

    Accepts any of: docs-style path (``"structure-prediction/esmfold"``), tool
    directory name (``"esmfold"``), run function name (``"run_esmfold"``), or
    registry key (``"esmfold-prediction"``).

    Args:
        tool (str): Tool identifier in any supported format.

    Returns:
        Path: Absolute path to ``proto_tools/tools/{category}/{toolkit}``.

    Raises:
        ValueError: If no toolkit matches.
    """
    from proto_tools.tools.tool_registry import ToolRegistry

    if "/" in tool:
        category, toolkit = tool.split("/", 1)
        cat_us = category.replace("-", "_")
        tk_us = toolkit.replace("-", "_")
        for spec in ToolRegistry.list_all():
            p = spec.source_file.parent
            if p.name == tk_us and p.parent.name == cat_us:
                return p
        raise ValueError(f"Could not resolve toolkit dir for '{tool}'")

    func_prefix = tool if tool.startswith("run_") else f"run_{tool}"
    tool_normalized = tool.replace("-", "_").removeprefix("run_")

    for spec in ToolRegistry.list_all():
        p = spec.source_file.parent
        if (
            spec.key == tool
            or spec.function.__name__ == func_prefix
            or spec.function.__name__.startswith(func_prefix)
            or p.name == tool_normalized
        ):
            return p

    raise ValueError(f"Could not resolve toolkit for '{tool}'")


def toolkit_specs(tool: str) -> list[Any]:
    """Return the ToolSpecs whose source files live in the resolved toolkit dir."""
    from proto_tools.tools.tool_registry import ToolRegistry

    target = resolve_toolkit_dir(tool)
    return sorted(
        (s for s in ToolRegistry.list_all() if s.source_file.parent == target),
        key=lambda s: s.key,
    )


def _normalize_tool_key(identifier: str) -> str:
    """Resolve any tool identifier form to its unique registry key.

    Accepts:

    - A registered tool key, e.g. ``"esm2-embedding"``
    - A run-function name with or without the ``run_`` prefix, e.g.
      ``"run_esm2_embeddings"`` or ``"esm2_embeddings"``
    - Any identifier shape ``resolve_toolkit_dir`` accepts (docs path, toolkit
      directory name), *as long as it pinpoints exactly one registered
      tool* — a toolkit with multiple tools (e.g. ``"esm2"``) is rejected
      with a helpful error listing the candidates.

    Args:
        identifier (str): Any of the supported identifier forms.

    Returns:
        str: The matching tool's registry key.

    Raises:
        ValueError: If no tool matches, or if the identifier matches a
            multi-tool toolkit and is therefore ambiguous.
    """
    from proto_tools.tools.tool_registry import ToolRegistry

    specs = ToolRegistry.list_all()

    for spec in specs:
        if spec.key == identifier:
            return spec.key

    func_name = identifier if identifier.startswith("run_") else f"run_{identifier}"
    for spec in specs:
        if spec.function.__name__ == func_name:
            return spec.key

    try:
        target = resolve_toolkit_dir(identifier)
    except ValueError as exc:
        raise ValueError(f"Could not resolve tool identifier '{identifier}'") from exc

    candidates = sorted(s.key for s in specs if s.source_file.parent == target)
    if not candidates:
        raise ValueError(f"No tools registered under identifier '{identifier}'")
    if len(candidates) > 1:
        raise ValueError(
            f"Identifier '{identifier}' is ambiguous (resolves to a multi-tool toolkit); "
            f"specify one of: {', '.join(candidates)}"
        )
    return candidates[0]


def _read_readme(tool: str) -> str:
    """Read the toolkit's ``README.md`` from disk."""
    return (resolve_toolkit_dir(tool) / "README.md").read_text()


def _section_bounds(md: str) -> list[tuple[str, int, int]]:
    """Return ``(heading_text, body_start, body_end)`` for each H2 in order.

    ``body_start`` is the index immediately after the heading line's newline;
    ``body_end`` is the index of the next H2 (or end of string).
    """
    matches = list(_H2_RE.finditer(md))
    bounds: list[tuple[str, int, int]] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        if body_start < len(md) and md[body_start] == "\n":
            body_start += 1
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        bounds.append((m.group(1).strip(), body_start, body_end))
    return bounds


# Single shields.io badge anchor <a href="URL"><img ... alt="LABEL"></a>; <img> may be self-closed or not.
_BADGE_ANCHOR_RE = re.compile(
    r'<a\s+href="([^"]+)"\s*>\s*<img\b[^>]*?\balt="([^"]*)"[^>]*?/?>\s*</a>',
    re.IGNORECASE | re.DOTALL,
)


def _linkify_badges(text: str) -> str:
    """Replace shields.io ``<a><img></a>`` badge HTML with plain markdown links.

    Badge rows (e.g. the guide badges in a toolkit's ``## Toolkit Notes``)
    render correctly on GitHub and the docs site, but as *extracted text* for
    an agent or human they are unreadable HTML noise. Each badge anchor
    becomes ``[alt text](href)`` — standard markdown, agent-parseable, and
    faithful to the link the badge pointed at. The source README on disk is
    never modified; only the extracted/structured outputs are cleaned.
    """
    return _BADGE_ANCHOR_RE.sub(lambda m: f"[{m.group(2).strip()}]({m.group(1)})", text)


def _h2_body(md: str, heading: str) -> str:
    """Return the body of an H2 section by exact heading text, or empty string."""
    for h, start, end in _section_bounds(md):
        if h == heading:
            return _linkify_badges(md[start:end].strip())
    return ""


# Background may carry an optional `### Learning Resources` subsection: external
# explainers (blog posts, talks) aimed at human readers. It is excluded from
# agent-facing Background pulls by default; opt in with include_learning_resources=True.
_LEARNING_RESOURCES_HEADING = "Learning Resources"
_LEARNING_RESOURCES_RE = re.compile(
    rf"^###\s+{re.escape(_LEARNING_RESOURCES_HEADING)}\s*$.*?(?=^###\s|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _strip_learning_resources(background: str) -> str:
    """Remove the optional ``### Learning Resources`` subsection from a Background body."""
    return _LEARNING_RESOURCES_RE.sub("", background).strip()


_H3_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_H4_RE = re.compile(r"^####\s+(.+)$", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def _parse_tool_entries(tools_body: str) -> list[ToolReadmeEntry]:
    """Parse the body of the ``## Tools`` section into per-tool entries.

    Each H3 inside the section becomes one entry; ``#### Applications`` and
    ``#### Usage Tips`` H4s populate the matching fields. Anything else under
    the H3 (the intro paragraph, plus any other H4s) is concatenated into
    ``intro``.
    """
    h3_matches = list(_H3_RE.finditer(tools_body))
    entries: list[ToolReadmeEntry] = []

    for i, h3 in enumerate(h3_matches):
        h3_end = h3.end()
        block_end = h3_matches[i + 1].start() if i + 1 < len(h3_matches) else len(tools_body)
        block = tools_body[h3_end:block_end]

        heading = h3.group(1).strip()
        keys = _INLINE_CODE_RE.findall(heading)
        key = keys[0] if keys else heading
        label_match = re.match(r"^(.+?)\s*\(`[^`]+`\)\s*$", heading)
        label = label_match.group(1).strip() if label_match else re.sub(r"`[^`]+`", "", heading).strip()

        h4_matches = list(_H4_RE.finditer(block))
        intro_end = h4_matches[0].start() if h4_matches else len(block)
        intro = block[:intro_end].strip()

        applications: str | None = None
        usage_tips: str | None = None
        for j, h4 in enumerate(h4_matches):
            h4_end = h4.end()
            section_end = h4_matches[j + 1].start() if j + 1 < len(h4_matches) else len(block)
            body = block[h4_end:section_end].strip()
            name = h4.group(1).strip().lower()
            if name == "applications":
                applications = body
            elif name == "usage tips":
                usage_tips = body

        entries.append(
            ToolReadmeEntry(
                key=key,
                label=label,
                intro=intro,
                applications=applications,
                usage_tips=usage_tips,
            )
        )

    return entries


# =============================================================================
# Public README API
# =============================================================================


def get_readme(tool: str) -> str:
    """Return a toolkit's README text.

    Args:
        tool (str): Tool identifier (registry key, docs path, run-function
            name, or toolkit directory name).

    Returns:
        str: README content.

    Raises:
        ValueError: If ``tool`` doesn't resolve to a registered toolkit.
        OSError: If the README cannot be read from disk.
    """
    return _read_readme(tool)


def get_readme_section(tool: str, heading: str, *, include_learning_resources: bool = False) -> str | None:
    """Return one H2 section's body by exact heading text.

    Args:
        tool (str): Tool identifier — same forms as ``get_readme``.
        heading (str): H2 heading text (e.g. ``"Background"``).
        include_learning_resources (bool): When False (the default), the
            optional ``### Learning Resources`` subsection is stripped from the
            ``Background`` body; it holds human-facing external links that are
            noise for agents. Set True to keep it. No effect on other sections.

    Returns:
        str | None: The section body (without its heading), or None if no
            matching H2 is found.
    """
    body = _h2_body(get_readme(tool), heading)
    if heading == "Background" and not include_learning_resources:
        body = _strip_learning_resources(body)
    return body or None


def get_readme_sections(tool: str, *, include_learning_resources: bool = False) -> ReadmeSections:
    """Return a toolkit's README parsed into a typed structure.

    Args:
        tool (str): Tool identifier — same forms as ``get_readme``.
        include_learning_resources (bool): When False (the default), the
            optional ``### Learning Resources`` subsection is stripped from
            ``background``; it holds human-facing external links that are noise
            for agents. Set True to keep it.

    Returns:
        ReadmeSections: Parsed structure with overview / background / tools /
            toolkit_notes plus any non-canonical H2s in ``other_sections``.
    """
    md = _read_readme(tool)
    # Collapse badge HTML once up front so every parsed slice inherits clean text.
    md = _linkify_badges(md)

    title_match = _H1_RE.search(md)
    title = title_match.group(1).strip() if title_match else ""

    other: dict[str, str] = {}
    overview = ""
    background = ""
    toolkit_notes = ""
    tools: list[ToolReadmeEntry] = []

    for heading, start, end in _section_bounds(md):
        body = md[start:end].strip()
        if heading == "Overview":
            overview = body
        elif heading == "Background":
            background = body if include_learning_resources else _strip_learning_resources(body)
        elif heading == "Tools":
            tools = _parse_tool_entries(body)
        elif heading == "Toolkit Notes":
            toolkit_notes = body
        else:
            other[heading] = body

    return ReadmeSections(
        title=title,
        overview=overview,
        background=background,
        tools=tools,
        toolkit_notes=toolkit_notes,
        other_sections=other,
    )


def get_tool_docs(
    tool: str,
    *,
    include_toolkit_notes: bool = True,
    include_license: bool = True,
) -> ToolReadmeEntry | None:
    """Return one specific tool's H3 subsection from its toolkit README.

    Args:
        tool (str): Any tool identifier form — registry key, run-function
            name (with or without ``run_``), docs path, or single-tool
            toolkit directory name. Multi-tool toolkit identifiers (e.g.
            ``"esm2"`` for the four-tool ESM2 toolkit) raise ``ValueError``
            because the lookup is ambiguous. See ``_normalize_tool_key``.
        include_toolkit_notes (bool): When True (default), also populate the
            returned entry's ``toolkit_notes`` field from the toolkit's
            ``## Toolkit Notes`` section. The notes apply to every tool in the
            toolkit, so attaching them by default gives agents the full
            relevant context with one call.
        include_license (bool): When True (default), also populate the
            returned entry's ``license`` field from the toolkit's
            ``license.yaml`` (via ``ToolRegistry.get_license``), so agents see
            gating and usage terms without a second call.

    Returns:
        ToolReadmeEntry | None: The matching tool's entry, or None if the
            toolkit's README has no H3 whose key matches the resolved tool.

    Raises:
        ValueError: If ``tool`` doesn't resolve, or resolves ambiguously to
            a multi-tool toolkit.
    """
    tool_key = _normalize_tool_key(tool)
    sections = get_readme_sections(tool_key)
    entry = next((t for t in sections.tools if t.key == tool_key), None)
    if entry is None:
        return None
    if include_toolkit_notes and sections.toolkit_notes:
        entry = entry.model_copy(update={"toolkit_notes": sections.toolkit_notes})
    if include_license:
        from proto_tools.tools.tool_registry import ToolRegistry

        license_data = ToolRegistry.get_license(tool_key)
        if license_data is not None:
            entry = entry.model_copy(update={"license": license_data})
    return entry


# =============================================================================
# Pydantic model docs
# =============================================================================


def _format_type(annotation: Any) -> str:
    """Stringify a Python type annotation for display."""
    text = str(annotation).replace("typing.", "")
    text = re.sub(r"<class '([^']+)'>", r"\1", text)
    text = re.sub(r"<enum '([^']+)'>", r"\1", text)
    # Shorten dotted qualified names (module paths) to their final component.
    return re.sub(r"(?:[A-Za-z_]\w*\.)+([A-Za-z_]\w*)", r"\1", text)


def _format_default(field_info: Any) -> Any:
    """Resolve a Pydantic v2 ``FieldInfo`` default value (or None for required)."""
    if field_info.is_required():
        return None
    return field_info.get_default(call_default_factory=True)


def _clean_docstring(obj: Any) -> str:
    """Return a dedented, stripped docstring (or empty string)."""
    return inspect.cleandoc(obj.__doc__ or "")


def _build_metric_specs(metrics_class: type[Metrics]) -> tuple[list[MetricSpecDoc], str | None]:
    """Build the metric-spec table and primary-metric name from a Metrics subclass."""
    primary = (
        metrics_class.model_fields["primary_metric"].default if "primary_metric" in metrics_class.model_fields else None
    )
    specs = [
        MetricSpecDoc(
            name=name,
            type_str=spec.get("type"),
            min=spec.get("min"),
            max=spec.get("max"),
            unit=spec.get("unit"),
            availability=spec.get("availability"),
            description=spec.get("description"),
            better_values_are=spec.get("better_values_are"),
            is_primary=(name == primary),
        )
        for name, spec in metrics_class.metric_spec.items()
    ]
    return specs, primary


def get_model_doc(
    model_class: type[BaseModel],
    metrics_class: type[Metrics] | None = None,
    iterable_output_field: str | None = None,
) -> ModelDoc:
    """Return a ``ModelDoc`` view of a Pydantic model.

    Includes the class docstring plus a row per field with name, stringified
    type, default value, description, and required flag. Output-model
    metadata fields (``tool_id``, ``execution_time``, etc.) are filtered out.

    Args:
        model_class (type[BaseModel]): Any Pydantic v2 ``BaseModel`` subclass.
        metrics_class (type[Metrics] | None): Optional Metrics subclass the tool emits;
            when given, its ``metric_spec`` populates ``ModelDoc.metric_specs`` and its
            ``primary_metric`` populates ``ModelDoc.primary_metric``.
        iterable_output_field (str | None): The tool's iterable output field name; when
            metrics are present this is recorded as ``ModelDoc.metrics_per_item_field``
            so consumers know the metrics are reported once per item of that list.

    Returns:
        ModelDoc: Normalized view of the model's documentation.
    """
    from proto_tools.utils.tool_io import _OUTPUT_METADATA_FIELDS

    exclude = _OUTPUT_METADATA_FIELDS
    field_docs = field_docs_from_docstrings(model_class)

    fields: list[FieldDoc] = []
    for name, info in model_class.model_fields.items():
        if name in exclude:
            continue
        fields.append(
            FieldDoc(
                name=name,
                type_str=_format_type(info.annotation),
                default=_format_default(info),
                description=info.description,
                doc=field_docs.get(name),
                required=info.is_required(),
            )
        )

    metric_specs: list[MetricSpecDoc] | None = None
    primary_metric: str | None = None
    metrics_per_item_field: str | None = None
    if metrics_class is not None and metrics_class.metric_spec:
        metric_specs, primary_metric = _build_metric_specs(metrics_class)
        metrics_per_item_field = iterable_output_field

    return ModelDoc(
        name=model_class.__name__,
        docstring=_clean_docstring(model_class),
        fields=fields,
        metric_specs=metric_specs,
        primary_metric=primary_metric,
        metrics_per_item_field=metrics_per_item_field,
    )


def field_docs_from_docstrings(model_class: type[BaseModel]) -> dict[str, str]:
    """Map each field of ``model_class`` to its full docstring description.

    Walks the MRO most-derived first and parses each class's *own* Google-style
    docstring (``cls.__doc__``, which is never inherited for classes) for
    ``Attributes:`` entries. The first description seen for a field wins, so a
    subclass that re-documents an inherited field with richer text overrides the
    parent. Fields inherited from a base (whose docs live only in that base's
    docstring) are still picked up further up the MRO.

    Args:
        model_class (type[BaseModel]): A Pydantic model whose docstrings document
            fields in a Google-style ``Attributes:`` section.

    Returns:
        dict[str, str]: Field name to docstring description, restricted to the
            model's own fields.
    """
    field_docs: dict[str, str] = {}
    for cls in model_class.__mro__:
        own_doc = cls.__doc__
        if not own_doc:
            continue
        try:
            parsed = parse_docstring(inspect.cleandoc(own_doc), style=DocstringStyle.GOOGLE)
        except Exception:
            logger.debug("Could not parse docstring for %s", cls.__name__, exc_info=True)
            continue
        for param in parsed.params:
            name = param.arg_name
            if name.startswith("*") or not param.description:
                continue
            field_docs.setdefault(name, param.description.strip())
    return {name: doc for name, doc in field_docs.items() if name in model_class.model_fields}


def _models_in_annotation(annotation: Any) -> list[type[BaseModel]]:
    """Return the ``BaseModel`` subclasses nested anywhere in a type annotation."""
    models: list[type[BaseModel]] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        models.append(annotation)
    for arg in get_args(annotation):
        models.extend(_models_in_annotation(arg))
    return models


def _collect_nested_models(model_class: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Map ``$defs`` names to the nested ``BaseModel`` classes reachable from fields."""
    found: dict[str, type[BaseModel]] = {}

    def _walk(cls: type[BaseModel]) -> None:
        for info in cls.model_fields.values():
            for nested in _models_in_annotation(info.annotation):
                if nested.__name__ not in found:
                    found[nested.__name__] = nested
                    _walk(nested)

    _walk(model_class)
    return found


def _apply_field_docs(schema: dict[str, Any], model_class: type[BaseModel]) -> None:
    """Set ``x-proto-doc`` on each property of ``schema`` from ``model_class`` docstrings."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    docs = field_docs_from_docstrings(model_class)
    for name, prop in properties.items():
        doc = docs.get(name)
        if doc and isinstance(prop, dict):
            prop["x-proto-doc"] = doc


def inject_field_docs(schema: dict[str, Any], model_class: type[BaseModel]) -> dict[str, Any]:
    """Add ``x-proto-doc`` per-field documentation to a model's JSON schema, in place.

    Each property gets an ``x-proto-doc`` key carrying the field's full
    Google-style docstring description, so consumers can render help richer than
    the terse ``description``. Nested model definitions under ``$defs`` are
    annotated too, matching how clients recurse into nested objects.

    Args:
        schema (dict[str, Any]): Output of ``model_class.model_json_schema()``.
        model_class (type[BaseModel]): The model the schema was generated from.

    Returns:
        dict[str, Any]: The same ``schema`` dict, mutated in place.
    """
    _apply_field_docs(schema, model_class)
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        nested = _collect_nested_models(model_class)
        for name, def_schema in defs.items():
            nested_cls = nested.get(name)
            if nested_cls is not None and isinstance(def_schema, dict):
                _apply_field_docs(def_schema, nested_cls)
    return schema


# =============================================================================
# Example notebook extraction
# =============================================================================


def _cell_source_to_str(source: Any) -> str:
    """Normalize an nbformat ``cell.source`` (str or list[str]) to one string."""
    if isinstance(source, list):
        return "".join(source)
    return str(source) if source is not None else ""


def get_example_notebook(tool: str) -> str | None:
    """Render a toolkit's ``examples/example.ipynb`` as markdown + fenced code.

    Markdown cells are emitted as-is; code cells are wrapped in
    ```python ... ``` fences. Outputs, raw cells, and notebook metadata are
    dropped. A ``# example notebook: <path>`` provenance header is prepended.

    Args:
        tool (str): Tool identifier (registry key, run-function name, docs path,
            or toolkit directory name).

    Returns:
        str | None: Rendered notebook text, or ``None`` when the toolkit has
            no ``examples/example.ipynb``.

    Raises:
        ValueError: If ``tool`` doesn't resolve to a registered toolkit.
        OSError: If the notebook exists but cannot be read or parsed.
    """
    toolkit_dir = resolve_toolkit_dir(tool)
    notebook_path = toolkit_dir / "examples" / "example.ipynb"
    if not notebook_path.is_file():
        return None

    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OSError(f"Could not parse notebook {notebook_path}: {exc}") from exc

    chunks: list[str] = [f"# example notebook: {notebook_path}", ""]
    for raw_cell in nb.get("cells", []):
        kind = raw_cell.get("cell_type")
        if kind not in {"markdown", "code"}:
            continue
        source = _cell_source_to_str(raw_cell.get("source", "")).rstrip()
        if kind == "markdown":
            chunks.append(source)
        else:
            chunks.append("```python")
            chunks.append(source)
            chunks.append("```")
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"
