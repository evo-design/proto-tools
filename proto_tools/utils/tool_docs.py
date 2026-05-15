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

Two surfaces:

1. **README extraction** — ``get_readme`` / ``get_readme_sections`` /
   ``get_readme_section`` / ``get_tool_docs`` return raw or structured
   slices of a toolkit's README. ``get_tool_docs`` also attaches the parsed
   ``license.yaml`` by default (the human-facing License callout above
   ``## Overview`` is not a parsed section; structured license metadata is
   the agent-facing source of truth).
2. **Pydantic model docs** — ``get_model_doc`` returns a normalized view of
   a model's class docstring plus per-field name / type / default /
   description / required flag.
"""

from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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
    background: str = Field(description="Body of the '## Background' section.")
    tools: list[ToolReadmeEntry] = Field(
        default_factory=list,
        description="H3 subsections inside the '## Tools' section, one per registered tool.",
    )
    toolkit_notes: str = Field(
        default="",
        description="Body of the '## Toolkit Notes' section. Empty when the section is not present.",
    )
    qc_pending: bool = Field(
        default=False,
        description="True if the README still has the '> [!NOTE] **TODO:** ...' callout.",
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
        description="Per-field description (from ``Field(description=...)``).",
    )
    required: bool = Field(description="Whether the field is required (no default).")


class ModelDoc(BaseModel):
    """Normalized view of a Pydantic model's docs."""

    name: str = Field(description="Class name.")
    docstring: str = Field(description="Cleaned class docstring.")
    fields: list[FieldDoc] = Field(default_factory=list, description="Per-field docs.")


# =============================================================================
# README extraction
# =============================================================================


# TODO(#743): remove all QC-pending plumbing (this regex, _strip_review_callout,
# _has_qc_callout, the qc_pending field) once every README is migrated.
_TODO_CALLOUT_RE = re.compile(
    r"^>\s*\[!NOTE\]\s*\n>\s*\*\*TODO:\*\*\s*This README still needs to be reviewed[^\n]*\n+",
    re.MULTILINE,
)

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


def _strip_review_callout(md: str) -> str:
    """Strip the ``> [!NOTE] **TODO: review**`` callout from a README, if present."""
    return _TODO_CALLOUT_RE.sub("", md)


def _has_qc_callout(md: str) -> bool:
    """Return True if the README still carries the QC-pending callout."""
    return _TODO_CALLOUT_RE.search(md) is not None


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

    The ``> [!NOTE] **TODO:** ...`` callout that flags un-reviewed READMEs is
    always stripped.

    Args:
        tool (str): Tool identifier (registry key, docs path, run-function
            name, or toolkit directory name).

    Returns:
        str: README content.

    Raises:
        ValueError: If ``tool`` doesn't resolve to a registered toolkit.
        OSError: If the README cannot be read from disk.
    """
    return _strip_review_callout(_read_readme(tool))


def get_readme_section(tool: str, heading: str) -> str | None:
    """Return one H2 section's body by exact heading text.

    Args:
        tool (str): Tool identifier — same forms as ``get_readme``.
        heading (str): H2 heading text (e.g. ``"Background"``).

    Returns:
        str | None: The section body (without its heading), or None if no
            matching H2 is found.
    """
    body = _h2_body(get_readme(tool), heading)
    return body or None


def get_readme_sections(tool: str) -> ReadmeSections:
    """Return a toolkit's README parsed into a typed structure.

    The QC-pending callout is always stripped before parsing; the
    ``qc_pending`` field on the result still reflects whether the callout was
    present in the source.

    Args:
        tool (str): Tool identifier — same forms as ``get_readme``.

    Returns:
        ReadmeSections: Parsed structure with overview / background / tools /
            toolkit_notes plus any non-canonical H2s in ``other_sections``.
    """
    raw = _read_readme(tool)
    qc_pending = _has_qc_callout(raw)
    md = _strip_review_callout(raw)
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
            background = body
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
        qc_pending=qc_pending,
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
    return re.sub(r"<class '([^']+)'>", r"\1", str(annotation).replace("typing.", ""))


def _format_default(field_info: Any) -> Any:
    """Resolve a Pydantic v2 ``FieldInfo`` default value (or None for required)."""
    if field_info.is_required():
        return None
    return field_info.get_default(call_default_factory=True)


def _clean_docstring(obj: Any) -> str:
    """Return a dedented, stripped docstring (or empty string)."""
    return inspect.cleandoc(obj.__doc__ or "")


def get_model_doc(model_class: type[BaseModel]) -> ModelDoc:
    """Return a ``ModelDoc`` view of a Pydantic model.

    Includes the class docstring plus a row per field with name, stringified
    type, default value, description, and required flag. Output-model
    metadata fields (``tool_id``, ``execution_time``, etc.) are filtered out.

    Args:
        model_class (type[BaseModel]): Any Pydantic v2 ``BaseModel`` subclass.

    Returns:
        ModelDoc: Normalized view of the model's documentation.
    """
    from proto_tools.utils.tool_io import _OUTPUT_METADATA_FIELDS

    exclude = _OUTPUT_METADATA_FIELDS

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
                required=info.is_required(),
            )
        )

    return ModelDoc(
        name=model_class.__name__,
        docstring=_clean_docstring(model_class),
        fields=fields,
    )
