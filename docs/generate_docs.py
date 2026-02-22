#!/usr/bin/env python3
"""
Documentation generator for Bio Programming Tools.

This script auto-generates Mintlify MDX documentation from tool README.md files,
with schema-rich API references sourced from ToolRegistry/Pydantic models.

Run from repository root:
    python docs/generate_docs.py
"""
from __future__ import annotations

import enum
import importlib
import inspect
import json
import logging
import pkgutil
import re
import sys
import types
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

DOCS_DIR = PROJECT_ROOT / "docs"
TOOLS_DIR = DOCS_DIR / "tools"
TOOLS_SOURCE_DIR = PROJECT_ROOT / "bio_programming_tools" / "tools"

# Directories to exclude when discovering tool categories
EXCLUDED_TOOL_DIRS = {"__pycache__", "infra", "utils"}

# BaseToolOutput metadata fields to exclude from generated output docs
BASE_OUTPUT_METADATA_FIELDS = {
    "tool_id",
    "execution_time",
    "timestamp",
    "success",
    "warnings",
    "errors",
    "metadata",
}

logger = logging.getLogger(__name__)


# =============================================================================
# Auto-Import Helper
# =============================================================================


def auto_import_package_modules(package_name: str) -> None:
    """Recursively import all modules in a package to trigger registrations."""
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"  Warning: Could not import {package_name}: {e}")
        return

    if not hasattr(package, "__path__"):
        return

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, prefix=f"{package_name}."
    ):
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            print(f"  Warning: Could not import {module_name}: {e}")


# =============================================================================
# Shared Helpers
# =============================================================================


def discover_tool_categories() -> List[str]:
    """Auto-discover tool categories from bio_programming_tools/tools directory."""
    if not TOOLS_SOURCE_DIR.is_dir():
        print("  Warning: tools source directory not found")
        return []

    categories = sorted(
        d.name
        for d in TOOLS_SOURCE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name not in EXCLUDED_TOOL_DIRS
    )
    return categories


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return text.lower().replace("_", "-").replace(" ", "-")


def slug_to_label(slug: str) -> str:
    """Convert a slug to a human-readable label."""
    acronyms = {"orf": "ORF", "rna": "RNA", "dna": "DNA", "api": "API"}
    words = slug.split("-")
    result = []
    for word in words:
        if word.lower() in acronyms:
            result.append(acronyms[word.lower()])
        else:
            result.append(word.capitalize())
    return " ".join(result)


def extract_first_paragraph(text: str) -> str:
    """Extract first paragraph from text."""
    if not text:
        return ""
    paragraphs = text.strip().split("\n\n")
    return paragraphs[0].replace("\n", " ").strip()


# Compiled patterns for escape_mdx
_FENCE_RE = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"(`[^`]+`)")
_LT_RE = re.compile(r"<")
_CURLY_RE = re.compile(r"[{}]")

# JSX/HTML tags that are legitimate in Mintlify MDX and must not be escaped.
_LEGITIMATE_TAG_RE = re.compile(
    r"</?"
    r"(?:"
    # Mintlify components (PascalCase)
    r"ParamField|ResponseField|Tabs?|Accordion(?:Group)?"
    r"|Card(?:Group)?|CodeGroup|Expandable|Frame|Icon"
    r"|Note|Info|Warning|Tip|Check|Snippet|Steps?|Tooltip"
    # Standard HTML tags that may appear in README content
    r"|br|sub|sup|details|summary|img|a|p|div|span|em|strong"
    r"|table|thead|tbody|tr|th|td|ul|ol|li|hr|blockquote|pre|code"
    r")"
    r"(?:\s|>|/>)"
)


def escape_mdx(text: str) -> str:
    """Escape angle brackets that break MDX parsing.

    MDX interprets ``<`` as a JSX tag opener. This escapes ``<`` to ``&lt;``
    in prose while preserving fenced code blocks, inline code spans, and
    legitimate JSX/HTML tags (e.g. ``<ParamField>``, ``<br>``).
    """
    if not text:
        return text

    # Split on fenced code blocks. Odd-indexed segments are code blocks.
    segments = _FENCE_RE.split(text)

    result: list[str] = []
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            result.append(segment)
        else:
            result.append(_escape_prose_segment(segment))

    return "".join(result)


def _escape_prose_segment(segment: str) -> str:
    """Escape ``<`` in a prose segment, preserving inline code spans."""
    parts = _INLINE_CODE_RE.split(segment)

    result: list[str] = []
    for j, part in enumerate(parts):
        if j % 2 == 1:
            result.append(part)
        else:
            result.append(_escape_angle_brackets(part))

    return "".join(result)


def _escape_angle_brackets(text: str) -> str:
    """Replace ``<`` with ``&lt;`` unless it starts a legitimate tag.

    Also escapes curly braces (``{``/``}``) to ``\\{``/``\\}`` so MDX
    does not interpret them as JSX expression delimiters.
    """

    def _replace(match: re.Match) -> str:
        if _LEGITIMATE_TAG_RE.match(text, match.start()):
            return "<"
        return "&lt;"

    result = _LT_RE.sub(_replace, text)
    result = _CURLY_RE.sub(lambda m: "\\" + m.group(), result)
    return result


# Language to icon mapping for Mintlify code blocks
CODE_BLOCK_ICONS = {
    "python": "python",
    "bash": "terminal",
    "shell": "terminal",
    "sh": "terminal",
    "json": "brackets",
    "yaml": "file",
    "yml": "file",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
}


def add_code_block_icons(text: str) -> str:
    """Add Mintlify icon syntax to fenced code blocks."""
    if not text:
        return text

    def replace_code_block(match: re.Match) -> str:
        lang = match.group(1)
        icon = CODE_BLOCK_ICONS.get(lang)
        if icon:
            return f'```{lang} {lang} icon="{icon}"'
        return match.group(0)

    pattern = (
        r"```("
        + "|".join(re.escape(lang) for lang in CODE_BLOCK_ICONS)
        + r')(?!\s+\w+\s+icon=)'
    )
    return re.sub(pattern, replace_code_block, text)


def extract_readme_description(content: str) -> str:
    """Extract description from README (overview section or first clean paragraph)."""
    overview_match = re.search(r"##\s+Overview\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL)
    if overview_match:
        return extract_first_paragraph(overview_match.group(1))

    content_no_title = re.sub(r"^#\s+.+\n", "", content, count=1)
    paragraphs = content_no_title.strip().split("\n\n")
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith("#") and not p.startswith("```"):
            clean = p.replace("\n", " ")
            return clean[:200]

    return ""


# =============================================================================
# Pydantic Schema Extraction
# =============================================================================


def _own_field_names(model_class: Type[BaseModel]) -> set[str]:
    """Return field names defined directly on model_class (not inherited)."""
    inherited: set[str] = set()
    for parent in model_class.__mro__[1:]:
        if hasattr(parent, "model_fields"):
            inherited |= parent.model_fields.keys()
    return model_class.model_fields.keys() - inherited


def _safe_field_default(model_field: Any) -> Any:
    """Return field default when defined, else None."""
    default = getattr(model_field, "default", PydanticUndefined)
    if default is PydanticUndefined:
        return None
    return default


def _is_required(model_field: Any) -> bool:
    """Return whether model_field is required."""
    checker = getattr(model_field, "is_required", None)
    if callable(checker):
        return checker()
    return bool(checker)


def _json_schema_extra(model_field: Any) -> dict[str, Any]:
    """Extract JSON schema extra metadata if available."""
    extra = getattr(model_field, "json_schema_extra", None)
    if isinstance(extra, dict):
        return extra
    return {}


def _normalize_primitive_type(type_name: str) -> str:
    """Normalize primitive type names for docs display."""
    mapping = {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "none": "null",
        "null": "null",
        "any": "any",
        "object": "object",
    }
    return mapping.get(type_name.lower(), type_name)


def _extract_ndarray_dtype(annotation: Any) -> Optional[str]:
    """Extract a readable ndarray dtype from annotation repr when possible."""
    text = str(annotation)

    for pattern in [r"dtype\[(.*?)\]", r"numpy\.dtype\[(.*?)\]"]:
        match = re.search(pattern, text)
        if match:
            dtype = match.group(1).split(".")[-1]
            return dtype.replace("]", "").strip()
    return None


def _annotation_looks_like_dataframe(annotation: Any) -> bool:
    """Return True when annotation appears to be pandas.DataFrame."""
    if annotation is None:
        return False
    if isinstance(annotation, str):
        text = annotation
    else:
        module = getattr(annotation, "__module__", "")
        qualname = getattr(annotation, "__qualname__", getattr(annotation, "__name__", ""))
        text = f"{module}.{qualname}"
        if not qualname:
            text = str(annotation)
    return "DataFrame" in text and "pandas" in text


def _annotation_looks_like_ndarray(annotation: Any) -> bool:
    """Return True when annotation appears to be numpy ndarray/NDArray."""
    if annotation is None:
        return False
    if isinstance(annotation, str):
        text = annotation
    else:
        module = getattr(annotation, "__module__", "")
        qualname = getattr(annotation, "__qualname__", getattr(annotation, "__name__", ""))
        text = f"{module}.{qualname}"
        if not qualname:
            text = str(annotation)
    return "ndarray" in text.lower() or "NDArray" in text


def _format_ndarray_type(annotation: Any) -> str:
    """Return ndarray type string, optionally including dtype."""
    dtype = _extract_ndarray_dtype(annotation)
    if dtype:
        return f"ndarray[{dtype}]"
    return "ndarray"


def _annotation_to_type_and_enum(annotation: Any) -> tuple[str, Optional[List[Any]]]:
    """Map a Python annotation to a doc type string and optional enum values."""
    if annotation in (None, Any):
        return "any", None

    if _annotation_looks_like_dataframe(annotation):
        return "DataFrame", None
    if _annotation_looks_like_ndarray(annotation):
        return _format_ndarray_type(annotation), None

    if isinstance(annotation, str):
        lowered = annotation.lower()
        if "dataframe" in lowered:
            return "DataFrame", None
        if "ndarray" in lowered:
            return "ndarray", None
        return _normalize_primitive_type(annotation), None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        if args:
            return _annotation_to_type_and_enum(args[0])
        return "any", None

    if origin in (Union, types.UnionType):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if not non_none_args:
            return "null", None
        if len(non_none_args) == 1:
            return _annotation_to_type_and_enum(non_none_args[0])
        mapped = [_annotation_to_type_and_enum(arg)[0] for arg in non_none_args]
        return " | ".join(mapped), None

    if origin is Literal:
        values = list(args)
        return "enum", values

    if origin in (list, List):
        item_type = _annotation_to_type_and_enum(args[0])[0] if args else "any"
        return f"List[{item_type}]", None

    if origin in (dict, Dict):
        key_type = _annotation_to_type_and_enum(args[0])[0] if len(args) >= 1 else "any"
        value_type = _annotation_to_type_and_enum(args[1])[0] if len(args) >= 2 else "any"
        return f"Dict[{key_type}, {value_type}]", None

    if origin in (tuple, Tuple):
        if not args:
            return "Tuple", None
        if len(args) == 2 and args[1] is Ellipsis:
            item_type = _annotation_to_type_and_enum(args[0])[0]
            return f"Tuple[{item_type}, ...]", None
        part_types = [_annotation_to_type_and_enum(arg)[0] for arg in args]
        return f"Tuple[{', '.join(part_types)}]", None

    if inspect.isclass(annotation):
        if issubclass(annotation, enum.Enum):
            enum_values = [member.value for member in annotation]
            return "enum", enum_values

        primitive_name = _normalize_primitive_type(annotation.__name__)
        if primitive_name != annotation.__name__:
            return primitive_name, None

        if annotation.__module__ == "builtins":
            return annotation.__name__, None
        return annotation.__name__, None

    return "object", None


def _parse_schema_property(
    name: str,
    prop: Dict[str, Any],
    required_set: set[str],
) -> Dict[str, Any]:
    """Extract a field definition from JSON schema property data."""
    field: Dict[str, Any] = {
        "name": name,
        "type": _normalize_primitive_type(prop.get("type", "any")),
        "required": name in required_set,
        "default": prop.get("default"),
        "description": prop.get("description", ""),
        "title": prop.get("title", name),
        "advanced": prop.get("advanced", False),
        "hidden": prop.get("hidden", False),
    }

    if "anyOf" in prop:
        mapped_types: List[str] = []
        for option in prop["anyOf"]:
            option_type = option.get("type")
            if option_type:
                mapped_types.append(_normalize_primitive_type(option_type))
            elif "enum" in option:
                mapped_types.append("enum")
        mapped_types = [t for t in mapped_types if t != "null"]
        if mapped_types:
            field["type"] = " | ".join(mapped_types)

    if "enum" in prop:
        field["type"] = "enum"
        field["enum_values"] = prop["enum"]

    if field["type"] == "array" and "items" in prop:
        item_type = _normalize_primitive_type(prop["items"].get("type", "any"))
        field["type"] = f"List[{item_type}]"

    return field


def _build_field_from_model_field(name: str, model_field: Any) -> Dict[str, Any]:
    """Build a docs field definition by introspecting Pydantic model_field."""
    type_name, enum_values = _annotation_to_type_and_enum(model_field.annotation)
    extra = _json_schema_extra(model_field)

    field: Dict[str, Any] = {
        "name": name,
        "type": type_name,
        "required": _is_required(model_field),
        "default": _safe_field_default(model_field),
        "description": getattr(model_field, "description", "") or "",
        "title": getattr(model_field, "title", None) or name,
        "advanced": bool(extra.get("advanced", False)),
        "hidden": bool(extra.get("hidden", False)),
    }
    if enum_values:
        field["enum_values"] = enum_values
    return field


def _should_prefer_introspection(schema_field: Dict[str, Any], inferred_field: Dict[str, Any]) -> bool:
    """Return True when introspected field should override schema-derived field."""
    inferred_type = str(inferred_field.get("type", "any"))
    schema_type = str(schema_field.get("type", "any"))

    if inferred_type in {"DataFrame", "ndarray"} or inferred_type.startswith("ndarray["):
        return True

    if schema_type in {"any", "object"} and not schema_field.get("enum_values"):
        return True

    if "any" in schema_type and inferred_type != schema_type and inferred_type not in {"any", "object"}:
        return True

    if not schema_field.get("description") and inferred_field.get("description"):
        return True

    return False


def parse_pydantic_fields(
    model_class: Type[BaseModel],
    exclude: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Extract model fields for docs, preferring schema with resilient fallbacks.

    Uses model_json_schema() when available, then falls back to model_fields
    introspection for opaque/missing/non-JSON-schema-native types.

    Returns list of dicts with keys:
        name, type, required, default, description, title, advanced, hidden, enum_values
    """
    exclude = exclude or set()

    schema_fields: Dict[str, Dict[str, Any]] = {}
    try:
        schema = model_class.model_json_schema()
        properties = schema.get("properties", {})
        required_set = set(schema.get("required", []))
        for name, prop in properties.items():
            if name in exclude:
                continue
            schema_fields[name] = _parse_schema_property(name, prop, required_set)
    except Exception as exc:
        logger.warning(
            "Could not generate JSON schema for %s; using model field introspection: %s",
            model_class.__name__,
            exc,
        )

    parsed_fields: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for name, model_field in model_class.model_fields.items():
        if name in exclude:
            continue
        seen_names.add(name)

        schema_field = schema_fields.get(name)
        try:
            inferred_field = _build_field_from_model_field(name, model_field)
        except Exception as exc:
            logger.warning(
                "Could not introspect %s.%s; falling back to schema/default field: %s",
                model_class.__name__,
                name,
                exc,
            )
            if schema_field:
                parsed_fields.append(schema_field)
            else:
                parsed_fields.append(
                    {
                        "name": name,
                        "type": "object",
                        "required": _is_required(model_field),
                        "default": _safe_field_default(model_field),
                        "description": "",
                        "title": name,
                        "advanced": False,
                        "hidden": False,
                    }
                )
            continue

        if schema_field and not _should_prefer_introspection(schema_field, inferred_field):
            if not schema_field.get("description") and inferred_field.get("description"):
                schema_field["description"] = inferred_field["description"]
            parsed_fields.append(schema_field)
        else:
            parsed_fields.append(inferred_field)

    for name, schema_field in schema_fields.items():
        if name not in seen_names:
            parsed_fields.append(schema_field)

    # Reorder: own fields first, inherited base fields last.
    own_names = _own_field_names(model_class)
    own = [f for f in parsed_fields if f["name"] in own_names]
    inherited = [f for f in parsed_fields if f["name"] not in own_names]
    return own + inherited


# =============================================================================
# Mintlify Component Formatters
# =============================================================================


def format_param_field(param: Dict[str, Any]) -> str:
    """Format a single parameter as a Mintlify <ParamField> component."""
    attrs = [f'path="{param["name"]}"', f'type="{param["type"]}"']
    if param["required"]:
        attrs.append("required")
    if param.get("default") is not None:
        attrs.append(f'default="{param["default"]}"')

    desc = escape_mdx(param.get("description", ""))
    lines = [f'<ParamField {" ".join(attrs)}>']
    lines.append(f"  {desc}")

    if param.get("enum_values"):
        enum_str = "`, `".join(str(v) for v in param["enum_values"])
        lines.append("")
        lines.append(f"  Options: `{enum_str}`")

    lines.append("</ParamField>")
    return "\n".join(lines)


def format_param_fields(params: List[Dict[str, Any]]) -> str:
    """Format a list of parameters as Mintlify <ParamField> components."""
    visible = [p for p in params if not p.get("hidden")]
    if not visible:
        return "_No parameters_"
    return "\n\n".join(format_param_field(p) for p in visible)


def format_response_field(field: Dict[str, Any]) -> str:
    """Format a single field as a Mintlify <ResponseField> component."""
    attrs = [f'name="{field["name"]}"', f'type="{field["type"]}"']
    if field["required"]:
        attrs.append("required")

    desc = escape_mdx(field.get("description", ""))
    lines = [f'<ResponseField {" ".join(attrs)}>']
    lines.append(f"  {desc}")
    lines.append("</ResponseField>")
    return "\n".join(lines)


def format_response_fields(fields: List[Dict[str, Any]]) -> str:
    """Format a list of fields as Mintlify <ResponseField> components."""
    if not fields:
        return "_No fields_"
    return "\n\n".join(format_response_field(f) for f in fields)


# =============================================================================
# Tool Documentation Generator
# =============================================================================


def try_import_tool_registry():
    """Graceful import of ToolRegistry; returns None if deps missing."""
    try:
        auto_import_package_modules("bio_programming_tools.tools")
        from bio_programming_tools import ToolRegistry

        return ToolRegistry
    except Exception as e:
        print(f"  Warning: Could not import ToolRegistry: {e}")
        print("  Falling back to README-only tool docs")
        return None


def _tool_source_dir(spec) -> Optional[Path]:
    """Get the source directory for a tool function using inspect.getfile().

    Walks up from the source file to find the nearest directory containing a
    README.md, which is the canonical tool directory.
    """
    try:
        unwrapped = inspect.unwrap(spec.function)
        source_file = Path(inspect.getfile(unwrapped))
    except (TypeError, OSError):
        return None

    current = source_file.parent
    while current != current.parent:
        if (current / "README.md").exists():
            return current
        if current.name == "tools":
            break
        current = current.parent
    return source_file.parent


def _strip_hand_written_schemas(
    readme_text: str,
    *,
    strip_input: bool = True,
    strip_config: bool = True,
    strip_output: bool = True,
    strip_important: bool = True,
) -> str:
    """Strip selected hand-written schema sections from README.

    This lets us keep manual fallback documentation (for example Output
    sections) if schema extraction fails for a model.
    """
    sections: List[str] = []
    if strip_input:
        sections.extend([r"Inputs?", r"Input Parameters?"])
    if strip_config:
        sections.append(r"Configurations?")
    if strip_output:
        sections.extend([r"Outputs?", r"Output Specification?"])
    if strip_important:
        sections.append(r"Important Parameters?")

    if not sections:
        return readme_text

    pattern = re.compile(
        rf"^##\s+(?:{'|'.join(sections)})\s*\n"
        r"(?:(?!^##\s).*\n?)*",
        re.MULTILINE,
    )
    return pattern.sub("", readme_text)


def _build_tool_api_section(spec) -> tuple[str, set[str]]:
    """Build API Reference MDX for a ToolSpec.

    Returns:
        (section_markdown, generated_sections)
        generated_sections contains any of: {"input", "config", "output"}.
    """
    sections: List[str] = []
    generated_sections: set[str] = set()

    try:
        input_fields = parse_pydantic_fields(spec.input_model)
        if input_fields:
            sections.append("### Input\n")
            sections.append(format_param_fields(input_fields))
            generated_sections.add("input")
    except Exception as exc:
        logger.warning("Could not parse input model for %s: %s", spec.key, exc)

    try:
        config_params = parse_pydantic_fields(spec.config_model)
        if config_params:
            sections.append("\n### Configuration\n")
            sections.append(format_param_fields(config_params))
            generated_sections.add("config")
    except Exception as exc:
        logger.warning("Could not parse config model for %s: %s", spec.key, exc)

    try:
        output_fields = parse_pydantic_fields(
            spec.output_model,
            exclude=BASE_OUTPUT_METADATA_FIELDS,
        )
        if output_fields:
            sections.append("\n### Output\n")
            sections.append(format_response_fields(output_fields))
            generated_sections.add("output")
    except Exception as exc:
        logger.warning("Could not parse output model for %s: %s", spec.key, exc)

    return "\n".join(sections), generated_sections


def generate_tool_docs() -> Dict[str, List[str]]:
    """Generate MDX documentation for all tools.

    Merges README content with ToolRegistry-backed API reference sections.
    """
    categories = discover_tool_categories()
    if not categories:
        return {}

    tool_registry = try_import_tool_registry()

    dir_to_specs: Dict[Path, list] = {}
    if tool_registry is not None:
        for spec in tool_registry.list_all():
            src_dir = _tool_source_dir(spec)
            if src_dir is not None:
                dir_to_specs.setdefault(src_dir.resolve(), []).append(spec)

    category_pages: Dict[str, List[str]] = {}

    for category in categories:
        category_dir = TOOLS_SOURCE_DIR / category

        readme_paths = sorted(category_dir.rglob("README.md"))
        skip_tool_dirs = {"standalone", "__pycache__"}
        readme_paths = [
            p
            for p in readme_paths
            if p.parent != category_dir and p.parent.name not in skip_tool_dirs
        ]

        if not readme_paths:
            continue

        category_slug = slugify(category)
        output_dir = TOOLS_DIR / category_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        for readme_path in readme_paths:
            tool_dir = readme_path.parent
            tool_slug = slugify(tool_dir.name)
            readme_text = readme_path.read_text()

            specs = dir_to_specs.get(tool_dir.resolve(), [])

            title_match = re.match(r"^#\s+(.+)", readme_text)
            title = title_match.group(1).strip() if title_match else slug_to_label(tool_slug)

            desc = extract_readme_description(readme_text)
            if not desc and specs:
                desc = specs[0].description

            mdx_parts: List[str] = []
            safe_title = title.replace('"', '\\"')
            safe_desc = desc.replace('"', '\\"')
            mdx_parts.append(
                f'---\ntitle: "{safe_title}"\ndescription: "{safe_desc}"\n---\n'
            )

            generated_sections_per_spec: List[set[str]] = []
            api_reference_parts: List[str] = []

            if specs:
                if len(specs) == 1:
                    api_section, generated_sections = _build_tool_api_section(specs[0])
                    generated_sections_per_spec.append(generated_sections)
                    api_reference_parts.append("\n## API Reference\n")
                    api_reference_parts.append(api_section)
                else:
                    api_reference_parts.append("\n## API Reference\n")
                    api_reference_parts.append("<Tabs>")
                    for spec in sorted(specs, key=lambda s: s.key):
                        api_section, generated_sections = _build_tool_api_section(spec)
                        generated_sections_per_spec.append(generated_sections)
                        api_reference_parts.append(f'  <Tab title="{spec.label}">')
                        api_reference_parts.append(f"  **Key:** `{spec.key}`\n")
                        api_reference_parts.append(f"  {spec.description}\n")
                        api_reference_parts.append(api_section)
                        api_reference_parts.append("  </Tab>")
                    api_reference_parts.append("</Tabs>")

            body = readme_text
            if generated_sections_per_spec:
                strip_input = all("input" in s for s in generated_sections_per_spec)
                strip_config = all("config" in s for s in generated_sections_per_spec)
                strip_output = all("output" in s for s in generated_sections_per_spec)
                body = _strip_hand_written_schemas(
                    body,
                    strip_input=strip_input,
                    strip_config=strip_config,
                    strip_output=strip_output,
                    strip_important=(strip_input or strip_config),
                )

            body = add_code_block_icons(body)
            body = escape_mdx(body)
            mdx_parts.append(body)

            mdx_parts.extend(api_reference_parts)

            mdx = "\n".join(mdx_parts) + "\n"

            output_path = output_dir / f"{tool_slug}.mdx"
            output_path.write_text(mdx)

            page_path = f"tools/{category_slug}/{tool_slug}"
            category_pages.setdefault(category_slug, []).append(page_path)
            print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")

    generated_pages = {
        page_path for pages in category_pages.values() for page_path in pages
    }
    stale_pages = prune_stale_tool_pages(generated_pages)
    for page_path in stale_pages:
        print(f"  Removed stale: docs/{page_path}.mdx")

    return category_pages


# =============================================================================
# Validation
# =============================================================================


def prune_stale_tool_pages(generated_pages: set[str]) -> List[str]:
    """Remove stale generated tool pages not present in the current run."""
    stale_pages: List[str] = []

    if not TOOLS_DIR.exists():
        return stale_pages

    for path in sorted(TOOLS_DIR.rglob("*.mdx")):
        page_path = str(path.relative_to(DOCS_DIR).with_suffix("")).replace("\\", "/")
        if page_path not in generated_pages:
            path.unlink()
            stale_pages.append(page_path)

    # Remove empty category directories left behind after stale page cleanup.
    for directory in sorted(TOOLS_DIR.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    return stale_pages


def _expected_tool_pages_from_readmes() -> set[str]:
    """Compute expected tool page paths from source README files."""
    expected: set[str] = set()
    skip_tool_dirs = {"standalone", "__pycache__"}

    for category in discover_tool_categories():
        category_dir = TOOLS_SOURCE_DIR / category
        for readme_path in sorted(category_dir.rglob("README.md")):
            if (
                readme_path.parent == category_dir
                or readme_path.parent.name in skip_tool_dirs
            ):
                continue
            category_slug = slugify(category)
            tool_slug = slugify(readme_path.parent.name)
            expected.add(f"tools/{category_slug}/{tool_slug}")

    return expected


def validate_generated_docs(tool_pages: Dict[str, List[str]]) -> None:
    """Validate generated docs coverage and navigation consistency."""
    expected_from_readmes = _expected_tool_pages_from_readmes()
    generated_from_mapping = {
        page_path for pages in tool_pages.values() for page_path in pages
    }
    if generated_from_mapping != expected_from_readmes:
        missing = sorted(expected_from_readmes - generated_from_mapping)
        extra = sorted(generated_from_mapping - expected_from_readmes)
        raise ValueError(
            "README coverage mismatch in generated tool pages.\n"
            f"  Missing pages: {missing}\n"
            f"  Extra pages: {extra}"
        )

    generated_on_disk = {
        str(path.relative_to(DOCS_DIR).with_suffix("")).replace("\\", "/")
        for path in TOOLS_DIR.rglob("*.mdx")
    }
    if generated_on_disk != generated_from_mapping:
        missing = sorted(generated_from_mapping - generated_on_disk)
        extra = sorted(generated_on_disk - generated_from_mapping)
        raise ValueError(
            "Disk output mismatch for generated tool docs.\n"
            f"  Missing files: {missing}\n"
            f"  Extra files: {extra}"
        )

    docs_json_path = DOCS_DIR / "docs.json"
    docs = json.loads(docs_json_path.read_text())
    tools_tab = next(
        (tab for tab in docs.get("navigation", {}).get("tabs", []) if tab.get("tab") == "Tools"),
        None,
    )
    if tools_tab is None:
        raise ValueError("docs.json missing 'Tools' tab")

    nav_pages = {
        page
        for group in tools_tab.get("groups", [])
        for page in group.get("pages", [])
    }
    if nav_pages != generated_from_mapping:
        missing = sorted(generated_from_mapping - nav_pages)
        extra = sorted(nav_pages - generated_from_mapping)
        raise ValueError(
            "Navigation mismatch for tool docs.\n"
            f"  Missing in navigation: {missing}\n"
            f"  Unknown in navigation: {extra}"
        )


# =============================================================================
# Navigation Updater
# =============================================================================


def update_docs_json(tool_pages: Dict[str, List[str]]) -> None:
    """Update docs.json navigation with generated tool pages."""
    docs_json_path = DOCS_DIR / "docs.json"

    if not docs_json_path.exists():
        print("  Warning: docs.json not found, skipping navigation update")
        return

    docs = json.loads(docs_json_path.read_text())

    if "navigation" not in docs:
        print("  Error: docs.json missing 'navigation' key")
        return

    if "tabs" not in docs["navigation"]:
        print("  Error: docs.json missing 'navigation.tabs' key")
        return

    tools_tab = None
    for tab in docs["navigation"]["tabs"]:
        if tab.get("tab") == "Tools":
            tools_tab = tab
            break

    if not tools_tab:
        print("  Warning: No 'Tools' tab found in docs.json")
        return

    new_groups = []
    for category_slug in sorted(tool_pages.keys()):
        group_label = slug_to_label(category_slug)
        pages = sorted(tool_pages[category_slug])

        new_groups.append({"group": group_label, "pages": pages})

    tools_tab["groups"] = new_groups
    docs_json_path.write_text(json.dumps(docs, indent="\t") + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point for documentation generation."""
    print("=" * 60)
    print("Bio Programming Tools Documentation Generator")
    print("=" * 60)

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Generating tool documentation...")
    tool_pages = generate_tool_docs()
    total_tools = sum(len(pages) for pages in tool_pages.values())
    print(f"  Total: {total_tools} tools across {len(tool_pages)} categories")

    print("\n[2/3] Updating docs.json navigation...")
    update_docs_json(tool_pages)
    print("  Updated: docs/docs.json")

    print("\n[3/3] Validating generated docs integrity...")
    validate_generated_docs(tool_pages)
    print("  Validation passed")

    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"  - Tools: {total_tools}")
    print(f"  - Categories: {len(tool_pages)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
