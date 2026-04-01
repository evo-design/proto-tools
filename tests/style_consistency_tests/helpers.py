"""tests/style_consistency_tests/helpers.py.

Shared helpers for style-consistency tests.
"""

from __future__ import annotations

import ast
import contextlib
import re
from collections.abc import Iterable
from pathlib import Path

# ── Existing helpers ────────────────────────────────────────────────────────


def find_missing_fields_in_docstring(docstring: str, field_names: Iterable[str]) -> list[str]:
    """Return field names not mentioned in the docstring."""
    return [name for name in field_names if name not in docstring]


def field_description_is_valid(description: str | None, max_length: int = 100) -> str:
    """Return an error string if the description is invalid, or '' if valid."""
    if description is None:
        return "is None"
    if len(description) > max_length:
        return f"is too long (currently {len(description)} characters, must be under {max_length} characters)"
    if not description.strip():
        return "description is empty or just whitespace"
    if "\n" in description:
        return "description contains newline characters. Please use single line descriptions."
    return ""


# ── Type normalization ──────────────────────────────────────────────────────

# Old-style typing names that map to builtin lowercase equivalents (PEP 585)
_TYPING_TO_BUILTIN = {
    "List": "list",
    "Dict": "dict",
    "Tuple": "tuple",
    "Set": "set",
    "FrozenSet": "frozenset",
    "Type": "type",
}


def normalize_type(type_string: str) -> str:
    """Normalize a type string to canonical modern Python form.

    Converts:
        - List -> list, Dict -> dict, Tuple -> tuple, Set -> set, FrozenSet -> frozenset
        - Optional[X] -> X | None
        - Union[X, Y] -> X | Y
        - None | X -> X | None (None always last in unions)
        - typing.X -> X (strips typing. prefix)
        - Whitespace normalized via ast.unparse round-trip

    Args:
        type_string: A type annotation string to normalize.

    Returns:
        str: The normalized type string in modern Python form.
    """
    if not type_string or not type_string.strip():
        return type_string

    t = type_string.strip()

    # Strip surrounding quotes from forward references: 'ClassName' -> ClassName
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1]

    # Strip leading 'typing.' prefix
    t = re.sub(r"\btyping\.", "", t)

    try:
        tree = ast.parse(t, mode="eval")
        new_tree = _normalize_ast(tree.body)
        return ast.unparse(new_tree)
    except (SyntaxError, ValueError):
        # Fallback: basic string normalization for unparseable types
        return _normalize_string_fallback(t)


def _normalize_ast(node: ast.expr) -> ast.expr:
    """Recursively normalize an AST type expression to modern Python form."""
    if isinstance(node, ast.Name):
        # List -> list, Dict -> dict, etc.
        if node.id in _TYPING_TO_BUILTIN:
            return ast.Name(id=_TYPING_TO_BUILTIN[node.id], ctx=ast.Load())
        return node

    if isinstance(node, ast.Attribute):
        # typing.List -> list, typing.Optional -> handle below
        if isinstance(node.value, ast.Name) and node.value.id == "typing":
            return _normalize_ast(ast.Name(id=node.attr, ctx=ast.Load()))
        return node

    if isinstance(node, ast.Subscript):
        # Handle Optional[X] -> X | None
        if isinstance(node.value, ast.Name) and node.value.id == "Optional":
            inner = _normalize_ast(node.slice)
            return _make_union_with_none(inner)

        # Handle Union[X, Y, ...] -> X | Y | ...
        if isinstance(node.value, ast.Name) and node.value.id == "Union":
            if isinstance(node.slice, ast.Tuple):
                elements = [_normalize_ast(e) for e in node.slice.elts]
            else:
                elements = [_normalize_ast(node.slice)]
            return _make_bitor_chain(elements)

        # Normalize the base name and slice recursively
        new_value = _normalize_ast(node.value)
        if isinstance(node.slice, ast.Tuple):
            new_slice = ast.Tuple(
                elts=[_normalize_ast(e) for e in node.slice.elts],
                ctx=ast.Load(),
            )
        else:
            new_slice = _normalize_ast(node.slice)

        return ast.Subscript(value=new_value, slice=new_slice, ctx=ast.Load())

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # Normalize both sides of X | Y, then ensure None is last
        left = _normalize_ast(node.left)
        right = _normalize_ast(node.right)
        elements = _flatten_bitor(left) + _flatten_bitor(right)
        return _make_bitor_chain(elements)

    if isinstance(node, ast.Constant):
        return node

    if isinstance(node, ast.List):
        return ast.List(elts=[_normalize_ast(e) for e in node.elts], ctx=ast.Load())

    if isinstance(node, ast.Tuple):
        return ast.Tuple(elts=[_normalize_ast(e) for e in node.elts], ctx=ast.Load())

    return node


def _flatten_bitor(node: ast.expr) -> list[ast.expr]:
    """Flatten a chain of X | Y | Z into a list of elements."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flatten_bitor(node.left) + _flatten_bitor(node.right)
    return [node]


def _make_bitor_chain(elements: list[ast.expr]) -> ast.expr:
    """Build a normalized X | Y | ... chain with None always last."""
    none_elements = [e for e in elements if _is_none(e)]
    non_none = [e for e in elements if not _is_none(e)]

    ordered = non_none + none_elements

    if len(ordered) == 1:
        return ordered[0]

    result = ordered[0]
    for elem in ordered[1:]:
        result = ast.BinOp(left=result, op=ast.BitOr(), right=elem)
    return result


def _make_union_with_none(inner: ast.expr) -> ast.expr:
    """Create X | None from an inner type expression."""
    return ast.BinOp(
        left=inner,
        op=ast.BitOr(),
        right=ast.Constant(value=None),
    )


def _is_none(node: ast.expr) -> bool:
    """Check if an AST node represents None."""
    return isinstance(node, ast.Constant) and node.value is None


def _normalize_string_fallback(t: str) -> str:
    """Basic string normalization for types that can't be parsed as Python AST."""
    for old, new in _TYPING_TO_BUILTIN.items():
        t = re.sub(rf"\b{old}\b", new, t)
    return re.sub(r"\s+", " ", t).strip()


# ── Returns type extraction (workaround for docstring_parser union bug) ────

_RETURNS_TYPE_RE = re.compile(
    r"^\s*"
    r"("
    r"[^\s:]+(?:\[.*?\])?"
    r"(?:\s*\|\s*[^\s:]+(?:\[.*?\])?)*"
    r")"
    r"\s*:\s*"
    r"(.+)",
    re.DOTALL,
)


def extract_returns_type(docstring: str) -> str | None:
    """Extract the return type from a docstring's Returns: section.

    Falls back to manual regex parsing when docstring_parser can't handle
    union types with spaces (e.g., ``str | None: description``).

    Args:
        docstring (str): The full docstring text.

    Returns:
        str | None: The return type string, or None if not found.
    """
    match = re.search(
        r"(?:^|\n)\s*Returns:\s*\n(.*?)(?=\n\s*(?:Raises|Examples?|Note|Yields|$)|\Z)",
        docstring,
        re.DOTALL,
    )
    if not match:
        return None
    body = match.group(1)
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        type_match = _RETURNS_TYPE_RE.match(stripped)
        if type_match:
            return type_match.group(1).strip()
        return None
    return None


# ── File and docstring collection helpers ───────────────────────────────────


def collect_docstrings_with_annotations(
    repo_root: Path,
    directories: list[str],
    exclude_patterns: list[str] | None = None,
) -> list[tuple[str, str, str, dict[str, str], str | None, str, dict[str, str]]]:
    """Collect multi-line docstrings with their corresponding type annotations.

    For functions: extracts parameter annotations from the signature.
    For classes: extracts attribute annotations from class body assignments.

    Args:
        repo_root (Path): Absolute path to the repository root.
        directories (list[str]): Directory names relative to repo_root to scan.
        exclude_patterns (list[str] | None): Optional glob patterns to exclude.

    Returns:
        list[tuple[str, str, str, dict[str, str], str | None, str, dict[str, str]]]:
            (file_path, qualified_name, docstring, all_annotations,
             return_type, node_kind, own_annotations) tuples.
            all_annotations includes inherited fields (for type checking).
            own_annotations only has fields defined on this class/function (for completeness).
    """
    exclude_patterns = exclude_patterns or []
    results: list[tuple[str, str, str, dict[str, str], str | None, str, dict[str, str]]] = []

    # First pass: build global class annotation map for inheritance resolution
    class_annotations: dict[str, dict[str, str]] = {}
    all_trees: list[tuple[str, ast.Module]] = []

    for directory in directories:
        dir_path = repo_root / directory
        if not dir_path.is_dir():
            continue
        for py_file in sorted(dir_path.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel_path = str(py_file.relative_to(repo_root))
            if any(_matches_pattern(rel_path, pat) for pat in exclude_patterns):
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            all_trees.append((rel_path, tree))

            # Collect class annotations for inheritance resolution
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    attrs = _extract_class_annotations(node)
                    if attrs:
                        class_annotations[node.name] = attrs

    # Second pass: collect docstrings with annotations
    for rel_path, tree in all_trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            docstring = ast.get_docstring(node)
            if docstring is None or not _is_multiline_docstring(docstring):
                continue

            qualified_name = _get_qualified_name(node, tree)

            if isinstance(node, ast.ClassDef):
                own_annotations = _extract_class_annotations(node)
                all_annotations = _resolve_class_annotations(node, class_annotations)
                results.append((rel_path, qualified_name, docstring, all_annotations, None, "class", own_annotations))
            else:
                annotations = _extract_function_annotations(node)
                return_type = None
                if node.returns is not None:
                    with contextlib.suppress(Exception):
                        return_type = ast.unparse(node.returns)
                results.append((rel_path, qualified_name, docstring, annotations, return_type, "function", annotations))

    return sorted(results, key=lambda x: (x[0], x[1]))


# ── Continuation-line indentation checking ─────────────────────────────────

# Sections where entries follow the pattern: name (type): description
_NAMED_SECTIONS = {"Args", "Arguments", "Attributes"}

# Sections where entries follow the pattern: type: description
_TYPED_SECTIONS = {"Returns", "Yields"}

# Sections where entries follow the pattern: ExceptionType: description
_RAISES_SECTIONS = {"Raises"}

_ALL_SECTION_NAMES = _NAMED_SECTIONS | _TYPED_SECTIONS | _RAISES_SECTIONS

# Matches a section header like "    Args:" or "    Returns:"
_SECTION_HEADER_RE = re.compile(r"^(\s*)(" + "|".join(_ALL_SECTION_NAMES) + r")\s*:\s*$")

# Named entry: "name (type): desc" or "name: desc"
_NAMED_ENTRY_RE = re.compile(r"^\w[\w\d_]*\s*(?:\(.*?\))?\s*:")

# Typed entry (Returns/Yields): a type expression followed by ": desc"
# Type can include brackets, pipes, commas, dots, spaces within brackets
_TYPED_ENTRY_RE = re.compile(r"^[^\s:]+(?:\[.*?\])?(?:\s*\|\s*[^\s:]+(?:\[.*?\])?)*\s*:")

# Raises entry: "ExceptionName: desc"
_RAISES_ENTRY_RE = re.compile(r"^\w[\w\d_]*\s*:")


def _is_entry_line(stripped_line: str, section_name: str) -> bool:
    """Check whether a stripped line looks like the start of a new entry."""
    if section_name in _NAMED_SECTIONS:
        return bool(_NAMED_ENTRY_RE.match(stripped_line))
    if section_name in _TYPED_SECTIONS:
        return bool(_TYPED_ENTRY_RE.match(stripped_line))
    if section_name in _RAISES_SECTIONS:
        return bool(_RAISES_ENTRY_RE.match(stripped_line))
    return False


def find_continuation_indent_violations(
    docstring: str,
) -> list[tuple[str, int, str]]:
    """Find continuation lines in docstring sections with incorrect indentation.

    In Google-style docstrings, continuation lines within a section entry must
    be indented further than the entry line itself.

    Args:
        docstring (str): The raw docstring text.

    Returns:
        list[tuple[str, int, str]]: List of (section_name, line_number, line_text)
            for each violation found. Line numbers are 1-indexed within the
            docstring.
    """
    lines = docstring.split("\n")
    violations: list[tuple[str, int, str]] = []
    i = 0

    while i < len(lines):
        header_match = _SECTION_HEADER_RE.match(lines[i])
        if not header_match:
            i += 1
            continue

        section_name = header_match.group(2)
        section_indent = len(header_match.group(1))
        i += 1

        # Find the entry indent from first non-blank content line
        entry_indent = None
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            entry_indent = len(lines[i]) - len(lines[i].lstrip())
            break

        if entry_indent is None:
            continue

        # Process lines within this section
        in_entry = False
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Blank line, skip
            if not stripped:
                i += 1
                continue

            current_indent = len(line) - len(line.lstrip())

            # Left the section: dedented to or past section header level,
            # or hit a new section header
            if current_indent <= section_indent:
                break
            if _SECTION_HEADER_RE.match(line):
                break

            if current_indent == entry_indent:
                if _is_entry_line(stripped, section_name):
                    in_entry = True
                elif in_entry:
                    # Continuation at entry indent: violation
                    violations.append((section_name, i + 1, stripped))
            # Lines indented more than entry_indent are fine (proper continuation)

            i += 1

    return violations


# ── Annotation extraction ──────────────────────────────────────────────────


def _extract_function_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Extract parameter annotations from a function definition."""
    annotations: dict[str, str] = {}

    all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
    for arg in all_args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is not None:
            with contextlib.suppress(Exception):
                annotations[arg.arg] = ast.unparse(arg.annotation)

    # *args and **kwargs
    if node.args.vararg and node.args.vararg.annotation:
        with contextlib.suppress(Exception):
            annotations[f"*{node.args.vararg.arg}"] = ast.unparse(node.args.vararg.annotation)
    if node.args.kwarg and node.args.kwarg.annotation:
        with contextlib.suppress(Exception):
            annotations[f"**{node.args.kwarg.arg}"] = ast.unparse(node.args.kwarg.annotation)

    return annotations


def _extract_class_annotations(node: ast.ClassDef) -> dict[str, str]:
    """Extract attribute annotations from a class body (ast.AnnAssign nodes)."""
    annotations: dict[str, str] = {}
    for item in node.body:
        if not isinstance(item, ast.AnnAssign):
            continue
        if not isinstance(item.target, ast.Name):
            continue
        name = item.target.id
        # Skip private attributes and UPPER_CASE constants
        if name.startswith("_"):
            continue
        if name.isupper():
            continue
        try:
            type_str = ast.unparse(item.annotation)
        except Exception:
            continue
        # Skip ClassVar annotations
        if "ClassVar" in type_str:
            continue
        annotations[name] = type_str
    return annotations


def _resolve_class_annotations(
    node: ast.ClassDef,
    class_annotations: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Resolve class annotations including inherited ones from base classes."""
    # Start with base class annotations (in MRO order)
    resolved: dict[str, str] = {}
    for base in node.bases:
        base_name = None
        if isinstance(base, ast.Name):
            base_name = base.id
        elif isinstance(base, ast.Attribute):
            base_name = base.attr
        if base_name and base_name in class_annotations:
            resolved.update(class_annotations[base_name])

    # Override with own annotations (own class takes precedence)
    own = _extract_class_annotations(node)
    resolved.update(own)
    return resolved


# ── Internal helpers ────────────────────────────────────────────────────────


def _is_multiline_docstring(docstring: str) -> bool:
    """Return True if the docstring has a blank line (multi-line structured format)."""
    return "\n\n" in docstring.strip()


def _get_qualified_name(node: ast.AST, tree: ast.Module) -> str:
    """Get a qualified name like 'ClassName.method_name' for a node."""
    name = getattr(node, "name", "?")
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.ClassDef):
            continue
        for child in ast.iter_child_nodes(parent):
            if child is node:
                return f"{parent.name}.{name}"
    return name


def _matches_pattern(path: str, pattern: str) -> bool:
    """Check if a path matches a simple glob-like pattern."""
    from fnmatch import fnmatch

    return fnmatch(path, pattern)
