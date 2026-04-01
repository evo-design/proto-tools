"""tests/style_consistency_tests/test_docstring_style.py.

Tests that multi-line docstrings follow Google style with type annotations
matching the actual function signatures and class annotations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from docstring_parser import DocstringStyle
from docstring_parser import parse as parse_docstring

from tests.style_consistency_tests.helpers import (
    collect_docstrings_with_annotations,
    extract_returns_type,
    find_continuation_indent_violations,
    normalize_type,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Only scan source code, not tests
_SOURCE_DIRS = ["proto_tools"]

_EXCLUDE_PATTERNS = [
    "proto_tools/tools/*/standalone/*",
    "tool_envs/*",
]


def _get_docstrings_with_annotations():
    return collect_docstrings_with_annotations(_REPO_ROOT, _SOURCE_DIRS, _EXCLUDE_PATTERNS)


_DOCSTRINGS = _get_docstrings_with_annotations()


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    _DOCSTRINGS,
    ids=[f"{fp}::{n}" for fp, n, *_ in _DOCSTRINGS],
)
def test_docstring_types_match_signatures(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Docstring Args/Attributes types must be present and match signatures.

    Every parameter in Args: or Attributes: that has a type annotation in
    the function signature or class body must also have a matching type in
    the docstring (in parentheses after the param name).
    """
    try:
        parsed = parse_docstring(docstring, style=DocstringStyle.GOOGLE)
    except Exception:
        pytest.skip(f"Could not parse docstring for {file_path}::{name}")
        return

    violations = []

    for param in parsed.params:
        # Skip *args, **kwargs
        if param.arg_name.startswith("*"):
            continue

        # Look up the annotation from the signature/class
        sig_type = annotations.get(param.arg_name)
        if sig_type is None:
            # No annotation in signature, skip
            continue

        norm_sig = normalize_type(sig_type)

        if param.type_name is None:
            violations.append(f"  {param.arg_name}: missing type in docstring (should be '{norm_sig}')")
            continue

        norm_doc = normalize_type(param.type_name)

        if norm_doc != norm_sig:
            violations.append(
                f"  {param.arg_name}: docstring type '{param.type_name}' "
                f"!= signature type '{sig_type}' "
                f"(normalized: '{norm_doc}' vs '{norm_sig}')"
            )

    assert not violations, f"{file_path}::{name} has docstring param type mismatches:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    [item for item in _DOCSTRINGS if item[5] == "function" and item[4] is not None],
    ids=[f"{fp}::{n}" for fp, n, _, _, rt, nk, _ in _DOCSTRINGS if nk == "function" and rt is not None],
)
def test_docstring_return_type_matches_signature(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Docstring Returns: type must be present and match the return annotation.

    If a function has a return type annotation and a multi-line docstring with
    a Returns: section, the return type in the docstring must match.
    Skips functions returning None.
    """
    if return_type is None:
        pytest.skip("No return annotation")
        return

    norm_sig = normalize_type(return_type)

    # Skip -> None returns
    if norm_sig == "None":
        pytest.skip("Returns None")
        return

    try:
        parsed = parse_docstring(docstring, style=DocstringStyle.GOOGLE)
    except Exception:
        pytest.skip(f"Could not parse docstring for {file_path}::{name}")
        return

    if not parsed.returns:
        pytest.skip("No Returns: section in docstring")
        return

    doc_return_type = parsed.returns.type_name

    # Fallback: docstring_parser can't handle union types with spaces
    # (e.g., "str | None: description"), so we parse manually
    if doc_return_type is None:
        doc_return_type = extract_returns_type(docstring)

    if doc_return_type is None:
        pytest.fail(f"{file_path}::{name}: Returns: section missing type (should be '{norm_sig}')")
        return

    norm_doc = normalize_type(doc_return_type)

    assert norm_doc == norm_sig, (
        f"{file_path}::{name}: return type mismatch, "
        f"docstring '{doc_return_type}' != signature '{return_type}' "
        f"(normalized: '{norm_doc}' vs '{norm_sig}')"
    )


@pytest.mark.parametrize(
    "file_path, name, docstring, annotations, return_type, node_kind, own_annotations",
    _DOCSTRINGS,
    ids=[f"{fp}::{n}" for fp, n, *_ in _DOCSTRINGS],
)
def test_docstring_continuation_indentation(
    file_path: str,
    name: str,
    docstring: str,
    annotations: dict[str, str],
    return_type: str | None,
    node_kind: str,
    own_annotations: dict[str, str],
):
    """Continuation lines in docstring sections must be indented past the entry line.

    In Google-style docstrings, multi-line descriptions under Args:, Returns:,
    Raises:, Attributes:, and Yields: sections must indent continuation lines
    further than the entry's first line.
    """
    violations = find_continuation_indent_violations(docstring)
    if not violations:
        return

    details = "\n".join(f"  {section} line {lineno}: {text}" for section, lineno, text in violations)
    pytest.fail(f"{file_path}::{name} has continuation lines at entry indent (should be indented further):\n{details}")
