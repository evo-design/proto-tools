"""Shared helpers for style-consistency tests."""

from __future__ import annotations

from typing import Iterable


def find_missing_fields_in_docstring(
    docstring: str, field_names: Iterable[str]
) -> list[str]:
    """Return field names not mentioned in the docstring."""
    return [name for name in field_names if name not in docstring]


def field_description_is_valid(description: str | None, max_length: int = 100) -> str:
    """Return an error string if the description is invalid, or '' if valid."""
    if description is None:
        return "is None"
    if len(description) > max_length:
        return (
            f"is too long (currently {len(description)} characters, "
            f"must be under {max_length} characters)"
        )
    if not description.strip():
        return "description is empty or just whitespace"
    if "\n" in description:
        return "description contains newline characters. Please use single line descriptions."
    return ""
