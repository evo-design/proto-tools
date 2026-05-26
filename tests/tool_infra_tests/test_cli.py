"""tests/tool_infra_tests/test_cli.py.

Smoke tests for the ``proto-tools`` CLI entry point. Each verb maps to a
``ToolRegistry`` classmethod, so coverage here is intentionally thin —
just enough to catch breakage in argparse wiring, exit codes, and the
text-vs-JSON output toggle. Behavioral coverage of the underlying
functions lives in ``test_tool_docs.py``.

All tests invoke the CLI via the in-process ``main()`` rather than a
subprocess so they stay fast (no Python startup cost per call).
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from proto_tools.cli import main


def _run(*argv: str) -> tuple[int, str, str]:
    """Invoke ``main(argv)`` and capture (exit_code, stdout, stderr)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


# ── Discovery verbs ─────────────────────────────────────────────────────────


def test_list_default_outputs_text() -> None:
    code, out, _ = _run("list")
    assert code == 0
    assert "esm2-embedding" in out
    assert "[masked_models]" in out


def test_list_category_filter() -> None:
    code, out, _ = _run("list", "--category", "masked_models")
    assert code == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert all("[masked_models]" in line for line in lines)
    assert any(line.startswith("esm2-embedding") for line in lines)


def test_list_json_payload_is_valid() -> None:
    code, out, _ = _run("list", "--category", "masked_models", "--json")
    assert code == 0
    payload = json.loads(out)
    keys = [item["key"] for item in payload]
    assert "esm2-embedding" in keys


def test_categories_outputs_known_value() -> None:
    code, out, _ = _run("categories")
    assert code == 0
    assert "masked_models" in out.splitlines()


def test_catalog_json_groups_by_category() -> None:
    code, out, _ = _run("catalog", "--json")
    assert code == 0
    payload = json.loads(out)
    assert "masked_models" in payload
    assert any(item["key"] == "esm2-embedding" for item in payload["masked_models"])


# ── Per-tool docs ───────────────────────────────────────────────────────────


def test_docs_text_includes_canonical_sections() -> None:
    code, out, _ = _run("docs", "esm2-embedding")
    assert code == 0
    assert "ESM2 Embeddings" in out
    assert "Applications" in out
    assert "Usage Tips" in out
    assert "Toolkit Notes" in out


def test_docs_no_toolkit_notes_flag() -> None:
    code, out, _ = _run("docs", "esm2-embedding", "--no-toolkit-notes")
    assert code == 0
    assert "Toolkit Notes" not in out


def test_docs_accepts_run_function_name() -> None:
    code, out, _ = _run("docs", "run_esm2_embeddings")
    assert code == 0
    assert "esm2-embedding" in out


def test_docs_json_roundtrips_to_pydantic_payload() -> None:
    code, out, _ = _run("docs", "esm2-embedding", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["key"] == "esm2-embedding"
    assert payload["toolkit_notes"]


# ── Error paths ─────────────────────────────────────────────────────────────


def test_ambiguous_toolkit_exits_two() -> None:
    code, _, err = _run("docs", "esm2")
    assert code == 2
    assert "ambiguous" in err
    assert "esm2-embedding" in err  # candidate list should be in the message


def test_unknown_identifier_exits_two() -> None:
    code, _, err = _run("docs", "not-a-real-tool")
    assert code == 2
    assert "Could not resolve" in err


def test_unknown_section_exits_one() -> None:
    code, _, err = _run("section", "esm2", "Nonexistent Section")
    assert code == 1
    assert "not found" in err


# ── Schema / example-input ─────────────────────────────────────────────────


def test_schema_input_is_valid_json() -> None:
    code, out, _ = _run("schema", "esm2-embedding", "--input")
    assert code == 0
    payload = json.loads(out)
    assert "properties" in payload
    assert "sequences" in payload["properties"]


def test_example_input_is_valid_json() -> None:
    code, out, _ = _run("example-input", "esm2-embedding")
    assert code == 0
    payload = json.loads(out)
    assert "sequences" in payload


# ── Example notebook ───────────────────────────────────────────────────────


def test_example_renders_markdown_and_code_fences() -> None:
    code, out, _ = _run("example", "esm2-embedding")
    assert code == 0
    assert out.startswith("# example notebook:")
    assert "example.ipynb" in out
    assert "```python" in out


def test_example_missing_notebook_exits_one() -> None:
    code, _, err = _run("example", "mmseqs2-clustering")
    assert code == 1
    assert "No example notebook found" in err


# ── Model doc verbs ────────────────────────────────────────────────────────


@pytest.mark.parametrize("verb", ["input", "config", "output"])
def test_model_doc_verbs(verb: str) -> None:
    code, out, _ = _run(verb, "esm2-embedding")
    assert code == 0
    assert "ESM2Embeddings" in out
