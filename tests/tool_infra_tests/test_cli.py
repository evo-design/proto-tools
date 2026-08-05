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


# ── Agent context ───────────────────────────────────────────────────────────


def test_agent_context_prints_primer() -> None:
    code, out, _ = _run("agent-context")
    assert code == 0
    assert "Input -> Config -> run_*() -> Output" in out
    assert "github.com/evo-design/proto-tools/tree/main/notes" in out


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


# ── Signature ──────────────────────────────────────────────────────────────


def test_signature_names_symbols_the_key_cannot_predict() -> None:
    """The point of the verb: registry key 'blast-create-db' does not yield these names."""
    code, out, _ = _run("signature", "blast-create-db")
    assert code == 0
    assert "CreateBlastDbInput" in out  # not BlastCreateDbInput
    assert "run_create_blast_db" in out  # not run_blast_create_db
    assert "CreateBlastDbConfig" in out
    assert "CreateBlastDbOutput" in out
    assert "proto_tools.tools.sequence_alignment.blast.create_blast_db" in out


def test_signature_names_the_output_without_importing_it() -> None:
    """Callers never construct an Output, so importing it would leave an unused name."""
    code, out, _ = _run("signature", "ipsae-scoring")
    assert code == 0
    imports, call = out.split("\n\nresult =", maxsplit=1)
    assert "IPSAEScoringOutput" not in imports
    assert "-> IPSAEScoringOutput" in call


def test_signature_lists_required_input_fields_only() -> None:
    code, out, _ = _run("signature", "ipsae-scoring")
    assert code == 0
    assert "IPSAEScoringInput(structure=..., binder_chain=..., target_chains=...)" in out


def test_signature_resolves_a_run_function_name() -> None:
    code, out, _ = _run("signature", "run_create_blast_db")
    assert code == 0
    assert "CreateBlastDbInput" in out


def test_signature_stays_small_for_a_tool_with_a_huge_example() -> None:
    """The reason the verb exists: cost is fixed, not proportional to the example payload.

    borzoi's example input is a 524,288 bp window (the model context length, enforced by
    ``BorzoiInput``), so ``example-input`` cannot be made cheap for it at any fixture size.
    """
    code, signature, _ = _run("signature", "borzoi-prediction")
    assert code == 0
    _, example, _ = _run("example-input", "borzoi-prediction")
    assert len(signature) < 1024
    assert len(example) > 500_000


def test_signature_json_carries_symbols_and_modules() -> None:
    code, out, _ = _run("signature", "blast-create-db", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["input_class"] == "CreateBlastDbInput"
    assert payload["run_function"] == "run_create_blast_db"
    assert payload["output_class"] == "CreateBlastDbOutput"
    assert payload["required_input_fields"] == ["fasta"]
    assert payload["modules"]["input_class"].startswith("proto_tools.tools.")


@pytest.mark.extensive
def test_signature_parses_and_imports_resolve_for_every_tool() -> None:
    """Every rendered signature is valid Python whose imports bind the names it uses."""
    import ast

    from proto_tools.tools.tool_registry import ToolRegistry

    specs = {s.key: s for s in [*ToolRegistry.list_cpu_tools(), *ToolRegistry.list_gpu_tools()]}
    failures: list[str] = []
    for key, spec in sorted(specs.items()):
        code, out, _ = _run("signature", key)
        if code != 0:
            failures.append(f"{key}: exit {code}")
            continue
        try:
            tree = ast.parse(out)
        except SyntaxError as exc:
            failures.append(f"{key}: {exc}")
            continue
        imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        namespace: dict[str, object] = {}
        try:
            exec(compile(ast.Module(body=imports, type_ignores=[]), "<generated>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        failures.extend(
            f"{key}: emitted import did not bind {symbol}"
            for symbol in (spec.input_model.__name__, spec.config_model.__name__, spec.function.__name__)
            if symbol not in namespace
        )

    assert not failures, "signatures with unusable imports:\n" + "\n".join(failures)


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


def test_example_input_as_python_runs_verbatim() -> None:
    """The emitted snippet is executable as-is, not just syntactically valid."""
    code, out, _ = _run("example-input", "random-protein-sample", "--as-python")
    assert code == 0

    namespace: dict[str, object] = {}
    exec(out, namespace)  # noqa: S102 -- executing our own generated snippet is the assertion
    assert namespace["result"].success is True  # type: ignore[union-attr]


def test_example_input_as_python_names_symbols_the_key_cannot_predict() -> None:
    """The point of the flag: registry key 'blast-create-db' does not yield these names."""
    code, out, _ = _run("example-input", "blast-create-db", "--as-python")
    assert code == 0
    assert "CreateBlastDbInput" in out  # not BlastCreateDbInput
    assert "run_create_blast_db" in out  # not run_blast_create_db
    assert "CreateBlastDbConfig" in out


@pytest.mark.extensive
def test_example_input_as_python_imports_resolve_for_every_tool() -> None:
    """Every generated snippet parses and its imports bind the names it then uses."""
    import ast

    from proto_tools.tools.tool_registry import ToolRegistry

    specs = {s.key: s for s in [*ToolRegistry.list_cpu_tools(), *ToolRegistry.list_gpu_tools()]}
    failures: list[str] = []
    for key, spec in sorted(specs.items()):
        code, out, _ = _run("example-input", key, "--as-python")
        if code != 0:
            continue  # tools without an example input are covered by the JSON path
        try:
            tree = ast.parse(out)
        except SyntaxError as exc:
            failures.append(f"{key}: {exc}")
            continue
        imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
        namespace: dict[str, object] = {}
        try:
            exec(compile(ast.Module(body=imports, type_ignores=[]), "<generated>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        failures.extend(
            f"{key}: emitted import did not bind {symbol}"
            for symbol in (spec.input_model.__name__, spec.function.__name__)
            if symbol not in namespace
        )

    assert not failures, "generated snippets with unusable imports:\n" + "\n".join(failures)


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


# ── doctor ─────────────────────────────────────────────────────────────────


def _modal_auth(monkeypatch, tmp_path, *, works: bool, from_env_vars: bool = False) -> None:
    """Pin Modal's credential state, so doctor is tested against it and not against this host."""
    import modal

    def result(*args, **kwargs):
        if works:
            return object()
        raise RuntimeError("AuthError: Token missing.")

    monkeypatch.setattr(modal.Client, "from_env", result)
    monkeypatch.setattr("proto_tools.modal.status.deployed_apps", lambda: {"esm2", "boltz2"})
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "absent.toml"))
    for var in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        if from_env_vars:
            monkeypatch.setenv(var, "placeholder")
        else:
            monkeypatch.delenv(var, raising=False)


def test_doctor_names_both_environment_variables_when_modal_is_unreachable(monkeypatch, tmp_path) -> None:
    """The remedy is the whole point of the verb: `modal token new` is unactionable in a sandbox."""
    _modal_auth(monkeypatch, tmp_path, works=False)
    code, out, err = _run("doctor")

    assert code == 1
    assert "MODAL_TOKEN_ID" in err and "MODAL_TOKEN_SECRET" in err
    assert "not found" in out
    assert "workspace" not in out, "nothing may claim a workspace it could not authenticate to"


def test_doctor_reports_which_mechanism_authenticated(monkeypatch, tmp_path) -> None:
    """Which of the three mechanisms was used is the fact that ends the diagnosis."""
    _modal_auth(monkeypatch, tmp_path, works=True, from_env_vars=True)
    code, out, _ = _run("doctor")

    assert code == 0
    assert "OK via MODAL_TOKEN_ID/MODAL_TOKEN_SECRET" in out
    assert "boltz2, esm2" in out, "deployed apps are listed so the caller sees what is reachable"


def test_doctor_json_carries_the_remedy_for_a_bug_report(monkeypatch, tmp_path) -> None:
    """The JSON form is what gets pasted into an issue, so it must carry the fix, not just the fault."""
    _modal_auth(monkeypatch, tmp_path, works=False)
    code, out, _ = _run("doctor", "--json")

    payload = json.loads(out)
    assert code == 1
    assert "not found" in payload["checks"]["modal auth"]
    assert any("MODAL_TOKEN_ID" in r for r in payload["remedies"])
