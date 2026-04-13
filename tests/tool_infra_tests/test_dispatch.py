"""Tests for standalone inference dispatch() contracts."""

import ast
from pathlib import Path

import pytest

_TOOLS_ROOT = Path(__file__).resolve().parents[2] / "proto_tools" / "tools"

_INFERENCE_SCRIPTS = sorted(_TOOLS_ROOT.glob("**/standalone/inference.py"))
_SCRIPT_IDS = [str(p.relative_to(_TOOLS_ROOT)).replace("/", ".") for p in _INFERENCE_SCRIPTS]


@pytest.mark.parametrize("script_path", _INFERENCE_SCRIPTS, ids=_SCRIPT_IDS)
def test_dispatch_signature(script_path: Path):
    """Each inference.py must define dispatch() taking exactly one positional arg."""
    tree = ast.parse(script_path.read_text(), filename=str(script_path))
    rel = script_path.relative_to(_TOOLS_ROOT)

    dispatch_nodes = [n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef) and n.name == "dispatch"]
    assert dispatch_nodes, f"{rel} is missing a top-level dispatch() function"

    positional = dispatch_nodes[0].args.posonlyargs + dispatch_nodes[0].args.args
    assert len(positional) == 1, f"{rel}: dispatch() should take exactly 1 positional arg, got {len(positional)}"


# Scripts that don't route by operation (single-purpose CLI tools, test mocks)
_NO_OPERATION_ROUTING = {
    "testing/",
    "sequence_alignment/",
    "structure_alignment/",
    "orf_prediction/",
    "gene_annotation/",
    "structure_scoring/structure_metrics/",
    "sequence_scoring/segmasker/",
}

_OPERATION_SCRIPTS = [
    p
    for p in _INFERENCE_SCRIPTS
    if not any(str(p.relative_to(_TOOLS_ROOT)).startswith(prefix) for prefix in _NO_OPERATION_ROUTING)
]


def _has_operation_silent_default(func_body: list[ast.stmt]) -> str | None:
    """Return a description if dispatch() uses .get("operation", ...) or .pop("operation", ...) with a default."""
    for node in ast.walk(ast.Module(body=func_body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("get", "pop"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or first_arg.value != "operation":
            continue
        # .get("operation") with no default is fine; .get("operation", X) is not
        if len(node.args) >= 2 or node.keywords:
            return f'.{node.func.attr}("operation", ...) with silent default at line {node.lineno}'
    return None


@pytest.mark.parametrize(
    "script_path",
    _OPERATION_SCRIPTS,
    ids=[str(p.relative_to(_TOOLS_ROOT)).replace("/", ".") for p in _OPERATION_SCRIPTS],
)
def test_dispatch_operation_no_silent_default(script_path: Path):
    """dispatch() must not use .get()/.pop() with a default for the 'operation' key."""
    source = script_path.read_text()
    tree = ast.parse(source, filename=str(script_path))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "dispatch":
            violation = _has_operation_silent_default(node.body)
            assert violation is None, (
                f"{script_path.relative_to(_TOOLS_ROOT)}: dispatch() has {violation}. "
                f'Use input_dict["operation"] or kwargs.pop("operation") without a default.'
            )
            return

    pytest.fail(f"{script_path.relative_to(_TOOLS_ROOT)}: dispatch() not found")
