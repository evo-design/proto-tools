"""tests/tool_infra_tests/test_dispatch.py

Tests for standalone inference dispatch() contracts."""

import ast
from pathlib import Path

import pytest

_TOOLS_ROOT = Path(__file__).resolve().parents[2] / "bio_programming_tools" / "tools"

_INFERENCE_SCRIPTS = sorted(_TOOLS_ROOT.glob("**/standalone/inference.py"))


@pytest.mark.parametrize(
    "script_path",
    _INFERENCE_SCRIPTS,
    ids=[
        str(p.relative_to(_TOOLS_ROOT)).replace("/", ".")
        for p in sorted(_TOOLS_ROOT.glob("**/standalone/inference.py"))
    ],
)
def test_dispatch_function_exists(script_path: Path):
    """Each inference.py must define a top-level dispatch() function."""
    source = script_path.read_text()
    tree = ast.parse(source, filename=str(script_path))

    top_level_functions = {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef)
    }

    assert "dispatch" in top_level_functions, (
        f"{script_path.relative_to(_TOOLS_ROOT)} is missing a top-level dispatch() function"
    )


@pytest.mark.parametrize(
    "script_path",
    _INFERENCE_SCRIPTS,
    ids=[
        str(p.relative_to(_TOOLS_ROOT)).replace("/", ".")
        for p in sorted(_TOOLS_ROOT.glob("**/standalone/inference.py"))
    ],
)
def test_dispatch_takes_input_dict(script_path: Path):
    """dispatch() must accept a single positional argument (input_dict)."""
    source = script_path.read_text()
    tree = ast.parse(source, filename=str(script_path))

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "dispatch":
            args = node.args
            # Should have exactly one positional arg (input_dict) plus no *args
            positional = args.posonlyargs + args.args
            assert len(positional) == 1, (
                f"{script_path.relative_to(_TOOLS_ROOT)}: dispatch() should take "
                f"exactly 1 positional arg, got {len(positional)}"
            )
            return

    pytest.fail(f"{script_path.relative_to(_TOOLS_ROOT)}: dispatch() not found")
