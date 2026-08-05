"""The remote-capability hook must be spelled the way the registry calls it.

A config that defines a differently named hook is never called. The tool still
runs, so nothing fails, and the restriction the hook expresses silently does not
apply. Three configs were in that state after ``device="cloud"`` became
``device="proto"``.
"""

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "proto_tools"

HOOK = "remote_unsupported_reason"


def _config_methods() -> list[tuple[Path, str]]:
    """Return every method whose name ends in ``_unsupported_reason``."""
    found = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found.extend(
            (path, node.name)
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name.endswith("_unsupported_reason")
        )
    return found


def test_only_one_hook_name_exists():
    """A stale name is dead code, and dead code here means an unenforced restriction."""
    wrong = [f"{p.relative_to(_PACKAGE_ROOT)}::{name}" for p, name in _config_methods() if name != HOOK]
    assert not wrong, f"these will never be called; the registry calls {HOOK!r}:\n  " + "\n  ".join(wrong)


def test_the_registry_calls_that_hook():
    """Guards against the inverse mistake of renaming the call site instead."""
    source = (_PACKAGE_ROOT / "tools" / "tool_registry.py").read_text()
    assert f".{HOOK}(" in source, f"tool_registry.py no longer calls {HOOK}"


@pytest.mark.parametrize("path,name", _config_methods(), ids=lambda v: getattr(v, "name", str(v)))
def test_every_hook_takes_the_device(path: Path, name: str):
    """A hook that ignores the device cannot distinguish proto from a caller's own workspace."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = [a.arg for a in node.args.args]
            assert "device" in args, f"{path.name}::{name} takes {args}, missing 'device'"
