"""tests/style_consistency_tests/test_thread_deadline_propagation.py.

Tests that tool code hands work to other threads through ``propagate()``. The tool's deadline lives
in a ``ContextVar``, which ``ThreadPoolExecutor`` workers and ``threading.Thread`` targets do not
inherit, so a bare ``pool.submit(fn)`` silently runs ``fn`` with no timeout. Enforced statically
because most of these paths only run against live services.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = _REPO_ROOT / "proto_tools" / "tools"
_EXCLUDE_PARTS = ("standalone", "tool_envs")
_EXECUTORS = {"ThreadPoolExecutor"}
_SUBMIT_METHODS = {"submit", "map"}

_FIX_HINT = "wrap the callable in propagate() from proto_tools.utils, e.g. pool.submit(propagate(fn), ...)"


def _name(node: ast.expr) -> str | None:
    """Return the trailing name of ``x`` or ``a.b.x``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_propagated(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Call) and _name(node.func) == "propagate"


def _violations(tree: ast.AST) -> list[tuple[int, str]]:
    """Return ``(line, call)`` for every thread hand-off that skips ``propagate()``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            pools = {
                item.optional_vars.id
                for item in node.items
                if isinstance(item.context_expr, ast.Call)
                and _name(item.context_expr.func) in _EXECUTORS
                and isinstance(item.optional_vars, ast.Name)
            }
            for call in (n for stmt in node.body for n in ast.walk(stmt) if isinstance(n, ast.Call)):
                func = call.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in _SUBMIT_METHODS
                    and _name(func.value) in pools
                    and not _is_propagated(call.args[0] if call.args else None)
                ):
                    found.append((call.lineno, f"{_name(func.value)}.{func.attr}"))
        elif isinstance(node, ast.Call) and _name(node.func) == "Thread":
            target = next((kw.value for kw in node.keywords if kw.arg == "target"), None)
            if not _is_propagated(target):
                found.append((node.lineno, "threading.Thread"))
    return found


_SOURCES = sorted(path for path in _TOOLS_DIR.rglob("*.py") if not any(part in _EXCLUDE_PARTS for part in path.parts))


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_thread_handoffs_propagate_the_deadline(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations = _violations(tree)
    assert not violations, "\n".join(
        f"{path.relative_to(_REPO_ROOT)}:{line}: {call} without propagate(); {_FIX_HINT}" for line, call in violations
    )


def test_check_catches_a_bare_submit() -> None:
    """The scan must recognize the pattern it exists to forbid."""
    bare = ast.parse("with ThreadPoolExecutor() as pool:\n    pool.submit(fetch, url)\n")
    wrapped = ast.parse("with ThreadPoolExecutor() as pool:\n    pool.submit(propagate(fetch), url)\n")
    assert _violations(bare) == [(2, "pool.submit")]
    assert _violations(wrapped) == []
