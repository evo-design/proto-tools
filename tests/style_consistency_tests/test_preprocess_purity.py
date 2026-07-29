"""tests/style_consistency_tests/test_preprocess_purity.py.

Tests that no ``preprocess`` override mutates its config. Mutating ``self`` makes preprocess
non-idempotent — the caller's config carries the change into later calls — and makes per-chunk
preprocess order-dependent when a batch is fanned across workers. Overrides must return an
updated copy instead. Enforced here as well as at runtime so a preprocess that CI never
executes (GPU-only or standalone-env tools) still fails review.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIRS = ["proto_tools"]
_EXCLUDE_PARTS = ("standalone", "tool_envs")

_FIX_HINT = (
    "preprocess must not mutate its config; return an updated copy instead: "
    "return inputs, self.model_copy(update={...})"
)


def _preprocess_defs() -> list[tuple[Path, ast.FunctionDef]]:
    """Collect every ``def preprocess`` in the scanned source tree."""
    found: list[tuple[Path, ast.FunctionDef]] = []
    for source_dir in _SOURCE_DIRS:
        for path in (_REPO_ROOT / source_dir).rglob("*.py"):
            if any(part in _EXCLUDE_PARTS for part in path.parts):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            found.extend(
                (path, node)
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "preprocess"
            )
    return found


_PREPROCESS_DEFS = _preprocess_defs()


def _self_assignment_targets(func: ast.FunctionDef) -> list[str]:
    """Return the path of every assignment rooted at ``self`` inside ``func``.

    Walks attribute and subscript chains, so ``self.a = x``, ``self.a.b = x``,
    ``self.a[0] = x``, and ``self.a[0].b = x`` are all reported. In-place container methods
    (``self.a.append(x)``) are out of scope: they assign nothing, so neither this check nor
    the runtime ``__setattr__`` guard observes them.
    """
    violations: list[str] = []
    for node in ast.walk(func):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            root = target
            segments: list[str] = []
            while True:
                if isinstance(root, ast.Attribute):
                    segments.append(f".{root.attr}")
                    root = root.value
                elif isinstance(root, ast.Subscript):
                    segments.append("[...]")
                    root = root.value
                else:
                    break
            if isinstance(root, ast.Name) and root.id == "self" and segments:
                violations.append("self" + "".join(reversed(segments)))
    return violations


@pytest.mark.parametrize(
    "assignment,expected",
    [
        ("self.a = 1", "self.a"),
        ("self.a.b = 1", "self.a.b"),
        ("self.a[0] = 1", "self.a[...]"),
        ("self.a[0].b = 1", "self.a[...].b"),
        ("self.a += 1", "self.a"),
    ],
    ids=["attr", "nested_attr", "subscript", "subscript_attr", "augmented"],
)
def test_self_assignment_detection_covers_attribute_and_subscript_chains(assignment, expected):
    """Assignments rooted at self are reported through both attribute and subscript chains."""
    tree = ast.parse(f"class C:\n    def preprocess(self, inputs):\n        {assignment}\n        return inputs")
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))

    assert _self_assignment_targets(func) == [expected]


def test_self_assignment_detection_ignores_local_names():
    """Assignments not rooted at self are not reported."""
    tree = ast.parse(
        "class C:\n    def preprocess(self, inputs):\n"
        "        other = 1\n        other.x = 2\n        inputs.y = 3\n        return inputs"
    )
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))

    assert _self_assignment_targets(func) == []


def test_preprocess_definitions_found():
    """The scan finds preprocess overrides, so a passing suite is not an empty scan."""
    assert len(_PREPROCESS_DEFS) > 1


@pytest.mark.parametrize(
    "path,func",
    _PREPROCESS_DEFS,
    ids=[f"{p.relative_to(_REPO_ROOT)}:{f.lineno}" for p, f in _PREPROCESS_DEFS],
)
def test_preprocess_does_not_mutate_self(path: Path, func: ast.FunctionDef):
    """No preprocess override assigns to ``self`` or to an attribute reachable through it."""
    violations = _self_assignment_targets(func)
    assert not violations, (
        f"{path.relative_to(_REPO_ROOT)}:{func.lineno} preprocess assigns to {', '.join(violations)}. {_FIX_HINT}"
    )


@pytest.mark.parametrize(
    "path,func",
    _PREPROCESS_DEFS,
    ids=[f"{p.relative_to(_REPO_ROOT)}:{f.lineno}" for p, f in _PREPROCESS_DEFS],
)
def test_preprocess_returns_inputs_or_inputs_and_config(path: Path, func: ast.FunctionDef):
    """Every preprocess return is either bare inputs or a two-element ``(inputs, config)``."""
    returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
    assert returns, f"{path.relative_to(_REPO_ROOT)}:{func.lineno} preprocess has no return"
    for node in returns:
        if isinstance(node.value, ast.Tuple):
            assert len(node.value.elts) == 2, (
                f"{path.relative_to(_REPO_ROOT)}:{node.lineno} preprocess returned a "
                f"{len(node.value.elts)}-tuple; return either inputs, or (inputs, config)."
            )
