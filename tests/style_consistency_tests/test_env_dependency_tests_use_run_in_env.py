"""tests/style_consistency_tests/test_env_dependency_tests_use_run_in_env.py.

Guards the rule that a test needing a toolkit's dependencies runs inside that toolkit's
environment, rather than against a copy installed next to proto-tools.

``pytest.importorskip("torch")`` reads like a safety net and is the opposite. The
orchestrator has no ML dependencies by design -- models live in isolated tool envs -- so
the import never succeeds there and the test does not fail, it *skips*: silently, green,
in CI (``pytest --cpu-only --no-envs``) and in an ordinary local run alike.

That is not hypothetical. Every objective test in ``test_esmfold2_inversion.py`` was
guarded this way, so eight of them spent months calling ``bin_midpoints()`` with no
argument, long after the signature gained ``n_bins``, while the suite stayed green. A
skipped test and a passing test are indistinguishable in the summary line.

The fix is ``run_in_env``: ship the assertions as a ``_{thing}_env_checks.py`` and run them
inside the toolkit's env, where its real dependencies are. See the testing section of
CLAUDE.md.

``importorskip`` stays legitimate for a genuinely optional extra that the orchestrator
could plausibly have -- add it to :data:`ALLOWED_OPTIONAL_IMPORTS` with a comment saying
why. The allowlist is deliberately empty: every entry should be argued for in review.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TESTS_ROOT = Path(__file__).parent.parent

#: Modules a test may legitimately ``importorskip``: genuinely optional extras, never a
#: dependency some toolkit owns. Empty by design -- adding one is a review decision.
ALLOWED_OPTIONAL_IMPORTS: frozenset[str] = frozenset()

#: Files that run *inside* a tool env, where the dependency is present and importing it is
#: the whole point. They are not collected by pytest in the base process.
_ENV_CHECK_SUFFIXES = ("_env_check.py", "_env_checks.py", "_checks.py")


def _collected_test_files() -> list[Path]:
    """Every file pytest collects in the base process, excluding in-env check scripts."""
    return sorted(
        path
        for path in _TESTS_ROOT.rglob("*.py")
        if not path.name.startswith("_") and not path.name.endswith(_ENV_CHECK_SUFFIXES)
    )


def _importorskipped_modules(source: str) -> list[tuple[int, str]]:
    """Every ``importorskip("mod")`` *call* in ``source``, as ``(line, module)``.

    Parsed rather than matched, so a docstring explaining this rule -- of which there are
    several, including this file's -- is not mistaken for a call.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        first = node.args[0]
        if name == "importorskip" and isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((node.lineno, first.value))
    return found


def test_no_test_importorskips_a_toolkit_dependency() -> None:
    """A toolkit's dependency is never importable in the base process, so this can only skip."""
    offenders: list[str] = []
    for path in _collected_test_files():
        for number, module in _importorskipped_modules(path.read_text()):
            if module not in ALLOWED_OPTIONAL_IMPORTS:
                offenders.append(f"{path.relative_to(_TESTS_ROOT)}:{number}: importorskip({module!r})")

    assert not offenders, (
        "These tests skip silently instead of running, because the orchestrator does not have the "
        "module they ask for. Move the assertions into a `_{thing}_env_checks.py` and run them with "
        "`run_in_env(<toolkit>, script=...)`, which executes them inside the toolkit's environment "
        "where the dependency actually lives. See the testing section of CLAUDE.md.\n"
        "If the module really is an optional extra rather than a toolkit's dependency, add it to "
        "ALLOWED_OPTIONAL_IMPORTS with a comment.\n  " + "\n  ".join(offenders)
    )
