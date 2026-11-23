"""tests/style_consistency_tests/test_env_dependency_tests_use_run_in_env.py.

Guards the rule that a test needing a toolkit's dependencies runs inside that toolkit's
environment, rather than against a copy installed next to proto-tools.

The main environment has no ML dependencies. Using ``pytest.importorskip("torch")``
there skips tests when the dependency is absent, both in CI
(``pytest --cpu-only --no-envs``) and in local runs. Failures in those tests can go unnoticed.

This happened to eight objective tests in ``test_esmfold2_inversion.py``. They kept
calling ``bin_midpoints()`` without an argument after its signature gained ``n_bins``,
but the tests were skipped and the suite continued to pass.

Put the assertions in a ``_{thing}_env_checks.py`` file and use ``run_in_env`` to run it
inside the toolkit's environment with its dependencies. See the testing section of
CLAUDE.md.

Reserve ``importorskip`` for optional extras in the main environment. Add each one to
:data:`ALLOWED_OPTIONAL_IMPORTS` with a comment explaining why it is optional. The
allowlist is empty, and adding an entry requires review.
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

    Parse the source so examples inside docstrings are not mistaken for calls.
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
