"""Consistency checks for standalone-script logger usage.

Worker-runtime standalones (``inference.py``, ``run.py``, etc.) are loaded
inside the worker subprocess that ``ToolInstance`` spawns; ``standalone_helpers/``
is staged into the standalone dir alongside them by ``_copy_standalone_helpers``,
so they can ``from standalone_helpers import get_logger`` and route records
through the parent's structured-logging bridge.

``binary_config.py`` is the exception: it runs in the **setup-time subprocess**
that ``setup.sh`` invokes via ``python utils/install_binary.py <toolkit>``,
which executes ``install_binary._load_tool_config`` *before* any bridge staging.
``standalone_helpers`` isn't on ``sys.path`` in that subprocess, and there's no
parent drain thread to demultiplex tagged records anyway — setup.sh's stderr is
mirrored to ``PROTO_ENV_LOG_DIR`` directly. ``binary_config.py`` files therefore
don't use the bridge at all (they fall back to stdlib ``logging`` if they need
to log, and most don't log).

This module enforces the convention with three parametrized rules applied to
every ``.py`` file under ``proto_tools/tools/*/*/standalone/``:

1. **No standalone may call ``logging.getLogger(__name__)`` or
   ``getLogger(__name__)``** at module level.
2. **Every standalone must declare a module-level
   ``logger = get_logger(__name__)``** so future log calls are uniform — no
   "I have to add the import first" friction.
3. **The logger initializer must be ``get_logger``** imported from
   ``standalone_helpers`` (or ``standalone_helpers.proto_logging``).

Files exempted from all three rules are listed in ``_EXEMPT`` and split by
reason: ``_SETUP_TIME_EXEMPT`` (every ``binary_config.py``, see above) and
``_VENDORED_EXEMPT`` (upstream-vendored code we shouldn't fork). New
``_VENDORED_EXEMPT`` entries require justification.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "tools"
_HELPERS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "proto_tools"
    / "utils"
    / "standalone_helpers_source"
    / "standalone_helpers"
)


def _discover_standalone_scripts() -> list[Path]:
    """Find every Python file under ``proto_tools/tools/*/*/standalone/``.

    Skips ``__init__.py`` (package markers, no logger).
    """
    scripts: list[Path] = []
    for pattern in ("standalone/*.py", "standalone/**/*.py"):
        scripts.extend(_TOOLS_DIR.rglob(pattern))
    # Skip __init__.py and auto-generated standalone_helpers/ copies (source tree is the authoritative version; tested separately).
    return sorted(s for s in set(scripts) if s.name != "__init__.py" and "standalone_helpers" not in s.parts)


_ALL_SCRIPTS = _discover_standalone_scripts()

# binary_config.py runs at setup time in a subprocess that loads it via
# install_binary._load_tool_config — before standalone_helpers/ is staged into
# the standalone dir, and without a parent drain thread to demux tagged records.
# Records here flow through setup.sh's stderr to PROTO_ENV_LOG_DIR, not the bridge.
_SETUP_TIME_EXEMPT: set[Path] = {p for p in _ALL_SCRIPTS if p.name == "binary_config.py"}

# Standalone files imported verbatim from upstream third-party projects (we don't fork-and-edit; new entries require justification).
_VENDORED_EXEMPT: set[Path] = {
    # Upstream ipsae script copied verbatim; uses its own print-based output.
    _TOOLS_DIR / "structure_scoring" / "ipsae" / "standalone" / "ipsae.py",
}

_EXEMPT: set[Path] = _SETUP_TIME_EXEMPT | _VENDORED_EXEMPT

_ID = lambda p: str(p.relative_to(_TOOLS_DIR))  # noqa: E731


def _module_level_logger_assignment(tree: ast.Module) -> ast.Assign | None:
    """Return the module-level ``logger = ...(__name__)`` assignment, or None.

    Only matches ``logger`` (the convention); other names like ``log`` or
    ``LOGGER`` are flagged so the conventional name is enforced too.
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id != "logger":
            continue
        if not isinstance(node.value, ast.Call):
            continue
        # Must be called with a single arg matching __name__.
        if not (len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Name)):
            continue
        if node.value.args[0].id != "__name__":
            continue
        return node
    return None


def _imports_get_logger_from_helpers(tree: ast.Module) -> bool:
    """Whether the file does ``from standalone_helpers[.proto_logging] import ... get_logger ...``."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module not in {"standalone_helpers", "standalone_helpers.proto_logging"}:
            continue
        if any(alias.name == "get_logger" for alias in node.names):
            return True
    return False


@pytest.mark.parametrize("script_path", _ALL_SCRIPTS, ids=[_ID(p) for p in _ALL_SCRIPTS])
def test_standalone_does_not_use_logging_getLogger(script_path: Path) -> None:
    """``logger = logging.getLogger(__name__)`` and ``getLogger(__name__)`` are forbidden.

    Both produce loggers outside the ``worker.*`` namespace, so their records
    fall outside the structured-logging bridge in the parent's drain thread.
    """
    if script_path in _EXEMPT:
        pytest.skip(f"exempt: {script_path.relative_to(_TOOLS_DIR)}")
    src = script_path.read_text()
    # Parse so we don't trip on these strings appearing inside docstrings/comments.
    tree = ast.parse(src, filename=str(script_path))
    assignment = _module_level_logger_assignment(tree)
    if assignment is None:
        return  # No module-level logger — vacuously fine.
    call = assignment.value
    assert isinstance(call, ast.Call)
    func = call.func
    # Forbidden: ``logging.getLogger(__name__)`` (Attribute) and ``getLogger(__name__)`` (Name).
    if isinstance(func, ast.Attribute):
        is_logging_getLogger = (
            isinstance(func.value, ast.Name) and func.value.id == "logging" and func.attr == "getLogger"
        )
        assert not is_logging_getLogger, (
            f"{script_path.relative_to(_TOOLS_DIR)}: uses logging.getLogger(__name__); "
            "switch to ``from standalone_helpers import get_logger; logger = get_logger(__name__)``"
        )
    if isinstance(func, ast.Name):
        assert func.id != "getLogger", (
            f"{script_path.relative_to(_TOOLS_DIR)}: uses getLogger(__name__) "
            "(imported via ``from logging import getLogger``); switch to "
            "``from standalone_helpers import get_logger; logger = get_logger(__name__)``"
        )


@pytest.mark.parametrize("script_path", _ALL_SCRIPTS, ids=[_ID(p) for p in _ALL_SCRIPTS])
def test_standalone_declares_module_level_logger(script_path: Path) -> None:
    """Every standalone must have a module-level ``logger = get_logger(__name__)``.

    Also requires that ``get_logger`` is imported from ``standalone_helpers``
    (or ``standalone_helpers.proto_logging``). Uniform logger declaration means
    new log calls don't require touching imports first.
    """
    if script_path in _EXEMPT:
        pytest.skip(f"exempt: {script_path.relative_to(_TOOLS_DIR)}")
    src = script_path.read_text()
    tree = ast.parse(src, filename=str(script_path))
    assignment = _module_level_logger_assignment(tree)
    assert assignment is not None, (
        f"{script_path.relative_to(_TOOLS_DIR)}: missing module-level "
        "``logger = get_logger(__name__)``; add it near the top of the file."
    )
    call = assignment.value
    assert isinstance(call, ast.Call)
    func = call.func
    is_get_logger = isinstance(func, ast.Name) and func.id == "get_logger"
    assert is_get_logger, (
        f"{script_path.relative_to(_TOOLS_DIR)}: module-level logger must be initialized as "
        "``logger = get_logger(__name__)``"
    )
    assert _imports_get_logger_from_helpers(tree), (
        f"{script_path.relative_to(_TOOLS_DIR)}: missing ``from standalone_helpers import get_logger``"
    )


# ── standalone_helpers submodules: same convention via relative import ─────


def _discover_helper_submodules() -> list[Path]:
    """Find every ``.py`` under ``standalone_helpers/`` that should declare a logger.

    ``proto_logging.py`` is excluded — it defines ``get_logger`` and
    ``ProtoLogger`` and would create a self-import. ``__init__.py`` is the
    package wiring; it imports ``get_logger`` for re-export but doesn't need a
    module logger of its own.
    """
    return sorted(p for p in _HELPERS_DIR.glob("*.py") if p.name not in {"__init__.py", "proto_logging.py"})


_ALL_HELPERS = _discover_helper_submodules()


def _imports_get_logger_relative(tree: ast.Module) -> bool:
    """Whether the helper does ``from .proto_logging import ... get_logger ...``."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if (
            node.module == "proto_logging"
            and node.level == 1
            and any(alias.name == "get_logger" for alias in node.names)
        ):
            return True
    return False


@pytest.mark.parametrize(
    "helper_path",
    _ALL_HELPERS,
    ids=[p.name for p in _ALL_HELPERS],
)
def test_helper_declares_module_level_logger(helper_path: Path) -> None:
    """Every ``standalone_helpers`` submodule must declare ``logger = get_logger(__name__)``.

    Helpers run inside the same subprocess as the standalone, so their loggers
    flow through the same bridge. They use the relative import
    ``from .proto_logging import get_logger`` since they're inside the package.
    """
    src = helper_path.read_text()
    tree = ast.parse(src, filename=str(helper_path))
    assignment = _module_level_logger_assignment(tree)
    assert assignment is not None, f"{helper_path.name}: missing module-level ``logger = get_logger(__name__)``."
    call = assignment.value
    assert isinstance(call, ast.Call)
    func = call.func
    assert isinstance(func, ast.Name) and func.id == "get_logger", (
        f"{helper_path.name}: logger must be initialized as ``get_logger(__name__)``."
    )
    assert _imports_get_logger_relative(tree), f"{helper_path.name}: missing ``from .proto_logging import get_logger``."


@pytest.mark.parametrize("helper_path", _ALL_HELPERS, ids=[p.name for p in _ALL_HELPERS])
def test_helper_does_not_use_logging_getLogger(helper_path: Path) -> None:
    """Helpers must not call ``logging.getLogger`` or ``getLogger`` at module level either."""
    src = helper_path.read_text()
    tree = ast.parse(src, filename=str(helper_path))
    assignment = _module_level_logger_assignment(tree)
    if assignment is None:
        return
    call = assignment.value
    assert isinstance(call, ast.Call)
    func = call.func
    if isinstance(func, ast.Attribute):
        is_logging_getLogger = (
            isinstance(func.value, ast.Name) and func.value.id == "logging" and func.attr == "getLogger"
        )
        assert not is_logging_getLogger, (
            f"{helper_path.name}: uses logging.getLogger; "
            "switch to ``from .proto_logging import get_logger; logger = get_logger(__name__)``"
        )
    if isinstance(func, ast.Name):
        assert func.id != "getLogger", (
            f"{helper_path.name}: uses getLogger (from `from logging import getLogger`); "
            "switch to ``from .proto_logging import get_logger``"
        )
