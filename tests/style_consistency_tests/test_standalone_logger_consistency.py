"""Consistency checks for standalone-script logger usage.

Worker-runtime standalones (``inference.py``, ``run.py``, etc.) are loaded
inside the worker subprocess that ``ToolInstance`` spawns; ``standalone_helpers/``
reaches them on the ``PYTHONPATH`` published by ``_build_subprocess_env``, so they
can ``from standalone_helpers import get_logger`` and route records through the
parent's structured-logging bridge.

``binary_config.py`` is the exception: it runs in the **setup-time subprocess**
that ``setup.sh`` invokes via ``python utils/install_binary.py <toolkit>``. There
is no parent drain thread to demultiplex tagged records there — setup.sh's stderr
is mirrored to ``PROTO_ENV_LOG_DIR`` directly. ``binary_config.py`` files therefore
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
# install_binary._load_tool_config, without a parent drain thread to demux tagged records.
# Records here flow through setup.sh's stderr to PROTO_ENV_LOG_DIR, not the bridge.
_SETUP_TIME_EXEMPT: set[Path] = {p for p in _ALL_SCRIPTS if p.name == "binary_config.py"}

# Standalone files imported verbatim from upstream third-party projects (we don't fork-and-edit; new entries require justification).
_VENDORED_EXEMPT: set[Path] = {
    # Upstream ipsae script copied verbatim; uses its own print-based output.
    _TOOLS_DIR / "structure_scoring" / "ipsae" / "standalone" / "ipsae.py",
    # PARADE model/data modules copied verbatim from autosome-ru/parade. They are kept
    # byte-for-byte so the published Lightning checkpoints unpickle against the exact
    # ``legnet_classifier.LegNetClassifier`` class the paper trained; editing them (even
    # to add a logger) risks breaking checkpoint compatibility.
    _TOOLS_DIR / "sequence_scoring" / "parade" / "standalone" / "legnet.py",
    _TOOLS_DIR / "sequence_scoring" / "parade" / "standalone" / "legnet_classifier.py",
    _TOOLS_DIR / "sequence_scoring" / "parade" / "standalone" / "pl_regressor.py",
    _TOOLS_DIR / "sequence_scoring" / "parade" / "standalone" / "utrdata_cl.py",
    _TOOLS_DIR / "sequence_scoring" / "parade" / "standalone" / "stability_data.py",
    # CodonFM/Encodon model, data, tokenizer, and inference modules copied verbatim from
    # NVIDIA-BioNeMo/CodonFM under ``standalone/src/``. Kept byte-for-byte so the published
    # Encodon safetensors checkpoints load against the exact module topology upstream trained;
    # editing them (even to add a logger) risks breaking checkpoint compatibility.
    *(_TOOLS_DIR / "masked_models" / "codonfm" / "standalone" / "src").rglob("*.py"),
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
