"""Consistency check for ``--platform`` forcing in standalone setup scripts.

A conda prefix holds packages from exactly one subdir. ``ToolInstance._create_env``
creates the tool env natively (``micromamba create -p $VENV_PATH python=...``), so a
later ``micromamba install --platform <other> -p "$VENV_PATH"`` drops foreign-arch
packages into a prefix whose python/openssl/readline are native. The build succeeds
and the tool then dies at first invocation in dyld, linking foreign executables
against native libraries.

Forcing a platform is only safe when the transaction *creates* its own prefix, so
every package inside it agrees. ``gene_annotation/crispr_tracr_rna`` is the
reference: it needs x86_64-only bioconda packages and puts them in
``$VENV_PATH/conda_deps`` via ``micromamba create``, leaving the tool env native.

The rule this enforces: ``--platform`` may not appear on a ``micromamba install``
that targets the tool env prefix itself. A package with no build for the host arch
belongs in its own prefix.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "tools"
_SHARED_ENVS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "shared_envs"

# ``VAR=(--platform osx-64)``; setup scripts build mamba flags in arrays like this.
_ARRAY_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=\((.*)\)\s*$")
# ``"${VAR[@]}"`` / ``${VAR[*]}`` after shlex has stripped the quotes.
_ARRAY_EXPANSION_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\[[@*]\]\}$")

# The tool env prefix that ToolInstance creates natively before running setup.sh.
_MAIN_PREFIX = {"$VENV_PATH", "${VENV_PATH}"}

_MAMBA_TOKENS = {"$MAMBA_BIN", "${MAMBA_BIN}", "micromamba", "mamba", "conda"}


def _discover_setup_scripts() -> list[Path]:
    """Find every ``setup.sh`` under a toolkit's ``standalone/`` or a shared env."""
    scripts = list(_TOOLS_DIR.rglob("standalone/setup.sh"))
    scripts.extend(_SHARED_ENVS_DIR.rglob("setup.sh"))
    return sorted(scripts)


_ALL_SETUP_SCRIPTS = _discover_setup_scripts()

_ID = lambda p: p.parent.parent.name if p.parent.name == "standalone" else p.parent.name  # noqa: E731


def _logical_lines(source: str) -> list[str]:
    """Join backslash-continued lines so a multi-line command reads as one."""
    return re.sub(r"\\\n\s*", " ", source).splitlines()


def _platform_forcing_arrays(lines: list[str]) -> set[str]:
    """Names of arrays assigned a ``--platform`` flag anywhere in the script."""
    forcing: set[str] = set()
    for line in lines:
        match = _ARRAY_ASSIGN_RE.match(line)
        if match and "--platform" in match.group(2):
            forcing.add(match.group(1))
    return forcing


def _tokenize(line: str) -> list[str] | None:
    """Split a line into shell words, or None when it isn't parseable."""
    try:
        return shlex.split(line, comments=True)
    except ValueError:
        return None


def _install_prefix(tokens: list[str]) -> str | None:
    """Prefix targeted by an ``install`` command; ``''`` when none is given."""
    for flag in ("-p", "--prefix"):
        if flag in tokens:
            index = tokens.index(flag)
            return tokens[index + 1] if index + 1 < len(tokens) else ""
    return ""


@pytest.mark.parametrize("setup_script", _ALL_SETUP_SCRIPTS, ids=[_ID(p) for p in _ALL_SETUP_SCRIPTS])
def test_no_platform_forcing_on_install_into_tool_env(setup_script: Path) -> None:
    """``--platform`` must not be forced onto an install into the tool env prefix.

    Mixing subdirs inside one conda prefix is unsupported: the env builds cleanly and
    fails at runtime with ``incompatible architecture`` from dyld. Install foreign-arch
    packages into their own prefix (``micromamba create -p "$VENV_PATH/<name>"``) instead.
    """
    lines = _logical_lines(setup_script.read_text())
    forcing_arrays = _platform_forcing_arrays(lines)

    for line in lines:
        tokens = _tokenize(line)
        if not tokens or tokens[0] not in _MAMBA_TOKENS:
            continue
        # Expand array references so flags assembled above the call are visible here.
        expanded: list[str] = []
        for token in tokens:
            match = _ARRAY_EXPANSION_RE.match(token)
            if match:
                if match.group(1) in forcing_arrays:
                    expanded.append("--platform")
                continue
            expanded.append(token)
        if "--platform" not in expanded or "install" not in expanded:
            continue
        prefix = _install_prefix(expanded)
        assert prefix not in _MAIN_PREFIX and prefix != "", (
            f"{setup_script.relative_to(_TOOLS_DIR.parent)}: forces --platform on an install into "
            f"the tool env prefix ({prefix or 'no -p given'}), which mixes subdirs in one conda "
            "prefix and fails at runtime in dyld. Create a separate prefix for the foreign-arch "
            "packages, as gene_annotation/crispr_tracr_rna does."
        )
