"""tests/tool_infra_tests/test_torch_index_pruning.py.

Canary for PyTorch wheel-index pruning.

``download.pytorch.org/whl/<variant>`` drops old wheels over time, and uv's default
``first-index`` strategy will not look past it: once a package name appears on that index, uv
refuses to source *any* version of it from PyPI. A tool whose pinned torch (or whose torch's
transitive CUDA wheel) has since been pruned therefore stops installing, with nothing in this
repository having changed.

That makes the failure time-triggered rather than change-triggered, so the resolve check is marked
``integration``: it runs in the daily scheduled job, not as a gate on unrelated pull requests.
The parser-coverage check alongside it is offline and change-triggered, so it does gate PRs — a
newly added tool must not silently drop out of the canary's coverage.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from proto_tools.utils.compute_deps import _TORCH_COMPATIBILITY, _TORCH_INDEX_BASE
from proto_tools.utils.tool_instance import ToolInstance

PROTO_TOOLS_DIR = Path(__file__).resolve().parents[2] / "proto_tools"

# The canary always resolves for linux/x86_64, the platform every GPU tool here is built for.
PLATFORM_KEY = "linux-x86_64"

# Wheel variants compute_deps can hand a tool at build time. cu118 is excluded: it only applies to
# pre-CUDA-12 drivers (<525), which none of the GPU tools here support.
DYNAMIC_VARIANTS = tuple(sorted({entry[2] for entry in _TORCH_COMPATIBILITY.values()} - {"cu118"}))

# A requirement token naming torch itself -- not torch-geometric, torch_scatter, or borzoi-pytorch.
_TORCH_REQUIREMENT = re.compile(r"^torch(?:\[[^\]]+\])?(?:[=<>!~].*)?$")

# Flags that consume the following token, so it is never a requirement.
_VALUE_FLAGS = frozenset(
    {
        "-r",
        "-c",
        "-f",
        "-e",
        "--requirement",
        "--constraint",
        "--find-links",
        "--extra-index-url",
        "--index-url",
        "--index",
        "--index-strategy",
        "--python-version",
        "--python-platform",
        "--torch-backend",
        "--reinstall-package",
    }
)

_PROCESS_SUBSTITUTION = re.compile(r"<\([^)]*\)")
_INDEX_URL_FLAG = re.compile(r"--(?:extra-)?index-url[=\s]+[\"']?(\S*?)[\"']?(?:\s|$)")
_INDEX_STRATEGY_FLAG = re.compile(r"--index-strategy[=\s]+([\w-]+)")
_VARIANT_IN_URL = re.compile(r"download\.pytorch\.org/whl/(cu\d+|cpu|rocm[\d.]*)")
_ENV_DEFAULT = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}$", re.DOTALL)

# uv's message when a required distribution is absent from every index it is willing to consult.
_PRUNED_SIGNATURE = re.compile(
    r"(no versions? of|there are no versions|only .{0,80}? (?:is|are) available)", re.IGNORECASE
)
# Transient index trouble, which must not raise the same alarm as a genuine pruning event.
_TRANSIENT_SIGNATURE = re.compile(
    r"(error sending request|connection (?:reset|refused|closed)|timed out|timeout"
    r"|temporarily unavailable|502|503|504|429)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TorchInstall:
    """One torch-installing line found in a setup script.

    Attributes:
        label (str): Short identifier for the owning environment, e.g. ``"evo2"``.
        script (Path): Absolute path to the setup script.
        line_number (int): 1-indexed line the install appears on.
        spec (str): Torch requirement to resolve, e.g. ``"torch==2.6.0"``.
        index_variants (tuple[str, ...]): Wheel variants this install can be pointed at.
        index_strategy (str | None): Value of ``--index-strategy``, if the line sets one.
        python_version (str): Interpreter version the environment is built for.
    """

    label: str
    script: Path
    line_number: int
    spec: str
    index_variants: tuple[str, ...]
    index_strategy: str | None
    python_version: str


def _label_for(script: Path) -> str:
    """Derive a short, stable test id from a setup script's location.

    Args:
        script (Path): Absolute path to a ``setup.sh``.

    Returns:
        str: Identifier such as ``"evo2"``, ``"evo2-modal"``, or ``"biohub_esm-shared"``.
    """
    parts = script.parts
    if "standalone_overrides" in parts:
        return f"{parts[parts.index('standalone_overrides') - 1].removesuffix('_deployment')}-modal"
    if "shared_envs" in parts:
        return f"{parts[parts.index('shared_envs') + 1]}-shared"
    if "standalone" in parts:
        return parts[parts.index("standalone") - 1]
    return script.parent.name


def _strip_quotes(token: str) -> str:
    """Remove one layer of matching surrounding quotes."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _resolve_shell_default(expr: str) -> str:
    """Reduce a nested ``${VAR:-default}`` expansion to its innermost default.

    Args:
        expr (str): Shell expansion such as ``"${A:-${B:-torch==2.6.*}}"``.

    Returns:
        str: The innermost default, e.g. ``"torch==2.6.*"``.
    """
    current = _strip_quotes(expr.strip())
    while (match := _ENV_DEFAULT.fullmatch(current)) is not None:
        current = _strip_quotes(match.group(1).strip())
    return current


def _lookup_variable(name: str, text: str) -> str | None:
    """Find a shell variable's assigned value in a script.

    Args:
        name (str): Variable name, without ``$``.
        text (str): Full script source.

    Returns:
        str | None: The resolved value, or None if the variable is never assigned.
    """
    match = re.search(rf"^\s*(?:export\s+)?{re.escape(name)}=(.+?)\s*$", text, re.MULTILINE)
    return _resolve_shell_default(match.group(1)) if match else None


def _requirement_tokens(command: str) -> list[str]:
    """Extract positional requirement tokens from a pip install command.

    Args:
        command (str): The install command line, comments already stripped.

    Returns:
        list[str]: Tokens that sit in requirement position.
    """
    cleaned = _PROCESS_SUBSTITUTION.sub(" ", command)
    try:
        tokens = shlex.split(cleaned, comments=True)
    except ValueError:
        tokens = cleaned.split()

    requirements: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in _VALUE_FLAGS:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        requirements.append(token)
    # Drop the leading "uv pip install" / "pip install" verbs.
    while requirements and requirements[0] in {"uv", "pip", "install", "python", "python3", "-m"}:
        requirements.pop(0)
    return requirements


def _variants_for(command: str, text: str) -> tuple[str, ...] | None:
    """Determine which wheel variants an install line can target.

    Args:
        command (str): The install command line.
        text (str): Full script source, used to honour a forced index export.

    Returns:
        tuple[str, ...] | None: Variants to check, or None when the line delegates index
        selection to uv (``--torch-backend``), which this canary cannot reproduce off-GPU.
    """
    if "--torch-backend" in command:
        return None

    url_match = _INDEX_URL_FLAG.search(command)
    url = url_match.group(1) if url_match else ""

    # An install that names a variant outright is only ever exposed to that one.
    if (literal := _VARIANT_IN_URL.search(url)) is not None:
        return (literal.group(1),)

    # Otherwise the index comes from RECOMMENDED_TORCH_INDEX -- either forced by the script
    # (Modal overrides do this) or left to the host's driver at build time.
    if "RECOMMENDED_TORCH_INDEX" in url or not url:
        forced = _lookup_variable("RECOMMENDED_TORCH_INDEX", text)
        if forced and (literal := _VARIANT_IN_URL.search(forced)) is not None:
            return (literal.group(1),)
        return DYNAMIC_VARIANTS
    return DYNAMIC_VARIANTS


def _parse_script(script: Path) -> tuple[list[TorchInstall], list[tuple[int, str]]]:
    """Extract torch installs from one setup script.

    Args:
        script (Path): Absolute path to a ``setup.sh``.

    Returns:
        tuple[list[TorchInstall], list[tuple[int, str]]]: Parsed installs, and any line that
        looked like a torch install but could not be understood.
    """
    text = script.read_text(encoding="utf-8", errors="replace")
    version_file = script.parent / "python_version.txt"
    python_version = (
        ToolInstance._parse_python_version(version_file.read_text(encoding="utf-8"), PLATFORM_KEY, str(version_file))
        if version_file.is_file()
        else "3.12"
    )

    installs: list[TorchInstall] = []
    unparsed: list[tuple[int, str]] = []

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        is_helper = line.startswith("proto_install_pytorch")
        is_pip = ("pip install" in line) and not line.startswith("echo")
        if not (is_helper or is_pip):
            continue

        if is_helper:
            tokens = shlex.split(line, comments=True)[1:]
            positional = [t for t in tokens if not t.startswith("-")]
            raw_spec = positional[0] if positional else ""
            if raw_spec.startswith("$"):
                name = raw_spec.lstrip("$").strip("{}")
                resolved = _lookup_variable(name, text)
                if resolved is None:
                    unparsed.append((number, line))
                    continue
                raw_spec = resolved
            spec = _resolve_shell_default(raw_spec) if raw_spec else "torch"
            # A bare or empty first argument means "use the centralized recommendation".
            spec = spec or "torch"
        else:
            requirements = [_strip_quotes(t) for t in _requirement_tokens(line)]
            matches = [r for r in requirements if _TORCH_REQUIREMENT.fullmatch(r)]
            if not matches:
                continue  # installs torch-adjacent packages only (torch-geometric, etc.)
            if len(matches) > 1:
                unparsed.append((number, line))
                continue
            spec = matches[0]
            if "$" in spec:
                resolved = _lookup_variable(spec.lstrip("$").strip("{}"), text)
                spec = resolved or _resolve_shell_default(spec)

        if not spec or "$" in spec:
            unparsed.append((number, line))
            continue

        variants = _variants_for(line, text)
        if variants is None:
            continue  # uv picks the backend itself; not an index-pruning exposure

        strategy_match = _INDEX_STRATEGY_FLAG.search(line)
        installs.append(
            TorchInstall(
                label=_label_for(script),
                script=script,
                line_number=number,
                spec=spec,
                index_variants=variants,
                index_strategy=strategy_match.group(1) if strategy_match else None,
                python_version=python_version,
            )
        )

    return installs, unparsed


def _discover() -> tuple[list[TorchInstall], list[tuple[Path, int, str]]]:
    """Parse every setup script under ``proto_tools/``."""
    installs: list[TorchInstall] = []
    unparsed: list[tuple[Path, int, str]] = []
    for script in sorted(PROTO_TOOLS_DIR.glob("**/setup.sh")):
        found, bad = _parse_script(script)
        installs.extend(found)
        unparsed.extend((script, number, line) for number, line in bad)
    return installs, unparsed


TORCH_INSTALLS, UNPARSED_LINES = _discover()

CASES = [
    pytest.param(install, variant, id=f"{install.label}-{variant}")
    for install in TORCH_INSTALLS
    for variant in install.index_variants
]


def test_every_torch_install_line_is_parsed() -> None:
    """Every torch-installing line is understood, so no tool escapes the canary silently."""
    assert not UNPARSED_LINES, "unparsed torch install lines:\n" + "\n".join(
        f"  {script.relative_to(PROTO_TOOLS_DIR.parent)}:{number}: {line}" for script, number, line in UNPARSED_LINES
    )


def test_canary_covers_the_known_pinned_tools() -> None:
    """The pinned-torch tools named in notes/tool-environments.md are actually covered."""
    covered = {install.label for install in TORCH_INSTALLS}
    assert {"evo1", "evo2", "borzoi", "progen3", "germinal", "codonfm"} <= covered, (
        f"pinned-torch tools missing from canary coverage: {sorted(covered)}"
    )


def _compile(spec: str, index_url: str, strategy: str | None, python_version: str) -> subprocess.CompletedProcess[str]:
    """Run a dependency resolve against a wheel index without installing anything."""
    command = [
        "uv",
        "pip",
        "compile",
        "-",
        "-q",
        "--python-platform",
        "linux",
        "--python-version",
        python_version,
        "--extra-index-url",
        index_url,
    ]
    if strategy:
        command += ["--index-strategy", strategy]
    return subprocess.run(command, input=spec, capture_output=True, text=True, timeout=600, check=False)


@pytest.mark.integration
@pytest.mark.parametrize(("install", "variant"), CASES)
def test_pinned_torch_still_resolves(install: TorchInstall, variant: str) -> None:
    """Each tool's torch pin still resolves against the wheel index it would be built with."""
    # Not skipped when absent: a missing resolver must not read as "nothing has been pruned".
    # CI provisions uv via astral-sh/setup-uv before this job runs.
    if shutil.which("uv") is None:
        pytest.fail("uv is required for the torch index canary; install it with `pip install uv`")

    index_url = f"{_TORCH_INDEX_BASE}/{variant}"
    result = _compile(install.spec, index_url, install.index_strategy, install.python_version)
    if result.returncode != 0 and _TRANSIENT_SIGNATURE.search(result.stderr):
        result = _compile(install.spec, index_url, install.index_strategy, install.python_version)
        if result.returncode != 0 and _TRANSIENT_SIGNATURE.search(result.stderr):
            pytest.skip(f"{index_url} unreachable:\n{result.stderr.strip()[-500:]}")

    if result.returncode == 0:
        return

    location = f"{install.script.relative_to(PROTO_TOOLS_DIR.parent)}:{install.line_number}"
    hint = (
        " Add --index-strategy unsafe-best-match so uv may source the pruned wheel from PyPI."
        if _PRUNED_SIGNATURE.search(result.stderr) and install.index_strategy is None
        else ""
    )
    pytest.fail(
        f"{install.spec} no longer resolves against {index_url} ({location}).{hint}\n{result.stderr.strip()[-1500:]}"
    )
