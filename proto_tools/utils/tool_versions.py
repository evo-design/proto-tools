"""proto_tools/utils/tool_versions.py.

Toolkit versions pinned to a hash of the toolkit's source.

Each toolkit's ``versions.yaml`` maps version to source hash, newest first. The hash
covers the code a toolkit's results depend on: every file in the toolkit directory
except docs, examples, and example data, the shared environment it builds from, its
category's shared modules, and the first-party modules defining its models. The toolkit
and category ``__init__.py`` files only re-export tools, so they are left out. Python
comments and blank lines are ignored, so only code edits move it. See
``notes/versioning.md``.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import logging
import subprocess
import tokenize
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

VERSIONS_FILE = "versions.yaml"

HASH_LENGTH = 12

Bump = Literal["patch", "minor", "major"]

# Framework modules every tool shares. Hashing them would move every toolkit's hash on
# any framework edit.
FRAMEWORK_PREFIXES = ("proto_tools/utils/", "proto_tools/tools/tool_registry.py")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

_TOOLS_DIR = PACKAGE_ROOT / "proto_tools" / "tools"

# Toolkit-root entries that document a toolkit rather than define its behaviour. The
# root __init__.py only re-exports the toolkit's tools.
_UNHASHED_ENTRIES = frozenset(
    {"README.md", "cite.bib", "links.yaml", "license.yaml", "examples", VERSIONS_FILE, "__init__.py"}
)

# Toolkit-root data files read only by a tool's example_input().
_EXAMPLE_DATA_PREFIX = "example_"

_DOC_SUFFIXES = frozenset({".md", ".ipynb"})


def defining_files(tool_key: str) -> list[Path]:
    """Return the first-party files defining a tool's behaviour and its models.

    Autodetected: the run function's own module, plus the defining module of every
    class in each model's MRO, so a shared base such as
    ``structure_prediction/shared_data_models.py`` is covered without being declared.

    Args:
        tool_key (str): Registry key of the tool.

    Returns:
        list[Path]: Sorted absolute paths under ``proto_tools/``, framework modules excluded.
    """
    from proto_tools.tools import ToolRegistry

    spec = ToolRegistry.get(tool_key)
    found: set[Path] = {Path(spec.source_file).resolve()}
    for model in (spec.input_model, spec.config_model, spec.output_model):
        for klass in getattr(model, "__mro__", ()):
            try:
                origin = inspect.getsourcefile(klass)
            except TypeError:
                continue
            if origin:
                found.add(Path(origin).resolve())

    keep = []
    for path in found:
        try:
            rel = path.relative_to(PACKAGE_ROOT).as_posix()
        except ValueError:
            continue  # third-party or stdlib
        if rel.startswith("proto_tools/") and not rel.startswith(FRAMEWORK_PREFIXES):
            keep.append(path)
    return sorted(keep)


def packaged_toolkit_dirs() -> list[Path]:
    """Return every versioned toolkit shipped with proto-tools.

    Excludes external toolkits and the ``testing`` category, whose mock tools only exercise
    the framework.

    Returns:
        list[Path]: Sorted toolkit directories.
    """
    from proto_tools.tools import ToolRegistry

    return sorted(
        {
            spec.toolkit_dir
            for spec in ToolRegistry.list_all()
            if not spec.is_external and spec.toolkit_dir.parent.name != "testing"
        }
    )


def load_versions(toolkit_dir: Path) -> dict[str, str]:
    """Read a toolkit's ``versions.yaml``.

    Parsed with every scalar as a string, so an all-digit hash stays a string.

    Args:
        toolkit_dir (Path): The toolkit directory.

    Returns:
        dict[str, str]: Version to source hash, newest first; empty if the file is missing.
    """
    path = toolkit_dir / VERSIONS_FILE
    if not path.is_file():
        return {}
    data = yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506 -- BaseLoader builds only str/list/dict
    return dict(data or {})


def hashed_files(toolkit_dir: Path) -> dict[str, Path]:
    """Return the files a toolkit's source hash covers, keyed by location-independent label.

    Inside a git checkout only tracked and untracked-but-not-ignored files count, so git-ignored
    build artifacts never move the hash.

    Args:
        toolkit_dir (Path): The toolkit directory.

    Returns:
        dict[str, Path]: Label (``toolkit/...``, ``env/...``, or, for a shared module, its
            ``proto_tools/...`` path) to absolute file path.
    """
    from proto_tools.tools import ToolRegistry
    from proto_tools.utils.tool_instance import ToolInstance

    toolkit_dir = toolkit_dir.resolve()
    files = {
        f"toolkit/{p.relative_to(toolkit_dir).as_posix()}": p
        for p in _walk(toolkit_dir, _UNHASHED_ENTRIES)
        if not (p.parent == toolkit_dir and p.name.startswith(_EXAMPLE_DATA_PREFIX))
    }

    try:
        env_dir, _env_name = ToolInstance._packaged_env_def(toolkit_dir.name)
    except ValueError:
        env_dir = None  # no standalone environment, e.g. an HTTP-only database fetch
    if env_dir is not None and not env_dir.resolve().is_relative_to(toolkit_dir):
        env_dir = env_dir.resolve()
        files |= {f"env/{p.relative_to(env_dir).as_posix()}": p for p in _walk(env_dir)}

    # A category's __init__.py only re-exports its toolkits, so it never changes behaviour.
    shared = (
        {path for path in toolkit_dir.parent.glob("*.py") if path.name != "__init__.py"}
        if toolkit_dir.parent.parent == _TOOLS_DIR
        else set()
    )
    for spec in ToolRegistry.list_all():
        if spec.toolkit_dir.resolve() == toolkit_dir:
            shared.update(path for path in defining_files(spec.key) if not path.is_relative_to(toolkit_dir))
    files |= {path.relative_to(PACKAGE_ROOT).as_posix(): path for path in shared}

    tracked = _tracked_files(toolkit_dir)
    if tracked is None:
        return {
            label: path for label, path in files.items() if not ToolInstance._HELPER_ARTIFACTS.intersection(path.parts)
        }
    return {label: path for label, path in files.items() if path in tracked}


@functools.cache
def source_hash(toolkit_dir: Path) -> str:
    """Hash a toolkit's source. Cached for the life of the process.

    Args:
        toolkit_dir (Path): The toolkit directory.

    Returns:
        str: The first ``HASH_LENGTH`` hex characters of a SHA-256 over ``hashed_files``.
    """
    digest = hashlib.sha256()
    for label, path in sorted(hashed_files(toolkit_dir).items()):
        content = _normalized(path)
        digest.update(f"{label}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()[:HASH_LENGTH]


@functools.cache
def toolkit_version(toolkit_dir: Path) -> str | None:
    """Return the version stamped onto every output of a toolkit's tools.

    When the source no longer matches the newest entry, the version gains a ``+dirty``
    suffix and a warning is logged once.

    Args:
        toolkit_dir (Path): The toolkit directory.

    Returns:
        str | None: The newest version in ``versions.yaml``, or None for a toolkit without
            one (a tool defined in a test or notebook, say), which is then not hashed.
    """
    versions = load_versions(toolkit_dir)
    if not versions:
        return None

    version, recorded = next(iter(versions.items()))
    live = source_hash(toolkit_dir)
    if live == recorded:
        return version
    logger.warning(
        "%s source does not match %s %s (hash %s, recorded %s); outputs are tagged %s+dirty.",
        toolkit_dir.name,
        VERSIONS_FILE,
        version,
        live,
        recorded,
        version,
    )
    return f"{version}+dirty"


def next_version(version: str, bump: Bump) -> str:
    """Return the version after ``version`` at the given bump level.

    Args:
        version (str): An ``X.Y.Z`` version.
        bump (Bump): Which part to increment; the parts after it reset to zero.

    Returns:
        str: The bumped ``X.Y.Z`` version.
    """
    major, minor, patch = (int(part) for part in version.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def record_source_hash(toolkit_dir: Path, bump: Bump | None) -> str | None:
    """Pin a toolkit's current source hash in its ``versions.yaml``.

    A toolkit without the file starts at ``1.0.0``.

    Args:
        toolkit_dir (Path): The toolkit directory.
        bump (Bump | None): Add a new top entry at this level, or None to replace the top
            entry's hash in place.

    Returns:
        str | None: The version now pinned to the current hash, or None if the top entry
            already matched and nothing was written.
    """
    versions = load_versions(toolkit_dir)
    live = source_hash(toolkit_dir)
    if not versions:
        versions = {"1.0.0": live}
    else:
        version, recorded = next(iter(versions.items()))
        if live == recorded:
            return None
        if bump is None:
            versions[version] = live
        else:
            versions = {next_version(version, bump): live, **versions}
    (toolkit_dir / VERSIONS_FILE).write_text("".join(f"{v}: {h}\n" for v, h in versions.items()))
    return next(iter(versions))


def _walk(root: Path, top_level_skip: frozenset[str] = frozenset()) -> Iterator[Path]:
    """Yield the files under ``root`` that can define behaviour."""
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if parts[0] in top_level_skip or not path.is_file():
            continue
        if "__pycache__" in parts or path.name.startswith(".") or path.suffix in {".pyc", *_DOC_SUFFIXES}:
            continue
        yield path


def _tracked_files(toolkit_dir: Path) -> frozenset[Path] | None:
    """Return the source files of the git work tree holding ``toolkit_dir``, or None outside one.

    None also when git is unavailable or the work tree does not track the toolkit, as when
    proto-tools is installed into a virtualenv that sits inside some other repository.
    """
    repo = _repo_root(toolkit_dir)
    tracked = _git_files(repo) if repo is not None else None
    if tracked is None or not any(path.is_relative_to(toolkit_dir) for path in tracked):
        return None
    return tracked


@functools.cache
def _repo_root(directory: Path) -> Path | None:
    """Return the git work tree containing ``directory``, or None outside one."""
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


@functools.cache
def _git_files(repo: Path) -> frozenset[Path] | None:
    """Return a work tree's tracked and untracked-but-not-ignored files, or None if git fails."""
    try:
        listing = subprocess.run(  # noqa: S603 -- fixed git argv, repo path from the filesystem
            ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return frozenset(repo / name for name in listing.decode().split("\0") if name)


def _normalized(path: Path) -> bytes:
    """Return file content with line endings normalized and, for Python, comments and blank lines removed."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if path.suffix != ".py":
        return data
    try:
        return _strip_comments(data.decode()).encode()
    except (UnicodeDecodeError, SyntaxError, tokenize.TokenError):
        return data


def _strip_comments(source: str) -> str:
    """Drop comments, trailing whitespace, and blank lines from Python source.

    Comment positions come from ``tokenize``, which reports them identically on every
    supported Python, unlike the token stream itself (3.12 splits f-strings).
    """
    lines = source.split("\n")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col]
    return "\n".join(line.rstrip() for line in lines if line.strip())
