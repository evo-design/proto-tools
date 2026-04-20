"""tests/style_consistency_tests/test_python_version_consistency.py.

Consistency checks for tool standalone/python_version.txt files.

Every tool with a standalone/ directory must ship a python_version.txt —
the existence check below enforces this, and the format checks validate
every shipped file.
"""

from pathlib import Path

import pytest

from proto_tools.utils.tool_instance import ToolInstance

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "proto_tools"
_TOOLS_DIR = _PACKAGE_ROOT / "tools"
_SHARED_ENVS_DIR = _PACKAGE_ROOT / "shared_envs"


def _discover_python_version_files() -> list[Path]:
    """Find all python_version.txt files: per-tool standalones + shared envs."""
    files = list(_TOOLS_DIR.rglob("standalone/python_version.txt"))
    files += list(_SHARED_ENVS_DIR.glob("*/python_version.txt"))
    return sorted(files)


def _discover_standalone_dirs() -> list[Path]:
    """Find every standalone/ directory under proto_tools/tools that owns its env.

    Tools that opt into a shared env (via ``standalone/shared_env.txt``) get their
    ``python_version.txt`` from the shared env def, not from their own standalone
    dir. Skip those so the per-tool existence check only fires on real owners.
    """
    return sorted(p for p in _TOOLS_DIR.rglob("standalone") if p.is_dir() and not (p / "shared_env.txt").exists())


def _tool_id(path: Path) -> str:
    """Return a 'category/tool' (or 'shared_envs/<name>') identifier for a python_version.txt path."""
    try:
        rel = path.relative_to(_TOOLS_DIR)
        return f"{rel.parts[0]}/{rel.parts[1]}"
    except ValueError:
        rel = path.relative_to(_PACKAGE_ROOT)
        return f"{rel.parts[0]}/{rel.parts[1]}"


def _standalone_dir_id(path: Path) -> str:
    """Return 'category/tool' identifier for a standalone/ directory path."""
    rel = path.relative_to(_TOOLS_DIR)
    return f"{rel.parts[0]}/{rel.parts[1]}"


_ALL_FILES = _discover_python_version_files()
_ALL_IDS = [_tool_id(p) for p in _ALL_FILES]

_ALL_STANDALONE_DIRS = _discover_standalone_dirs()
_ALL_STANDALONE_IDS = [_standalone_dir_id(p) for p in _ALL_STANDALONE_DIRS]


@pytest.mark.parametrize("standalone_dir", _ALL_STANDALONE_DIRS, ids=_ALL_STANDALONE_IDS)
def test_every_standalone_tool_has_python_version_file(standalone_dir: Path) -> None:
    """Every tool with a standalone/ directory must pin its Python version."""
    assert (standalone_dir / "python_version.txt").exists(), (
        f"{standalone_dir}/python_version.txt is missing. "
        f"Every tool must pin its Python version — see notes/tool-environments.md."
    )


@pytest.mark.parametrize("version_file", _ALL_FILES, ids=_ALL_IDS)
def test_python_version_file_is_well_formed(version_file: Path) -> None:
    """Every shipped python_version.txt must parse cleanly via the canonical parser.

    The parser enforces the required `default` key, every value's
    `major.minor[.patch]` shape, and the `>=3.8` floor — so a single parse
    call covers the full format contract.
    """
    content = version_file.read_text()
    # Sentinel platform key that won't match any real override — exercises
    # the default-fallback path while still validating every override value.
    try:
        ToolInstance._parse_python_version(content, "test-platform-no-match", str(version_file))
    except (ValueError, RuntimeError) as e:
        pytest.fail(f"{version_file}:\n  {e}")


@pytest.mark.parametrize("version_file", _ALL_FILES, ids=_ALL_IDS)
def test_python_version_file_declares_default(version_file: Path) -> None:
    """Every shipped python_version.txt must explicitly declare a 'default' key.

    Redundant with the parser check above, but kept as a separate test so the
    failure message is precise: a tool author who forgets `default:` sees one
    test fail with "missing 'default' key" rather than a generic parse error.
    """
    content = version_file.read_text()
    has_default = False
    for raw in content.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, _ = line.partition(":")
        if sep and key.strip().lower() == "default":
            has_default = True
            break
    assert has_default, (
        f"{version_file} is missing required 'default' key. "
        f"Every python_version.txt must declare a default version like 'default: 3.11'."
    )
