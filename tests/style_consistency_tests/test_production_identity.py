"""Keep production-facing repository identity separate from development wiring."""

from pathlib import Path

from proto_tools.modal.client import PROTO_TOOLS_REPO

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_REPOSITORY = "https://github.com/evo-design/proto-tools"
_TEXT_SUFFIXES = {".bib", ".ipynb", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
_FORBIDDEN_MARKERS = {
    "proto-tools-dev": "private development repository",
    "github.com/adititm/proto-tools": "personal GitHub repository",
    "github.com/proto-bio/proto-tools": "legacy repository identity",
    "/home/adititm": "developer-specific home path",
    "/large_storage/hielab/userspace/adititm": "developer-specific workspace path",
}


def _production_files() -> list[Path]:
    """Return production package, metadata, documentation, and CI text files."""
    files = [_REPO_ROOT / ".gitmodules", _REPO_ROOT / "README.md", _REPO_ROOT / "pyproject.toml"]
    for directory in (_REPO_ROOT / ".github", _REPO_ROOT / "proto_tools"):
        files.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix in _TEXT_SUFFIXES)
    return sorted(path for path in files if path.exists())


def test_production_sources_exclude_development_wiring() -> None:
    """Private remotes and machine-local paths belong in the ProtoBench lock."""
    violations = []
    for path in _production_files():
        content = path.read_text(encoding="utf-8", errors="replace")
        for marker, description in _FORBIDDEN_MARKERS.items():
            if marker in content:
                violations.append(f"{path.relative_to(_REPO_ROOT)}: {description} ({marker})")

    assert not violations, "Production-facing files contain development wiring:\n" + "\n".join(violations)


def test_documented_installation_uses_official_repository() -> None:
    """Both base and MCP installation examples must work from official source."""
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    mcp_readme = (_REPO_ROOT / "proto_tools" / "mcp" / "README.md").read_text(encoding="utf-8")
    assert f"pip install git+{_OFFICIAL_REPOSITORY}.git" in readme
    assert f'pip install "proto-tools[mcp] @ git+{_OFFICIAL_REPOSITORY}.git"' in mcp_readme


def test_package_and_runtime_links_use_official_repository() -> None:
    """Package metadata and actionable runtime errors must share one identity."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        project_urls = tomllib.load(file)["project"]["urls"]

    assert project_urls["Homepage"] == _OFFICIAL_REPOSITORY
    assert project_urls["Repository"] == _OFFICIAL_REPOSITORY
    assert project_urls["Issues"] == f"{_OFFICIAL_REPOSITORY}/issues"
    assert PROTO_TOOLS_REPO == _OFFICIAL_REPOSITORY
