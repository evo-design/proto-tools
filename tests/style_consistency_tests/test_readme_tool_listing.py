"""tests/style_consistency_tests/test_readme_tool_listing.py."""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TOOLS_DIR = _REPO_ROOT / "proto_tools" / "tools"
_README = _REPO_ROOT / "README.md"

_EXCLUDED_DIRS = frozenset({"__pycache__", "infra", "utils", "testing"})


def _discover_tools_on_disk() -> set[str]:
    """Find all tool directories at depth tools/{category}/{tool}/ containing __init__.py.

    Returns:
        set[str]: Relative paths like ``proto_tools/tools/causal_models/evo1``.
    """
    tools = set()
    for category in sorted(_TOOLS_DIR.iterdir()):
        if not category.is_dir() or category.name in _EXCLUDED_DIRS:
            continue
        for tool in sorted(category.iterdir()):
            if not tool.is_dir() or tool.name in _EXCLUDED_DIRS:
                continue
            if (tool / "__init__.py").exists():
                tools.add(str(tool.relative_to(_REPO_ROOT)))
    return tools


def _extract_tools_from_readme() -> set[str]:
    """Parse the README Available Tools ``<pre>`` block and extract tool href paths.

    Returns:
        set[str]: Paths like ``proto_tools/tools/causal_models/evo1`` (trailing slash stripped).
    """
    text = _README.read_text()

    # Find the <pre>...</pre> block after "Available Tools"
    match = re.search(r"## Available Tools.*?<pre>(.*?)</pre>", text, re.DOTALL)
    assert match, "README.md must have an 'Available Tools' section with a <pre> block"

    pre_block = match.group(1)

    # Extract all href values — they look like proto_tools/tools/category/tool/
    hrefs = re.findall(r'href="([^"]+)"', pre_block)

    # Keep only tool-level paths (depth = 4 parts: proto_tools/tools/category/tool)
    tools = set()
    for href in hrefs:
        cleaned = href.rstrip("/")
        parts = cleaned.split("/")
        if len(parts) == 4 and parts[0] == "proto_tools" and parts[1] == "tools":
            tools.add(cleaned)
    return tools


def test_readme_lists_all_tools() -> None:
    """Every tool on disk must be listed in the README, and every README link must exist on disk."""
    tools_on_disk = _discover_tools_on_disk()
    tools_in_readme = _extract_tools_from_readme()

    missing_from_readme = tools_on_disk - tools_in_readme
    assert not missing_from_readme, "Available tool not listed in README.md Available Tools section:\n" + "\n".join(
        f"  {t}" for t in sorted(missing_from_readme)
    )

    missing_from_disk = tools_in_readme - tools_on_disk
    assert not missing_from_disk, "Tool listed in README.md but not found on disk (no __init__.py):\n" + "\n".join(
        f"  {t}" for t in sorted(missing_from_disk)
    )
