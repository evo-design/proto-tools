"""tests/style_consistency_tests/test_readme_consistency.py"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_TOOLS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "bio_programming_tools"
    / "tools"
)

# Directories that are not tool categories
_EXCLUDED_DIRS = frozenset({
    "__pycache__", "infra", "utils", "testing", "mutagenesis",
})

# Every tool README must contain these exact H2 sections.
_REQUIRED_SECTIONS = [
    "Overview",
    "Biological Background",
    "When to Use",
    "How It Works",
    "Quick Start Examples",
    "Best Practices & Gotchas",
    "References",
    "Related Tools",
]

# Optional H2 sections that are allowed but not required.
_OPTIONAL_SECTIONS = frozenset({
    "Tool Catalog",
    "Model Variants",
    "Execution Modes",
    "Input Parameters",
    "Configuration",
    "Output Specification",
    "Interpreting Results",
    "Important Parameters",
})


def _slugify(text: str) -> str:
    """Convert snake_case to kebab-case slug."""
    return text.lower().replace("_", "-").replace(" ", "-")


def _discover_tool_readmes() -> list[Path]:
    """Find all tool-level README.md files (category/tool/README.md)."""
    readmes = []
    for readme in sorted(_TOOLS_DIR.rglob("README.md")):
        rel = readme.relative_to(_TOOLS_DIR)
        parts = rel.parts
        if len(parts) != 3:
            continue
        category, tool_name, _ = parts
        if category in _EXCLUDED_DIRS or tool_name in _EXCLUDED_DIRS:
            continue
        readmes.append(readme)
    return readmes


def _tool_id(readme: Path) -> str:
    """Return 'category/tool' identifier for a README path."""
    rel = readme.relative_to(_TOOLS_DIR)
    return f"{rel.parts[0]}/{rel.parts[1]}"


_ALL_READMES = _discover_tool_readmes()
_ALL_IDS = [_tool_id(r) for r in _ALL_READMES]


def _lines_outside_code_blocks(text: str) -> list[tuple[int, str]]:
    """Return (1-based line number, content) for lines outside fenced code blocks.

    Fence-boundary lines (``` ...) are excluded from results.
    """
    lines = text.split("\n")
    result = []
    in_fence = False
    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if not in_fence:
            result.append((i, line))
    return result


def _strip_inline_code(line: str) -> str:
    """Remove inline code spans (`...`) from a line."""
    return re.sub(r"`[^`]+`", "", line)


def _strip_inline_math(line: str) -> str:
    """Remove inline math spans ($...$) from a line."""
    return re.sub(r"\$[^$]+\$", "", line)


# ── Formatting ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_bare_math_notation(readme: Path) -> None:
    """Math subscripts/superscripts outside code blocks must be wrapped in $...$."""
    text = readme.read_text()
    violations = []
    for lineno, line in _lines_outside_code_blocks(text):
        cleaned = _strip_inline_code(line)
        cleaned = _strip_inline_math(cleaned)
        if re.search(r"[_^]\{", cleaned):
            violations.append(f"  line {lineno}: {line.strip()}")
    assert not violations, (
        f"{_tool_id(readme)}/README.md has bare math notation (wrap in $...$):\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_trailing_space_in_bold(readme: Path) -> None:
    """Bold markers (**...**) must not have trailing whitespace inside."""
    text = readme.read_text()
    violations = []
    for lineno, line in _lines_outside_code_blocks(text):
        # Match **<non-space>...<whitespace>** where the closing ** is NOT
        # followed by a word char or backtick (which would indicate it's
        # actually an opening ** for the next bold span, not a closing one).
        if re.search(r"\*\*\S[^*]*\s\*\*(?![\w`])", line):
            violations.append(f"  line {lineno}: {line.strip()}")
    assert not violations, (
        f"{_tool_id(readme)}/README.md has trailing space inside bold markers:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_code_blocks_have_language(readme: Path) -> None:
    """Opening code fences should specify a language (```python, not bare ```)."""
    text = readme.read_text()
    violations = []
    in_fence = False
    for i, line in enumerate(text.split("\n"), 1):
        if re.match(r"^\s*```", line):
            if not in_fence and re.match(r"^\s*```\s*$", line):
                violations.append(f"  line {i}")
            in_fence = not in_fence
    assert not violations, (
        f"{_tool_id(readme)}/README.md has code blocks without language specifiers:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_unclosed_code_fences(readme: Path) -> None:
    """Code fences must come in pairs (even count of ```)."""
    text = readme.read_text()
    fence_count = len(re.findall(r"^\s*```", text, re.MULTILINE))
    assert fence_count % 2 == 0, (
        f"{_tool_id(readme)}/README.md has {fence_count} code fences (odd = unclosed)"
    )


# ── Structure ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_starts_with_h1(readme: Path) -> None:
    """README must start with an H1 heading (after optional badge HTML)."""
    lines = readme.read_text().strip().split("\n")
    # Skip leading HTML badge line(s) and blank lines
    first_content = next(
        (l for l in lines if l.strip() and not l.strip().startswith("<")),
        "",
    )
    assert re.match(r"^# \S", first_content), (
        f"{_tool_id(readme)}/README.md must have an H1 heading, got: {first_content!r}"
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_has_doc_badge(readme: Path) -> None:
    """README must have a right-aligned Proto Docs badge linking to the correct page."""
    rel = readme.relative_to(_TOOLS_DIR)
    category, tool_name, _ = rel.parts
    expected_path = f"/tools/{_slugify(category)}/{_slugify(tool_name)}"

    text = readme.read_text()
    assert 'img.shields.io/badge/View_in_Proto_Docs' in text, (
        f"{_tool_id(readme)}/README.md is missing the Proto Docs badge"
    )
    assert 'align="right"' in text, (
        f"{_tool_id(readme)}/README.md badge must use align=\"right\""
    )
    assert expected_path in text, (
        f"{_tool_id(readme)}/README.md badge should link to {expected_path}"
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_has_required_sections(readme: Path) -> None:
    """README must contain all required H2 sections (exact match)."""
    text = readme.read_text()
    h2_headings = re.findall(r"^## (.+)$", text, re.MULTILINE)

    missing = []
    for req in _REQUIRED_SECTIONS:
        if req == "When to Use":
            # Allow "When to Use This Tool" or "When to Use These Tools"
            if not any(h2.startswith("When to Use") for h2 in h2_headings):
                missing.append(req)
        elif req not in h2_headings:
            missing.append(req)
    assert not missing, (
        f"{_tool_id(readme)}/README.md is missing required sections: {missing}"
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_only_recognized_sections(readme: Path) -> None:
    """All H2 sections must be from the required or optional lists."""
    text = readme.read_text()
    h2_headings = re.findall(r"^## (.+)$", text, re.MULTILINE)

    allowed = _OPTIONAL_SECTIONS | {
        s for s in _REQUIRED_SECTIONS if s != "When to Use"
    }
    unrecognized = []
    for h2 in h2_headings:
        if h2 in allowed:
            continue
        if h2.startswith("When to Use"):
            continue
        unrecognized.append(h2)
    assert not unrecognized, (
        f"{_tool_id(readme)}/README.md has unrecognized H2 sections: {unrecognized}. "
        f"Use one of: {sorted(allowed | {'When to Use This Tool'})}"
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_duplicate_h2(readme: Path) -> None:
    """README should not have duplicate H2 headings."""
    text = readme.read_text()
    h2_headings = re.findall(r"^## (.+)$", text, re.MULTILINE)

    seen: set[str] = set()
    duplicates = [h2 for h2 in h2_headings if h2 in seen or seen.add(h2)]  # type: ignore[func-returns-value]
    assert not duplicates, (
        f"{_tool_id(readme)}/README.md has duplicate H2 sections: {duplicates}"
    )


# ── Links ─────────────────────────────────────────────────────────────────


# Domains that genuinely do not support HTTPS.
_HTTP_ONLY_DOMAINS = frozenset({"hmmer.org", "eddylab.org"})


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_http_urls(readme: Path) -> None:
    """URLs should use https://, not plain http:// (unless the domain lacks HTTPS)."""
    text = readme.read_text()
    violations = []
    for lineno, line in _lines_outside_code_blocks(text):
        for match in re.finditer(r"http://(\S+)", line):
            domain = match.group(1).split("/")[0]
            if domain not in _HTTP_ONLY_DOMAINS:
                violations.append(f"  line {lineno}: {match.group()}")
    assert not violations, (
        f"{_tool_id(readme)}/README.md uses http:// instead of https://:\n"
        + "\n".join(violations)
    )


def _extract_urls(text: str) -> list[tuple[int, str]]:
    """Extract all URLs from text outside code blocks.

    Handles Wikipedia-style URLs with parentheses (e.g., BLAST_(biotechnology)).
    Returns deduplicated (1-based line number, url) tuples.
    """
    urls: list[tuple[int, str]] = []
    seen: set[str] = set()
    for lineno, line in _lines_outside_code_blocks(text):
        # First extract from markdown links [text](url) which may have parens
        # (e.g., Wikipedia URLs like BLAST_(biotechnology)#Algorithm)
        for match in re.finditer(
            r"\[(?:[^\]]*)\]\((https?://[^()\s]*(?:\([^)]*\)[^()\s]*)*)\)", line
        ):
            url = match.group(1).rstrip(".,;:")
            if url not in seen and "img.shields.io" not in url:
                seen.add(url)
                urls.append((lineno, url))
        # Then extract bare URLs (not inside markdown link parens)
        for match in re.finditer(r'(?<!\()(https?://[^\s)>\]"]+)', line):
            url = match.group(1).rstrip(".,;:")
            if url not in seen and "img.shields.io" not in url:
                seen.add(url)
                urls.append((lineno, url))
    return urls


# HTTP codes that indicate a clearly broken link (not just access-restricted).
_BROKEN_HTTP_CODES = {404, 410}


@pytest.mark.integration
@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_all_links_reachable(readme: Path) -> None:
    """All URLs in READMEs must be reachable (not 404 or DNS failure).

    Only flags clearly broken links. Access-restricted pages (401, 403, 502)
    are treated as reachable since the URL itself is valid.
    """
    text = readme.read_text()
    urls = _extract_urls(text)
    if not urls:
        return

    broken = []
    for lineno, url in urls:
        try:
            req = urllib.request.Request(
                url,
                method="HEAD",
                headers={"User-Agent": "bio-programming-tools-link-checker/1.0"},
            )
            urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as exc:
            if exc.code in _BROKEN_HTTP_CODES:
                broken.append(f"  line {lineno}: {url} (HTTP {exc.code})")
            elif exc.code == 405:
                # Server rejects HEAD; retry with GET
                try:
                    req = urllib.request.Request(
                        url,
                        headers={
                            "User-Agent": "bio-programming-tools-link-checker/1.0",
                        },
                    )
                    urllib.request.urlopen(req, timeout=10)
                except urllib.error.HTTPError as exc2:
                    if exc2.code in _BROKEN_HTTP_CODES:
                        broken.append(f"  line {lineno}: {url} (HTTP {exc2.code})")
                except Exception:
                    broken.append(f"  line {lineno}: {url} (GET fallback failed)")
        except (urllib.error.URLError, OSError) as exc:
            broken.append(f"  line {lineno}: {url} ({exc})")

    assert not broken, (
        f"{_tool_id(readme)}/README.md has broken links:\n"
        + "\n".join(broken)
    )


# ── Markdown hygiene ──────────────────────────────────────────────────────


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_excessive_blank_lines(readme: Path) -> None:
    """No more than 2 consecutive blank lines."""
    text = readme.read_text()
    # 4+ newlines in a row = 3+ consecutive blank lines
    assert not re.search(r"\n{4,}", text), (
        f"{_tool_id(readme)}/README.md has more than 2 consecutive blank lines"
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_table_column_consistency(readme: Path) -> None:
    """Markdown tables must have consistent column counts across rows."""
    text = readme.read_text()
    violations = []
    in_code = False
    table_start: int | None = None
    table_cols: int | None = None

    for i, line in enumerate(text.split("\n"), 1):
        if re.match(r"^\s*```", line):
            in_code = not in_code
            continue
        if in_code:
            continue

        if line.strip().startswith("|"):
            # Don't count escaped pipes (\|) as column separators
            col_count = line.replace("\\|", "").count("|") - 1
            if table_start is None:
                table_start = i
                table_cols = col_count
            elif col_count != table_cols:
                violations.append(
                    f"  line {i}: expected {table_cols} columns "
                    f"(from line {table_start}), got {col_count}"
                )
        else:
            table_start = None
            table_cols = None

    assert not violations, (
        f"{_tool_id(readme)}/README.md has inconsistent table columns:\n"
        + "\n".join(violations)
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_trailing_whitespace(readme: Path) -> None:
    """Non-blank lines should not have trailing whitespace."""
    text = readme.read_text()
    violations = []
    for i, line in enumerate(text.split("\n"), 1):
        if line != line.rstrip() and line.strip():
            violations.append(f"  line {i}")
            if len(violations) >= 5:
                break

    assert not violations, (
        f"{_tool_id(readme)}/README.md has trailing whitespace on lines:\n"
        + "\n".join(violations)
    )
