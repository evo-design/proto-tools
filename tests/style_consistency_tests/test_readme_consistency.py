"""tests/style_consistency_tests/test_readme_consistency.py."""

import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from proto_tools.tools import ToolRegistry

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "tools"

# Directories that are not tool categories
_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "infra",
        "utils",
        "testing",
    }
)

# Every tool README must contain these exact H2 sections. Anything else is
# allowed; toolkits add whatever sections they need.
_REQUIRED_SECTIONS = [
    "Overview",
    "Background",
    "Tools",
]


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
        category, toolkit, _ = parts
        if category in _EXCLUDED_DIRS or toolkit in _EXCLUDED_DIRS:
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
    assert not violations, f"{_tool_id(readme)}/README.md has bare math notation (wrap in $...$):\n" + "\n".join(
        violations
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
    assert not violations, f"{_tool_id(readme)}/README.md has trailing space inside bold markers:\n" + "\n".join(
        violations
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
    assert not violations, f"{_tool_id(readme)}/README.md has code blocks without language specifiers:\n" + "\n".join(
        violations
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_unclosed_code_fences(readme: Path) -> None:
    """Code fences must come in pairs (even count of ```)."""
    text = readme.read_text()
    fence_count = len(re.findall(r"^\s*```", text, re.MULTILINE))
    assert fence_count % 2 == 0, f"{_tool_id(readme)}/README.md has {fence_count} code fences (odd = unclosed)"


# ── Structure ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_starts_with_h1(readme: Path) -> None:
    """README must start with an H1 heading (after optional badge HTML)."""
    lines = readme.read_text().strip().split("\n")
    # Skip leading HTML badge line(s) and blank lines
    first_content = next(
        (line for line in lines if line.strip() and not line.strip().startswith("<")),
        "",
    )
    assert re.match(r"^# \S", first_content), (
        f"{_tool_id(readme)}/README.md must have an H1 heading, got: {first_content!r}"
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_has_doc_badge(readme: Path) -> None:
    """README must have a right-aligned Proto Docs badge linking to the correct page."""
    rel = readme.relative_to(_TOOLS_DIR)
    category, toolkit, _ = rel.parts
    expected_path = f"/tools/{_slugify(category)}/{_slugify(toolkit)}"

    text = readme.read_text()
    assert "img.shields.io/badge/View_in_Proto_Docs" in text, (
        f"{_tool_id(readme)}/README.md is missing the Proto Docs badge"
    )
    assert 'align="right"' in text, f'{_tool_id(readme)}/README.md badge must use align="right"'
    assert expected_path in text, f"{_tool_id(readme)}/README.md badge should link to {expected_path}"


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_has_required_sections(readme: Path) -> None:
    """README must contain all required H2 sections (exact match)."""
    text = readme.read_text()
    h2_headings = re.findall(r"^## (.+)$", text, re.MULTILINE)

    missing = [req for req in _REQUIRED_SECTIONS if req not in h2_headings]
    assert not missing, f"{_tool_id(readme)}/README.md is missing required sections: {missing}"


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_duplicate_h2(readme: Path) -> None:
    """README should not have duplicate H2 headings."""
    text = readme.read_text()
    h2_headings = re.findall(r"^## (.+)$", text, re.MULTILINE)

    seen: set[str] = set()
    duplicates = [h2 for h2 in h2_headings if h2 in seen or seen.add(h2)]  # type: ignore[func-returns-value]
    assert not duplicates, f"{_tool_id(readme)}/README.md has duplicate H2 sections: {duplicates}"


def _toolkit_keys(readme: Path) -> list[str]:
    """Return the registry keys of every tool whose source file lives in this toolkit dir."""
    toolkit_dir = readme.parent
    return sorted(spec.key for spec in ToolRegistry.list_all() if spec.source_file.parent == toolkit_dir)


def _tools_section_h3s(text: str) -> list[str]:
    """Return the H3 heading text for every H3 inside the ``## Tools`` section.

    Headings stop being collected at the next H2 or end of file.
    """
    in_section = False
    headings: list[str] = []
    for line in text.split("\n"):
        h2 = re.match(r"^## (.+)$", line)
        if h2:
            in_section = h2.group(1).strip() == "Tools"
            continue
        if not in_section:
            continue
        h3 = re.match(r"^### (.+)$", line)
        if h3:
            headings.append(h3.group(1).strip())
    return headings


def _tools_section_h3_bodies(text: str) -> dict[str, str]:
    """Map each H3 in the ``## Tools`` section to its body text (everything until the next H3/H2)."""
    bodies: dict[str, str] = {}
    in_section = False
    current: str | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        h2 = re.match(r"^## (.+)$", line)
        if h2:
            if current is not None:
                bodies[current] = "\n".join(buf).strip()
                current = None
                buf = []
            in_section = h2.group(1).strip() == "Tools"
            continue
        if not in_section:
            continue
        h3 = re.match(r"^### (.+)$", line)
        if h3:
            if current is not None:
                bodies[current] = "\n".join(buf).strip()
            current = h3.group(1).strip()
            buf = []
            continue
        if current is not None:
            buf.append(line)
    if current is not None:
        bodies[current] = "\n".join(buf).strip()
    return bodies


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_tools_section_lists_all_tools(readme: Path) -> None:
    """The ``## Tools`` section must contain one H3 per registered tool, each with a body.

    Each H3 must reference the tool's registry key as inline code (e.g. ``### BLAST Search
    (`blast-search`)``). H3 entries that don't correspond to a registered tool are rejected.
    """
    keys = _toolkit_keys(readme)
    assert keys, f"{_tool_id(readme)} has no registered tools — toolkit may not export anything"

    text = readme.read_text()
    bodies = _tools_section_h3_bodies(text)
    h3_keys: dict[str, str] = {}
    for heading, body in bodies.items():
        for match in re.finditer(r"`([^`]+)`", heading):
            candidate = match.group(1)
            if candidate in keys:
                h3_keys[candidate] = body

    missing = [k for k in keys if k not in h3_keys]
    assert not missing, (
        f"{_tool_id(readme)}/README.md `## Tools` section is missing H3 entries for: {missing}. "
        f"Each tool needs its own H3 with the registry key in inline code, e.g. `` `{missing[0]}` ``."
    )

    stray = [h for h in bodies if not any(c in keys for c in re.findall(r"`([^`]+)`", h))]
    assert not stray, (
        f"{_tool_id(readme)}/README.md `## Tools` section has H3 entries without a matching registered tool key: "
        f"{stray}"
    )

    empty = [k for k, body in h3_keys.items() if not body]
    assert not empty, f"{_tool_id(readme)}/README.md `## Tools` section H3s have no description body: {empty}"


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
    assert not violations, f"{_tool_id(readme)}/README.md uses http:// instead of https://:\n" + "\n".join(violations)


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
        for match in re.finditer(r"\[(?:[^\]]*)\]\((https?://[^()\s]*(?:\([^)]*\)[^()\s]*)*)\)", line):
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


def _fetch_url(url: str, method: str = "HEAD", timeout: int = 10, attempts: int = 3) -> None:
    """Fetch ``url`` with ``method``, retrying transient network errors.

    Retries only on ``URLError``/``OSError`` (transient network glitches, DNS
    hiccups, SSL handshake timeouts) up to ``attempts`` times with a short
    linear backoff (1s, 2s, ...). ``HTTPError`` is propagated immediately
    because the server gave a definitive response — retrying won't change
    a 404 and just slows the test when a link is actually dead.
    """
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "proto-tools-link-checker/1.0"},
    )
    for attempt in range(attempts):
        try:
            urllib.request.urlopen(req, timeout=timeout)
            return
        except urllib.error.HTTPError:  # noqa: PERF203 — retry on transient errors
            raise
        except (urllib.error.URLError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)


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

    # Skip domains that are known-valid but unreliable for automated checks.
    # - bio-pro.mintlify.app: our own docs site, assume it's up
    # - doi.org: permanent academic identifiers, never break but resolver is slow
    # - proteininformationresource.org: valid PIR database, intermittently slow in CI
    _SKIP_DOMAINS = {"bio-pro.mintlify.app", "doi.org", "proteininformationresource.org"}
    # TODO: Remove when public
    _SKIP_URL_PREFIXES = ("https://github.com/proto-bio/proto-tools",)

    broken = []
    for lineno, url in urls:
        if any(domain in url for domain in _SKIP_DOMAINS):
            continue
        if url.startswith(_SKIP_URL_PREFIXES):
            continue
        try:
            _fetch_url(url, method="HEAD")
        except urllib.error.HTTPError as exc:
            if exc.code in _BROKEN_HTTP_CODES:
                broken.append(f"  line {lineno}: {url} (HTTP {exc.code})")
            elif exc.code == 405:
                # Server rejects HEAD; retry with GET
                try:
                    _fetch_url(url, method="GET")
                except urllib.error.HTTPError as exc2:
                    if exc2.code in _BROKEN_HTTP_CODES:
                        broken.append(f"  line {lineno}: {url} (HTTP {exc2.code})")
                except Exception:
                    broken.append(f"  line {lineno}: {url} (GET fallback failed)")
        except (urllib.error.URLError, OSError) as exc:
            broken.append(f"  line {lineno}: {url} ({exc})")

    assert not broken, f"{_tool_id(readme)}/README.md has broken links:\n" + "\n".join(broken)


# ── Markdown hygiene ──────────────────────────────────────────────────────


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_no_excessive_blank_lines(readme: Path) -> None:
    """No more than 2 consecutive blank lines."""
    text = readme.read_text()
    # 4+ newlines in a row = 3+ consecutive blank lines
    assert not re.search(r"\n{4,}", text), f"{_tool_id(readme)}/README.md has more than 2 consecutive blank lines"


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
                    f"  line {i}: expected {table_cols} columns (from line {table_start}), got {col_count}"
                )
        else:
            table_start = None
            table_cols = None

    assert not violations, f"{_tool_id(readme)}/README.md has inconsistent table columns:\n" + "\n".join(violations)


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

    assert not violations, f"{_tool_id(readme)}/README.md has trailing whitespace on lines:\n" + "\n".join(violations)
