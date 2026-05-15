"""tests/style_consistency_tests/test_license_consistency.py.

Tests for per-toolkit license metadata: ToolRegistry surface and license.yaml
schema enforcement.

Every toolkit directory under proto_tools/tools/{category}/{toolkit}/ must
ship a license.yaml capturing license metadata. SPDX-allowlisted licenses
resolve their text from proto_tools/tools/_licenses/{spdx}.txt (one shared
copy, no inline duplication); 'Custom (...)' licenses include inline text.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest
import yaml

from proto_tools.tools.tool_registry import ToolRegistry

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "proto_tools" / "tools"

# Canonical SPDX license texts live here, deduplicated across toolkits.
_LICENSES_DIR = _TOOLS_DIR / "_licenses"

# Repo-root README, home of the "Gated model access" table.
_ROOT_README = _TOOLS_DIR.parent.parent / "README.md"

# Directories under proto_tools/tools/ that aren't tool categories.
_EXCLUDED_DIRS = frozenset({"__pycache__", "infra", "utils", "testing"})

# SPDX identifiers we accept. Any non-SPDX license must use a "Custom (...)"
# string instead.
_ALLOWED_SPDX = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "GPL-3.0",
        "LGPL-3.0",
        "MPL-2.0",
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-4.0",
        "AGPL-3.0",
        "ISC",
        "Unlicense",
    }
)

# Required top-level keys in license.yaml.
_REQUIRED_TOP_KEYS = {"code", "commercial_use", "redistribution", "last_updated"}

# Required keys inside a code/weights block.
_REQUIRED_TERMS_KEYS = {"spdx", "url"}

# Allowed values for commercial_use (tri-state string).
_ALLOWED_COMMERCIAL_USE = frozenset({"yes", "no", "restricted"})

# Allowed values for the optional weights.access field.
_ALLOWED_WEIGHTS_ACCESS = frozenset({"hf-gated", "request"})

# ISO date format (YYYY-MM-DD) for last_updated.
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _discover_toolkit_dirs() -> list[Path]:
    """Find every category/toolkit directory under proto_tools/tools/."""
    toolkits: list[Path] = []
    for category_dir in sorted(_TOOLS_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name in _EXCLUDED_DIRS:
            continue
        if category_dir.name.startswith("_"):
            continue
        for toolkit_dir in sorted(category_dir.iterdir()):
            if not toolkit_dir.is_dir() or toolkit_dir.name in _EXCLUDED_DIRS:
                continue
            if toolkit_dir.name.startswith("_"):
                continue
            # Real toolkits ship a README.md; this filter excludes shared-helper
            # subdirs that live under a category but don't register any @tool.
            if not (toolkit_dir / "README.md").is_file():
                continue
            toolkits.append(toolkit_dir)
    return toolkits


def _toolkit_id(toolkit_dir: Path) -> str:
    """Return 'category/toolkit' identifier for parametrize ids."""
    return f"{toolkit_dir.parent.name}/{toolkit_dir.name}"


_ALL_TOOLKITS = _discover_toolkit_dirs()
_ALL_IDS = [_toolkit_id(t) for t in _ALL_TOOLKITS]


def _validate_terms_block(block: Any, label: str, errors: list[str]) -> None:
    """Validate a code/weights terms block against the schema.

    SPDX-allowlisted licenses resolve their text from _licenses/{spdx}.txt
    (the file must exist; inline text is forbidden so canonical text doesn't
    drift across toolkits). 'Custom (...)' licenses must include inline text
    since each vendor's terms are unique.
    """
    if not isinstance(block, dict):
        errors.append(f"{label}: expected mapping, got {type(block).__name__}")
        return
    missing = _REQUIRED_TERMS_KEYS - block.keys()
    if missing:
        errors.append(f"{label}: missing required keys {sorted(missing)}")
    spdx = block.get("spdx")
    is_custom = isinstance(spdx, str) and spdx.startswith("Custom (")
    if not isinstance(spdx, str) or not spdx.strip():
        errors.append(f"{label}.spdx: expected non-empty string")
    elif spdx not in _ALLOWED_SPDX and not is_custom:
        errors.append(
            f"{label}.spdx: '{spdx}' is not in the SPDX allowlist and does not "
            f"start with 'Custom ('. Add to _ALLOWED_SPDX in this test if it's "
            f"a real SPDX identifier."
        )
    url = block.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        errors.append(f"{label}.url: expected http(s):// URL")

    # Text rules: shared file for SPDX licenses, inline for Custom licenses.
    text = block.get("text")
    if isinstance(spdx, str) and spdx in _ALLOWED_SPDX:
        if not (_LICENSES_DIR / f"{spdx}.txt").is_file():
            errors.append(
                f"{label}.spdx: '{spdx}' is allowlisted but "
                f"_licenses/{spdx}.txt is missing — add the canonical text there"
            )
        if text is not None:
            errors.append(
                f"{label}.text: must not be inlined for SPDX-allowlisted licenses "
                f"(text comes from _licenses/{spdx}.txt)"
            )
    elif is_custom:
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{label}.text: required for 'Custom (...)' licenses (each vendor's terms are unique)")


@pytest.mark.parametrize("toolkit_dir", _ALL_TOOLKITS, ids=_ALL_IDS)
def test_toolkit_license_yaml_exists(toolkit_dir: Path) -> None:
    """Every toolkit must ship a license.yaml."""
    license_path = toolkit_dir / "license.yaml"
    assert license_path.is_file(), f"missing license.yaml for {toolkit_dir.name}"


@pytest.mark.parametrize("toolkit_dir", _ALL_TOOLKITS, ids=_ALL_IDS)
def test_toolkit_license_yaml_schema(toolkit_dir: Path) -> None:
    """license.yaml must parse and conform to the schema."""
    license_path = toolkit_dir / "license.yaml"
    if not license_path.exists():
        pytest.skip("missing — covered by the existence test")

    # Parse YAML; surface any parse errors with the toolkit context.
    try:
        data = yaml.safe_load(license_path.read_text())
    except yaml.YAMLError as exc:
        pytest.fail(f"{toolkit_dir.name}/license.yaml: invalid YAML — {exc}")

    if not isinstance(data, dict):
        pytest.fail(f"{toolkit_dir.name}/license.yaml: expected top-level mapping")

    errors: list[str] = []

    # Required top-level keys.
    missing_top = _REQUIRED_TOP_KEYS - data.keys()
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    # code block (required).
    if "code" in data:
        _validate_terms_block(data["code"], "code", errors)

    # weights block (optional). Falls back to code license when absent.
    if "weights" in data:
        _validate_terms_block(data["weights"], "weights", errors)
        weights_block = data["weights"]
        if isinstance(weights_block, dict) and "access" in weights_block:
            access = weights_block["access"]
            if access not in _ALLOWED_WEIGHTS_ACCESS:
                errors.append(f"weights.access: expected one of {sorted(_ALLOWED_WEIGHTS_ACCESS)}, got {access!r}")

    # commercial_use must be a tri-state string.
    commercial_use = data.get("commercial_use")
    if commercial_use not in _ALLOWED_COMMERCIAL_USE:
        errors.append(f"commercial_use: expected one of {sorted(_ALLOWED_COMMERCIAL_USE)}, got {commercial_use!r}")

    # redistribution must be a bool.
    redistribution = data.get("redistribution")
    if not isinstance(redistribution, bool):
        errors.append(f"redistribution: expected bool, got {type(redistribution).__name__}")

    # attribution_required is optional but must be a bool when set.
    if "attribution_required" in data and not isinstance(data["attribution_required"], bool):
        errors.append(f"attribution_required: expected bool, got {type(data['attribution_required']).__name__}")

    # notes is optional but must be a string when set.
    if "notes" in data and not isinstance(data["notes"], str):
        errors.append(f"notes: expected string, got {type(data['notes']).__name__}")

    # last_updated is the ISO date the metadata was last verified against upstream.
    last_updated = data.get("last_updated")
    if not isinstance(last_updated, str) or not _ISO_DATE_RE.fullmatch(last_updated):
        errors.append(f"last_updated: expected ISO date string YYYY-MM-DD, got {last_updated!r}")

    # proto_tools_original: opt-in marker for toolkits with no upstream to point at.
    if "proto_tools_original" in data and not isinstance(data["proto_tools_original"], bool):
        errors.append(f"proto_tools_original: expected bool, got {type(data['proto_tools_original']).__name__}")

    # Reject unknown top-level keys so typos surface immediately.
    allowed_top = _REQUIRED_TOP_KEYS | {
        "weights",
        "attribution_required",
        "notes",
        "proto_tools_original",
    }
    extra = set(data.keys()) - allowed_top
    if extra:
        errors.append(f"unknown top-level keys: {sorted(extra)}")

    if errors:
        pytest.fail(f"{toolkit_dir.name}/license.yaml schema violations:\n  - " + "\n  - ".join(errors))


# ── weights.access ↔ infra cross-checks ─────────────────────────────────────


def _toolkit_access_map() -> dict[str, str]:
    """Map 'category/toolkit' -> weights.access for toolkits that declare it."""
    out: dict[str, str] = {}
    for toolkit_dir in _ALL_TOOLKITS:
        license_path = toolkit_dir / "license.yaml"
        if not license_path.is_file():
            continue
        data = yaml.safe_load(license_path.read_text())
        weights = data.get("weights") if isinstance(data, dict) else None
        access = weights.get("access") if isinstance(weights, dict) else None
        if access:
            out[_toolkit_id(toolkit_dir)] = access
    return out


def _toolkits_calling_require_hf_token() -> set[str]:
    """Return 'category/toolkit' ids whose source calls require_hf_token()."""
    found: set[str] = set()
    for py in _TOOLS_DIR.rglob("*.py"):
        parts = py.relative_to(_TOOLS_DIR).parts
        if len(parts) < 3 or parts[0] in _EXCLUDED_DIRS:
            continue
        if "require_hf_token(" in py.read_text():
            found.add(f"{parts[0]}/{parts[1]}")
    return found


def test_hf_gated_access_matches_require_hf_token() -> None:
    """weights.access: hf-gated must match exactly the toolkits calling require_hf_token().

    Binds the hand-written metadata to the runtime-enforced gate so the two
    can never silently diverge.
    """
    declared = {tid for tid, access in _toolkit_access_map().items() if access == "hf-gated"}
    enforced = _toolkits_calling_require_hf_token()
    assert declared == enforced, (
        "weights.access: hf-gated is out of sync with require_hf_token() call sites. "
        f"Declared in license.yaml only: {sorted(declared - enforced)}; "
        f"calls require_hf_token() but not declared hf-gated: {sorted(enforced - declared)}."
    )


def _gated_readme_table_models() -> set[str]:
    """First-column model names from the root README 'Gated model access' table."""
    lines = _ROOT_README.read_text().splitlines()
    header = next(
        (i for i, ln in enumerate(lines) if re.match(r"\|\s*Model\s*\|\s*Source\s*\|\s*Access\s*\|", ln)),
        None,
    )
    assert header is not None, "root README is missing the '| Model | Source | Access |' gated-access table"
    models: set[str] = set()
    for ln in lines[header + 1 :]:
        if not ln.startswith("|"):
            break
        cell = ln.split("|")[1].strip()
        if not cell or set(cell) <= set("-: "):
            continue
        models.add(cell)
    return models


def test_gated_readme_table_matches_access_field() -> None:
    """The root README gated-access table must list exactly the access-restricted toolkits.

    Expected model names are each restricted toolkit's README H1, so the
    hand-maintained table cannot drift from the structured weights.access set.
    """
    restricted = {tid for tid, access in _toolkit_access_map().items() if access in ("hf-gated", "request")}
    expected: set[str] = set()
    for tid in restricted:
        readme = _TOOLS_DIR / tid / "README.md"
        h1 = re.search(r"^# (.+)$", readme.read_text(), re.MULTILINE)
        assert h1, f"{tid}/README.md has no H1 to match against the gated-access table"
        expected.add(h1.group(1).strip())
    table = _gated_readme_table_models()
    assert table == expected, (
        "Root README 'Gated model access' table is out of sync with weights.access. "
        f"In table only: {sorted(table - expected)}; "
        f"access-restricted but missing from table: {sorted(expected - table)}."
    )


def test_get_weights_access_matches_license_yaml() -> None:
    """ToolRegistry.get_weights_access reflects license.yaml for every tool."""
    access_by_toolkit = _toolkit_access_map()
    for spec in ToolRegistry.list_all():
        toolkit_dir = spec.source_file.parent
        toolkit_id = f"{toolkit_dir.parent.name}/{toolkit_dir.name}"
        expected = access_by_toolkit.get(toolkit_id, "open")
        actual = ToolRegistry.get_weights_access(spec.key)
        assert actual == expected, (
            f"{spec.key}: get_weights_access() returned {actual!r}, but {toolkit_id}/license.yaml implies {expected!r}."
        )


# ── ToolRegistry surface ────────────────────────────────────────────────────


def test_get_license_raises_for_unknown_tool() -> None:
    """get_license raises ValueError for unknown tool keys."""
    with pytest.raises(ValueError, match="Unknown tool"):
        ToolRegistry.get_license("nonexistent-tool-key")


def test_list_licenses_keys_match_registry() -> None:
    """All keys in list_licenses exist in the tool registry."""
    licenses = ToolRegistry.list_licenses()
    for key in licenses:
        spec = ToolRegistry.get(key)
        assert spec.key == key


# ── URL reachability (network-gated) ────────────────────────────────────────

# Browser-like UA so GitHub/HuggingFace etc. don't block the request.
_URL_CHECK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (proto-tools license-consistency-test)",
    "Accept": "*/*",
}
_URL_CHECK_TIMEOUT_S = 15


def _check_url(url: str) -> str | None:
    """Return None if reachable (2xx/3xx), else a short failure reason."""
    # Try HEAD first; some servers reject it (405 / 403) so fall back to GET.
    last = "no attempt"
    for method in ("HEAD", "GET"):
        try:
            req = Request(url, method=method, headers=_URL_CHECK_HEADERS)
            with urlopen(req, timeout=_URL_CHECK_TIMEOUT_S) as resp:
                if 200 <= resp.status < 400:
                    return None
                last = f"HTTP {resp.status}"
        except HTTPError as exc:
            if method == "HEAD" and exc.code in (403, 405):
                continue
            last = f"HTTP {exc.code}"
        except URLError as exc:
            last = f"network error: {exc.reason}"
        except TimeoutError:
            last = f"timeout after {_URL_CHECK_TIMEOUT_S}s"
        break
    return last


@pytest.mark.integration
@pytest.mark.parametrize("toolkit_dir", _ALL_TOOLKITS, ids=_ALL_IDS)
def test_toolkit_license_yaml_urls_reachable(toolkit_dir: Path) -> None:
    """Every URL in license.yaml must resolve to a 2xx/3xx response."""
    license_path = toolkit_dir / "license.yaml"
    if not license_path.exists():
        pytest.skip("missing — covered by the existence test")

    try:
        data = yaml.safe_load(license_path.read_text())
    except yaml.YAMLError:
        pytest.skip("invalid YAML — covered by the schema test")
    if not isinstance(data, dict):
        pytest.skip("invalid top-level — covered by the schema test")

    # proto-tools-original toolkits point at our own repo; nothing upstream
    # to verify reachable.
    if data.get("proto_tools_original") is True:
        pytest.skip("proto_tools_original — no upstream URL to check")

    targets: list[tuple[str, str]] = []
    for label in ("code", "weights"):
        block = data.get(label)
        if isinstance(block, dict):
            url = block.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                targets.append((f"{label}.url", url))

    if not targets:
        pytest.skip("no URLs to check")

    failures: list[str] = []
    for label, url in targets:
        reason = _check_url(url)
        if reason is not None:
            failures.append(f"{label} {url}: {reason}")

    if failures:
        pytest.fail(f"{toolkit_dir.name}/license.yaml unreachable URLs:\n  - " + "\n  - ".join(failures))


# ── SPDX/URL fingerprint match (network-gated) ──────────────────────────────

# Acceptable fingerprints for each SPDX. Structure: list of patterns where
# each pattern is a tuple of phrases that must ALL appear (case-insensitive).
# The license text matches if ANY pattern matches. Chosen to disambiguate
# close cousins (BSD-2 vs BSD-3, GPL-3 vs LGPL-3 vs AGPL-3) while accepting
# both the canonical license text and short SPDX-identifier declarations on
# vendor model-card pages. Custom (...) licenses are not fingerprintable here.
_SPDX_FINGERPRINTS: dict[str, list[tuple[str, ...]]] = {
    "Apache-2.0": [("apache license", "version 2.0")],
    "MIT": [
        ("mit license",),
        # Distinctive phrase from canonical MIT text for files that omit the
        # title.
        ("permission is hereby granted, free of charge", "without restriction"),
    ],
    "BSD-2-Clause": [("bsd 2-clause",), ("bsd-2-clause",)],
    "BSD-3-Clause": [
        ("bsd 3-clause",),
        ("bsd-3-clause",),
        # The no-endorsement clause is unique to BSD-3 (vs BSD-2). Catches
        # files whose header doesn't name the variant explicitly.
        ("may not be used to endorse or promote",),
    ],
    "GPL-3.0": [("gnu general public license", "version 3")],
    "LGPL-3.0": [("gnu lesser general public license", "version 3")],
    "AGPL-3.0": [("gnu affero general public license", "version 3")],
    "MPL-2.0": [("mozilla public license", "version 2.0")],
    "CC0-1.0": [("cc0 1.0 universal",), ("cc0-1.0",)],
    "CC-BY-4.0": [
        ("attribution 4.0 international",),
        ("cc-by-4.0",),
        ("cc-by 4.0",),
        ("creativecommons.org/licenses/by/4.0",),
    ],
    "CC-BY-SA-4.0": [
        ("attribution-sharealike 4.0 international",),
        ("cc-by-sa-4.0",),
        ("cc-by-sa 4.0",),
        ("creativecommons.org/licenses/by-sa/4.0",),
    ],
    "CC-BY-NC-4.0": [
        ("attribution-noncommercial 4.0 international",),
        ("cc-by-nc-4.0",),
        ("cc-by-nc 4.0",),
        ("creativecommons.org/licenses/by-nc/4.0",),
    ],
    "CC-BY-NC-SA-4.0": [
        ("attribution-noncommercial-sharealike 4.0 international",),
        ("cc-by-nc-sa-4.0",),
        ("cc-by-nc-sa 4.0",),
        ("creativecommons.org/licenses/by-nc-sa/4.0",),
    ],
    "ISC": [("isc license",)],
    "Unlicense": [("this is free and unencumbered software released into the public domain",)],
}

_GITHUB_BLOB_RE = re.compile(r"^/([^/]+)/([^/]+)/blob/(.+)$")

# Hosts that return HTML model-card / web pages rather than raw license text.
# License files referenced via these hosts can't be fingerprinted from the URL.
_NON_RAW_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})


def _to_raw_text_url(url: str) -> str | None:
    """Return a raw-text URL for the license file, or None if not feasible."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host == "github.com":
        m = _GITHUB_BLOB_RE.match(parsed.path)
        if m:
            owner, repo, rest = m.groups()
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"
        return None
    if host == "raw.githubusercontent.com":
        return url
    if host in _NON_RAW_HOSTS:
        return None
    # Direct text/markdown asset hosted elsewhere (e.g. mafft.cbrc.jp/.../license.txt).
    if parsed.path.endswith((".txt", ".md")):
        return url
    return None


@lru_cache(maxsize=128)
def _fetch_text(url: str) -> str | None:
    """Fetch URL body as text. Returns None on failure or HTML responses."""
    try:
        req = Request(url, method="GET", headers=_URL_CHECK_HEADERS)
        with urlopen(req, timeout=_URL_CHECK_TIMEOUT_S) as resp:
            if not (200 <= resp.status < 400):
                return None
            data = resp.read(200_000)  # cap at 200KB; license files are small
    except (HTTPError, URLError, TimeoutError, OSError):
        return None
    text = data.decode("utf-8", errors="replace")
    head = text.lstrip()[:500].lower()
    if head.startswith("<!doctype") or "<html" in head or "<body" in head:
        return None
    return text


@pytest.mark.integration
@pytest.mark.parametrize("toolkit_dir", _ALL_TOOLKITS, ids=_ALL_IDS)
def test_toolkit_license_yaml_spdx_matches_url(toolkit_dir: Path) -> None:
    """Stated SPDX must fingerprint-match the license text fetched from its URL."""
    license_path = toolkit_dir / "license.yaml"
    if not license_path.exists():
        pytest.skip("missing — covered by the existence test")

    try:
        data = yaml.safe_load(license_path.read_text())
    except yaml.YAMLError:
        pytest.skip("invalid YAML — covered by the schema test")
    if not isinstance(data, dict):
        pytest.skip("invalid top-level — covered by the schema test")

    if data.get("proto_tools_original") is True:
        pytest.skip("proto_tools_original — no upstream license to fingerprint")

    # Build (label, spdx, raw_url) targets, dropping anything we can't or
    # shouldn't fingerprint (Custom, HF model card, non-raw URLs, etc).
    targets: list[tuple[str, str, str]] = []
    for label in ("code", "weights"):
        block = data.get(label)
        if not isinstance(block, dict):
            continue
        spdx = block.get("spdx")
        url = block.get("url")
        if not isinstance(spdx, str) or not isinstance(url, str):
            continue
        if spdx not in _SPDX_FINGERPRINTS:
            continue
        raw_url = _to_raw_text_url(url)
        if raw_url is None:
            continue
        targets.append((label, spdx, raw_url))

    if not targets:
        pytest.skip("no SPDX-fingerprintable URLs to check")

    failures: list[str] = []
    checked = 0
    for label, spdx, raw_url in targets:
        text = _fetch_text(raw_url)
        if text is None:
            # Soft-skip per-target: fetch failed or response wasn't raw text.
            continue
        checked += 1
        # Lowercase + collapse all runs of whitespace to single spaces, so
        # phrases match across line wraps (e.g. "endorse\n  or promote" →
        # "endorse or promote").
        normalized = re.sub(r"\s+", " ", text.lower())
        # Match if ANY pattern fully matches (all phrases in pattern present).
        patterns = _SPDX_FINGERPRINTS[spdx]
        if not any(all(phrase in normalized for phrase in pattern) for pattern in patterns):
            failures.append(f"{label}.spdx='{spdx}' but {raw_url} matches none of the fingerprint patterns")

    if checked == 0:
        pytest.skip("no fetchable license texts (HTML/blocked/unavailable)")

    if failures:
        pytest.fail(f"{toolkit_dir.name}/license.yaml SPDX vs URL mismatch:\n  - " + "\n  - ".join(failures))
