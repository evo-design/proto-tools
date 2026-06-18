"""tests/style_consistency_tests/test_readme_consistency.py."""

import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

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
    "Toolkit Notes",
]

# Overview section length cap (characters of body text, excluding the heading).
_OVERVIEW_MAX_CHARS = 600

# Permissive OSI / public-domain SPDX identifiers used to gate the fully-open License callout.
_PERMISSIVE_SPDX = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "MPL-2.0",
        "ISC",
        "Unlicense",
        "CC0-1.0",
    }
)

# Guide paths every ## Toolkit Notes must link to (substring match: any host/badge syntax counts).
_TOOLKIT_NOTES_REQUIRED_GUIDE_PATHS = [
    "/tools/guides/tool-persistence",
    "/tools/guides/device-management",
    "/tools/guides/parallel-execution",
    "/tools/guides/cloud-inference",
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
    """README must have a right-aligned docs badge linking to the correct page."""
    rel = readme.relative_to(_TOOLS_DIR)
    category, toolkit, _ = rel.parts
    expected_path = f"/tools/{_slugify(category)}/{_slugify(toolkit)}"

    text = readme.read_text()
    expected_label = "View_Docs"
    assert f"img.shields.io/badge/{expected_label}" in text, (
        f"{_tool_id(readme)}/README.md is missing the '{expected_label}' docs badge"
    )
    assert 'align="right"' in text, f'{_tool_id(readme)}/README.md badge must use align="right"'
    assert expected_path in text, f"{_tool_id(readme)}/README.md badge should link to {expected_path}"


# The "Use in Proto Tools" badge advertises that a toolkit can be run on the
# hosted Proto Tools platform. Hosting requires redistributing the tool, so the
# badge is present only when `redistribution: true` in the toolkit's
# license.yaml. Hostable toolkits must use this exact markup verbatim so the
# badge is byte-for-byte identical catalog-wide (label, message, color,
# flat-square style, matching labelColor, lightning logo, alt text). Tools that
# cannot be hosted omit it entirely.
_PROTO_TOOLS_BADGE_LABEL = "Use_on_Proto"
_PROTO_TOOLS_BADGE_IMG = '<img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">'


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_proto_tools_badge_matches_license(readme: Path) -> None:
    """The 'Use in Proto Tools' badge must be present iff the license permits hosting.

    Hosting a tool on the Proto Tools platform requires redistributing it, so
    the badge (purple ``coming soon``, with the lightning logo) appears only
    when ``license.yaml`` declares ``redistribution: true``. Toolkits that
    cannot be hosted must omit the badge entirely.
    """
    text = readme.read_text()

    license_path = readme.parent / "license.yaml"
    assert license_path.exists(), f"{_tool_id(readme)} has no license.yaml to gate the Proto Tools badge."
    license_data = yaml.safe_load(license_path.read_text())
    assert "redistribution" in license_data, (
        f"{_tool_id(readme)}/license.yaml is missing the 'redistribution' field, which gates the "
        f"'{_PROTO_TOOLS_BADGE_LABEL}' badge."
    )

    has_badge = _PROTO_TOOLS_BADGE_LABEL in text
    if license_data["redistribution"]:
        assert _PROTO_TOOLS_BADGE_IMG in text, (
            f"{_tool_id(readme)}/README.md has redistribution=true, so it must carry the exact canonical "
            f"'{_PROTO_TOOLS_BADGE_LABEL}' badge markup (see _PROTO_TOOLS_BADGE_IMG in this module) — "
            f"identical label, color, flat-square style, labelColor, lightning logo, and alt text."
        )
    else:
        assert not has_badge, (
            f"{_tool_id(readme)}/README.md has redistribution=false, so it must NOT carry the "
            f"'{_PROTO_TOOLS_BADGE_LABEL}' badge — the tool cannot be hosted on Proto Tools."
        )


# Canonical markup for the badges that appear in every README regardless of
# license. Every tool README must contain each verbatim so they are
# byte-for-byte identical catalog-wide. The View Docs badge's <a href> is
# per-tool (validated by test_has_doc_badge), so only its constant <img> is
# pinned here.
_VIEW_DOCS_BADGE_IMG = '<img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs">'
_EXAMPLE_NOTEBOOK_BADGE = '<a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>'
_TOOLKIT_NOTES_GUIDE_BADGES = '<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>'


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_badges_use_compact_style(readme: Path) -> None:
    """Tool READMEs must use the compact ``flat-square`` shields.io badge style.

    The large ``style=for-the-badge`` style is not allowed for any badge.
    """
    text = readme.read_text()

    assert "style=for-the-badge" not in text, (
        f"{_tool_id(readme)}/README.md uses the large 'style=for-the-badge' shields.io style. "
        f"All badges must use the compact 'style=flat-square' style."
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_badges_are_canonical(readme: Path) -> None:
    """Tool READMEs must use the exact canonical markup for every fixed badge.

    Pins the View Docs ``<img>``, the Example Notebook anchor, and the four
    Toolkit Notes guide badges byte-for-byte so they are identical catalog-wide.
    The Proto badge is license-gated and verified in
    ``test_proto_tools_badge_matches_license``.
    """
    text = readme.read_text()

    canonical = {
        "View Docs badge <img>": _VIEW_DOCS_BADGE_IMG,
        "Example Notebook badge": _EXAMPLE_NOTEBOOK_BADGE,
        "Toolkit Notes guide badges": _TOOLKIT_NOTES_GUIDE_BADGES,
    }
    missing = [name for name, markup in canonical.items() if markup not in text]
    assert not missing, (
        f"{_tool_id(readme)}/README.md is missing the exact canonical markup for: {missing}. "
        f"These badges must be byte-for-byte identical across all QC-done READMEs "
        f"(see the canonical badge constants in this test module)."
    )


def _expected_license_callouts(lic: dict, name: str) -> list[str]:
    """Build the acceptable License callout strings from a parsed license.yaml.

    The "a/an" article in "has a/an {spdx} license" is not enforced (authors
    write whichever is grammatically correct), so the plain-SPDX branch yields
    both variants. ``Custom (...)`` SPDX values keep the article-free
    "is licensed under {spdx}" form (an article + trailing "license" reads
    wrong for vendor strings). A code/weights split is used only when the
    weights SPDX genuinely differs. A ``weights.access`` gating sentence is
    appended when present.

    Args:
        lic (dict): Parsed license.yaml mapping.
        name (str): Tool display name (the README H1 text).

    Returns:
        list[str]: Acceptable ``> [!NOTE]`` callouts; any one may appear.
    """
    code_spdx = lic["code"]["spdx"]
    code_url = lic["code"]["url"]
    weights = lic.get("weights")
    commercial_restricted = lic.get("commercial_use") != "yes"
    attribution = bool(lic.get("attribution_required"))
    is_custom = isinstance(code_spdx, str) and code_spdx.startswith("Custom (")
    access = weights.get("access") if isinstance(weights, dict) else None
    weights_spdx = weights["spdx"] if isinstance(weights, dict) else None

    # Pipeline toolkits (bundled_dependencies present): layer each license explicitly instead of collapsing restrictions onto the code lead.
    # Two flavors: model-weight pipelines (weights block present) keep the existing
    # "pipeline + model weights + restricted-use" intro; data-orchestrator toolkits
    # (no weights) federate over sibling toolkits and surface each sibling's data terms.
    deps = lic.get("bundled_dependencies")
    if deps:
        has_weights = bool(weights)
        split_weights = has_weights and weights_spdx != code_spdx
        if has_weights:
            intro = [
                f"{name}'s own code is licensed under {code_spdx}, but it runs as a "
                "pipeline that depends on bundled components and model weights under "
                "separate license terms, including non-commercial or restricted-use terms."
            ]
            agg_subject = "pipeline"
        else:
            intro = [
                f"{name}'s own code is licensed under {code_spdx}, and it federates "
                "over bundled data sources and components, each under its own license terms."
            ]
            agg_subject = "federation"
        if split_weights:
            intro.append(f"The bundled model weights are licensed under {weights_spdx}.")
        agg = []
        if commercial_restricted:
            agg.append("has restrictions around commercial use")
        if attribution:
            agg.append("may require explicit attribution when utilized")
        if agg:
            intro.append(f"As a whole the {agg_subject} " + " and ".join(agg) + ".")

        bullets = []
        for dep in deps:
            if "tool" in dep:
                category, toolkit = dep["tool"].split("/", 1)
                dep_url = f"https://bio-pro.mintlify.app/tools/{_slugify(category)}/{_slugify(toolkit)}"
                dep_lic = yaml.safe_load((_TOOLS_DIR / dep["tool"] / "license.yaml").read_text())
                # Data-orchestrator deps surface the sibling's data SPDX (governs use of retrieved records);
                # pipeline deps surface code SPDX. Falls back to code.spdx when no explicit data block exists.
                dep_data = dep_lic.get("data")
                if not has_weights:
                    ds = dep_data["spdx"] if isinstance(dep_data, dict) else dep_lic["code"]["spdx"]
                    if isinstance(ds, str) and ds.startswith("Custom (") and ds.endswith(")"):
                        ds = ds[len("Custom (") : -1]
                    dep_spdx = ds
                else:
                    dep_spdx = dep_lic["code"]["spdx"]
            else:
                dep_url = dep["url"]
                dep_spdx = dep.get("spdx")
            suffix = f": {dep_spdx}" if dep_spdx else ""
            bullets.append(f"> - [{dep['name']}]({dep_url}){suffix}")

        if split_weights:
            review = (
                f"Review the [code license]({code_url}) and the "
                f"[model weights license]({weights['url']}) before any commercial use or redistribution."
            )
        elif has_weights:
            review = f"Review the [code license]({code_url}) before any commercial use or redistribution."
        else:
            review = "Review each source's terms before commercial use or redistribution."

        return [
            f"> [!NOTE]\n> **License:** {' '.join(intro)}\n>\n"
            "> Bundled dependencies, each under its own license:\n>\n" + "\n".join(bullets) + f"\n>\n> {review}"
        ]

    # Database / API-wrapper toolkits: the shipped code is a thin client; what governs
    # use of the retrieved results is the external data resource's own terms.
    data = lic.get("data")
    if data:
        d_name = data["name"]
        d_spdx = data["spdx"]
        d_url = data["url"]
        d_spdx_l = str(d_spdx).lower()
        # Strip a "Custom (...)" wrapper for display; non-SPDX data terms read better unwrapped.
        d_disp = d_spdx
        if isinstance(d_spdx, str) and d_spdx.startswith("Custom (") and d_spdx.endswith(")"):
            d_disp = d_spdx[len("Custom (") : -1]
        if d_spdx_l.startswith("cc0"):
            distributed = f"distributed under {d_spdx} (public domain; no attribution required)"
            attribution_sentence = ""
        elif "public domain" in d_spdx_l:
            distributed = f"in the public domain ({d_disp})"
            attribution_sentence = ""
        elif bool(data.get("attribution_required")) or d_spdx_l.startswith("cc-by"):
            distributed = f"distributed under {d_disp}"
            attribution_sentence = f" Attribution to {d_name} is required when the data is redistributed."
        else:
            distributed = f"distributed under {d_disp}"
            attribution_sentence = ""
        body = (
            f"{name} retrieves data from {d_name}, {distributed}.{attribution_sentence} "
            f"The client wrapper code is {code_spdx}-licensed."
        )
        return [f"> [!NOTE]\n> **License:** {body} Please refer to [the data terms]({d_url}) for full terms."]

    # Fully open: permissive code (and weights, if any), commercial use allowed, weights not gated.
    fully_open = (
        not is_custom
        and code_spdx in _PERMISSIVE_SPDX
        and (weights is None or weights_spdx == code_spdx)
        and lic.get("commercial_use") == "yes"
        and not access
    )

    if fully_open:
        leads = [
            f"{name} is open source and free for academic and commercial use under a {code_spdx} license",
            f"{name} is open source and free for academic and commercial use under an {code_spdx} license",
        ]
        link = f"[the license]({code_url})"
    elif weights and weights["spdx"] != code_spdx:
        leads = [f"{name} uses {code_spdx} for code and {weights['spdx']} for model weights"]
        link = f"the [code license]({code_url}) and [model weights license]({weights['url']})"
    elif is_custom:
        leads = [f"{name} is licensed under {code_spdx}"]
        link = f"[the license]({code_url})"
    else:
        leads = [f"{name} has a {code_spdx} license", f"{name} has an {code_spdx} license"]
        link = f"[the license]({code_url})"

    clauses = []
    if commercial_restricted:
        clauses.append("has restrictions around commercial use")
    if attribution:
        clauses.append("may require explicit attribution when utilized")

    if access == "hf-gated":
        gating = (
            " Model weights are gated and require accepting the provider's terms "
            "and authenticating with a HuggingFace token."
        )
    elif access == "request":
        gating = " Model weights are not publicly distributed and must be requested from the provider."
    else:
        gating = ""

    callouts = []
    for lead in leads:
        body = f"{lead} and {' and '.join(clauses)}." if clauses else f"{lead}."
        callouts.append(f"> [!NOTE]\n> **License:** {body}{gating} Please refer to {link} for full terms.")
    return callouts


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_license_section_matches_yaml(readme: Path) -> None:
    """Tool READMEs must carry the canonical License callout above ``## Overview``.

    The callout is generated from the sibling ``license.yaml`` (code SPDX,
    optional weights SPDX, commercial-use / attribution flags, weights.access
    gating) and must sit between the badge line and the Overview section. The
    "a/an" article is not enforced; any other deviation is.
    """
    text = readme.read_text()

    license_path = readme.parent / "license.yaml"
    assert license_path.exists(), f"{_tool_id(readme)} has no license.yaml to build the License callout."
    lic = yaml.safe_load(license_path.read_text())

    h1 = re.search(r"^# (.+)$", text, re.MULTILINE)
    assert h1, f"{_tool_id(readme)}/README.md has no H1 to name the tool in the License callout."
    name = h1.group(1).strip()

    variants = _expected_license_callouts(lic, name)
    present = [v for v in variants if v in text]
    assert present, (
        f"{_tool_id(readme)}/README.md is missing the canonical License callout derived from license.yaml. "
        f"Expected one of:\n" + "\n--\n".join(variants)
    )

    overview = text.find("## Overview")
    assert overview != -1 and min(text.find(v) for v in present) < overview, (
        f"{_tool_id(readme)}/README.md License callout must appear above the '## Overview' section."
    )


def test_expected_license_callouts_logic() -> None:
    """Article variants, SPDX collapsing, Custom guard, and weights.access gating."""
    base = {
        "code": {"spdx": "MIT", "url": "https://example.com/code"},
        "commercial_use": "yes",
        "attribution_required": False,
    }

    # MIT code, commercial use allowed, no gated weights -> fully-open phrasing.
    no_weights = _expected_license_callouts(base, "Tool")
    assert sorted(no_weights) == sorted(
        [
            "> [!NOTE]\n> **License:** Tool is open source and free for academic"
            " and commercial use under a MIT license."
            " Please refer to [the license](https://example.com/code) for full terms.",
            "> [!NOTE]\n> **License:** Tool is open source and free for academic"
            " and commercial use under an MIT license."
            " Please refer to [the license](https://example.com/code) for full terms.",
        ]
    )

    # Weights under the same permissive SPDX stay fully-open (no code/weights split).
    same_spdx = _expected_license_callouts(
        {**base, "weights": {"spdx": "MIT", "url": "https://example.com/weights"}}, "Tool"
    )
    assert same_spdx == no_weights

    different_spdx = _expected_license_callouts(
        {**base, "weights": {"spdx": "CC-BY-4.0", "url": "https://example.com/w"}}, "Tool"
    )
    assert different_spdx == [
        "> [!NOTE]\n> **License:** Tool uses MIT for code and CC-BY-4.0 for model weights."
        " Please refer to the [code license](https://example.com/code)"
        " and [model weights license](https://example.com/w) for full terms."
    ]

    custom = _expected_license_callouts(
        {"code": {"spdx": "Custom (Foo License)", "url": "https://example.com/c"}, "commercial_use": "yes"},
        "Tool",
    )
    assert custom == [
        "> [!NOTE]\n> **License:** Tool is licensed under Custom (Foo License)."
        " Please refer to [the license](https://example.com/c) for full terms."
    ]

    hf_gated = _expected_license_callouts(
        {
            "code": {"spdx": "Apache-2.0", "url": "https://example.com/c"},
            "weights": {"spdx": "Custom (X)", "url": "https://example.com/w", "access": "hf-gated"},
            "commercial_use": "no",
            "attribution_required": True,
        },
        "Tool",
    )
    assert len(hf_gated) == 1
    assert (
        "Model weights are gated and require accepting the provider's terms "
        "and authenticating with a HuggingFace token." in hf_gated[0]
    )

    # Bundled-pipeline toolkit: own code, weights, and deps licensed separately; restriction attributed to the pipeline as a whole.
    pipeline = _expected_license_callouts(
        {
            "code": {"spdx": "Apache-2.0", "url": "https://example.com/c"},
            "weights": {"spdx": "CC-BY-4.0", "url": "https://example.com/w"},
            "commercial_use": "restricted",
            "attribution_required": True,
            "bundled_dependencies": [
                {"name": "PyRosetta", "tool": "structure_scoring/pyrosetta"},
                {"name": "IgLM", "spdx": "Custom (IgLM License)", "url": "https://github.com/Graylab/IgLM"},
            ],
        },
        "Tool",
    )
    assert pipeline == [
        "> [!NOTE]\n> **License:** Tool's own code is licensed under Apache-2.0,"
        " but it runs as a pipeline that depends on bundled components and model"
        " weights under separate license terms, including non-commercial or"
        " restricted-use terms. The bundled model weights are licensed under"
        " CC-BY-4.0. As a whole the pipeline has restrictions around commercial"
        " use and may require explicit attribution when utilized.\n>\n"
        "> Bundled dependencies, each under its own license:\n>\n"
        "> - [PyRosetta](https://bio-pro.mintlify.app/tools/structure-scoring/pyrosetta):"
        " Custom (PyRosetta Software License)\n"
        "> - [IgLM](https://github.com/Graylab/IgLM): Custom (IgLM License)\n>\n"
        "> Review the [code license](https://example.com/c) and the [model weights"
        " license](https://example.com/w) before any commercial use or redistribution."
    ]


def test_expected_license_callouts_data_block() -> None:
    """An optional ``data:`` block yields a data-resource callout (CC0 vs CC-BY)."""
    cc0 = {
        "code": {"spdx": "Apache-2.0", "url": "https://github.com/evo-design/proto-tools"},
        "data": {
            "name": "the RCSB Protein Data Bank",
            "spdx": "CC0-1.0",
            "url": "https://www.rcsb.org/pages/policies",
            "attribution_required": False,
        },
        "commercial_use": "yes",
    }
    assert _expected_license_callouts(cc0, "PDB") == [
        "> [!NOTE]\n> **License:** PDB retrieves data from the RCSB Protein Data Bank, "
        "distributed under CC0-1.0 (public domain; no attribution required). "
        "The client wrapper code is Apache-2.0-licensed. "
        "Please refer to [the data terms](https://www.rcsb.org/pages/policies) for full terms."
    ]

    ccby = {
        "code": {"spdx": "Apache-2.0", "url": "https://github.com/evo-design/proto-tools"},
        "data": {
            "name": "the AlphaFold Protein Structure Database",
            "spdx": "CC-BY-4.0",
            "url": "https://alphafold.ebi.ac.uk/faq",
            "attribution_required": True,
        },
        "commercial_use": "yes",
    }
    assert _expected_license_callouts(ccby, "AlphaFold DB") == [
        "> [!NOTE]\n> **License:** AlphaFold DB retrieves data from the AlphaFold Protein "
        "Structure Database, distributed under CC-BY-4.0. Attribution to the AlphaFold Protein "
        "Structure Database is required when the data is redistributed. "
        "The client wrapper code is Apache-2.0-licensed. "
        "Please refer to [the data terms](https://alphafold.ebi.ac.uk/faq) for full terms."
    ]


def test_expected_license_callouts_data_custom() -> None:
    """data: with Custom/public-domain terms strips the wrapper and reads cleanly."""
    pd = {
        "code": {"spdx": "Apache-2.0", "url": "https://github.com/evo-design/proto-tools"},
        "data": {
            "name": "NCBI's Entrez databases",
            "spdx": "Custom (U.S. Government public domain)",
            "url": "https://www.ncbi.nlm.nih.gov/home/about/policies/",
            "attribution_required": False,
        },
        "commercial_use": "yes",
    }
    assert _expected_license_callouts(pd, "NCBI Entrez") == [
        "> [!NOTE]\n> **License:** NCBI Entrez retrieves data from NCBI's Entrez databases, "
        "in the public domain (U.S. Government public domain). "
        "The client wrapper code is Apache-2.0-licensed. "
        "Please refer to [the data terms](https://www.ncbi.nlm.nih.gov/home/about/policies/) for full terms."
    ]

    custom_attr = {
        "code": {"spdx": "Apache-2.0", "url": "https://github.com/evo-design/proto-tools"},
        "data": {
            "name": "the Ensembl project",
            "spdx": "Custom (the EMBL-EBI Terms of Use)",
            "url": "https://www.ebi.ac.uk/about/terms-of-use/",
            "attribution_required": True,
        },
        "commercial_use": "yes",
    }
    assert _expected_license_callouts(custom_attr, "Ensembl") == [
        "> [!NOTE]\n> **License:** Ensembl retrieves data from the Ensembl project, "
        "distributed under the EMBL-EBI Terms of Use. Attribution to the Ensembl project is "
        "required when the data is redistributed. The client wrapper code is Apache-2.0-licensed. "
        "Please refer to [the data terms](https://www.ebi.ac.uk/about/terms-of-use/) for full terms."
    ]


def test_expected_license_callouts_bundled_data_orchestrator() -> None:
    """bundled_dependencies with no weights yields a federated data-orchestrator callout.

    Sibling toolkits with a ``data:`` block contribute their data SPDX (not their
    code SPDX), so the surfaced terms reflect what governs use of the retrieved
    records. The ``Custom (...)`` wrapper is stripped for display, and an
    attribution-required source gets an inline clause.
    """
    orchestrator = {
        "code": {"spdx": "Apache-2.0", "url": "https://github.com/evo-design/proto-tools"},
        "commercial_use": "yes",
        "attribution_required": False,
        "bundled_dependencies": [
            {"name": "NCBI Entrez", "tool": "database_retrieval/ncbi"},
            {"name": "UniProt", "tool": "database_retrieval/uniprot"},
            {"name": "RCSB PDB", "tool": "database_retrieval/pdb"},
        ],
    }
    assert _expected_license_callouts(orchestrator, "Unified Sequence Fetch") == [
        "> [!NOTE]\n> **License:** Unified Sequence Fetch's own code is licensed under Apache-2.0,"
        " and it federates over bundled data sources and components, each under its own license terms.\n>\n"
        "> Bundled dependencies, each under its own license:\n>\n"
        "> - [NCBI Entrez](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi):"
        " U.S. Government public domain\n"
        "> - [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot): CC-BY-4.0\n"
        "> - [RCSB PDB](https://bio-pro.mintlify.app/tools/database-retrieval/pdb): CC0-1.0\n>\n"
        "> Review each source's terms before commercial use or redistribution."
    ]


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_gated_weights_noted_in_toolkit_notes(readme: Path) -> None:
    """Toolkits with gated weights must flag it in ``## Toolkit Notes``.

    Agents pull the Toolkit Notes section via get_tool_docs, so an access
    restriction declared by ``weights.access`` must be visible there too, not
    only in the License callout.
    """
    text = readme.read_text()

    license_path = readme.parent / "license.yaml"
    if not license_path.exists():
        return
    lic = yaml.safe_load(license_path.read_text())
    weights = lic.get("weights")
    access = weights.get("access") if isinstance(weights, dict) else None
    if not access:
        return

    match = re.search(r"^## Toolkit Notes\s*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, (
        f"{_tool_id(readme)}/README.md declares weights.access='{access}' but has no "
        f"'## Toolkit Notes' section to flag it for agents."
    )
    assert "gated" in match.group(1).lower(), (
        f"{_tool_id(readme)}/README.md declares weights.access='{access}' but its '## Toolkit Notes' "
        f"section does not mention the gating; agents pulling tool docs won't see it."
    )


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


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_overview_within_char_limit(readme: Path) -> None:
    """Overview section visible-text body must be at most ``_OVERVIEW_MAX_CHARS`` characters.

    Markdown link URLs (the ``(url)`` portion of ``[text](url)``) are
    stripped before measuring; only the visible text the reader sees
    counts toward the cap.
    """
    text = readme.read_text()

    match = re.search(r"^## Overview\s*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, f"{_tool_id(readme)}/README.md has no '## Overview' section to measure"

    body = match.group(1).strip()
    # Collapse [text](url) -> text so URL bytes do not count against the cap.
    visible = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body)
    assert len(visible) <= _OVERVIEW_MAX_CHARS, (
        f"{_tool_id(readme)}/README.md Overview is {len(visible)} chars of visible text "
        f"(limit {_OVERVIEW_MAX_CHARS}). Trim it."
    )


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_toolkit_notes_links_required_guides(readme: Path) -> None:
    """``## Toolkit Notes`` must link to all four proto-docs guides.

    Tool Persistence, Device Management, Parallel Execution, and Cloud Inference
    apply to every tool at runtime, so the toolkit-level notes section must
    surface them as visible badges/links.

    A missing section or missing guides are hard failures.
    """
    text = readme.read_text()

    match = re.search(r"^## Toolkit Notes\s*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, (
        f"{_tool_id(readme)}/README.md has no '## Toolkit Notes' section. "
        f"Add the section with badge/link references to all four proto-docs guides: "
        f"Tool Persistence, Device Management, Parallel Execution, Cloud Inference."
    )

    section = match.group(1)
    missing = [path for path in _TOOLKIT_NOTES_REQUIRED_GUIDE_PATHS if path not in section]
    assert not missing, (
        f"{_tool_id(readme)}/README.md '## Toolkit Notes' is missing required guide links: {missing}. "
        f"Add badge/link references to all four proto-docs guides: Tool Persistence, "
        f"Device Management, Parallel Execution, and Cloud Inference."
    )


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


# Per-tool subsection contract for polished READMEs. Each H3 tool entry must
# follow the three-section pattern: an overview paragraph directly under the
# H3, then `#### Applications`, then `#### Usage Tips`.
_REQUIRED_TOOL_H4S = ("Applications", "Usage Tips")


@pytest.mark.parametrize("readme", _ALL_READMES, ids=_ALL_IDS)
def test_tools_section_lists_all_tools(readme: Path) -> None:
    """The ``## Tools`` section must contain one H3 per registered tool, each with a body.

    Each H3 must reference the tool's registry key as inline code (e.g. ``### BLAST Search
    (`blast-search`)``). H3 entries that don't correspond to a registered tool are rejected.

    Polished READMEs (those that have started using ``#### Applications`` or
    ``#### Usage Tips`` H4s under any tool) must additionally follow the three-section
    pattern under every H3: an overview paragraph directly under the header, then
    ``#### Applications`` and ``#### Usage Tips`` H4 subsections in that order, each
    with a non-empty body. READMEs that have not started the migration are exempted
    from the three-section check; see the TODO note below.
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

    pattern_issues: list[str] = []
    for key, body in h3_keys.items():
        h4_matches = list(re.finditer(r"^#### (.+)$", body, re.MULTILINE))
        h4_names = [m.group(1).strip() for m in h4_matches]

        if h4_names != list(_REQUIRED_TOOL_H4S):
            pattern_issues.append(f"  `{key}`: expected H4s {list(_REQUIRED_TOOL_H4S)}, got {h4_names}")
            continue

        first_h4_offset = h4_matches[0].start()
        overview_text = body[:first_h4_offset].strip()
        if not overview_text:
            pattern_issues.append(f"  `{key}`: missing overview paragraph before `#### {h4_names[0]}`")

        for i, m in enumerate(h4_matches):
            section_start = m.end()
            section_end = h4_matches[i + 1].start() if i + 1 < len(h4_matches) else len(body)
            section_body = body[section_start:section_end].strip()
            if not section_body:
                pattern_issues.append(f"  `{key}`: `#### {h4_names[i]}` has no body")

    assert not pattern_issues, (
        f"{_tool_id(readme)}/README.md tool entries don't follow the three-section pattern "
        f"(overview paragraph, `#### Applications`, `#### Usage Tips`):\n" + "\n".join(pattern_issues)
    )


# ── Links ─────────────────────────────────────────────────────────────────


# Domains that genuinely do not support HTTPS.
_HTTP_ONLY_DOMAINS = frozenset({"hmmer.org", "eddylab.org", "bioinfo.isyslab.info"})


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
    Inline code spans (`...`) are stripped first so URLs documented as API
    endpoints inside code spans aren't treated as clickable links.
    Returns deduplicated (1-based line number, url) tuples.
    """
    urls: list[tuple[int, str]] = []
    seen: set[str] = set()
    for lineno, raw_line in _lines_outside_code_blocks(text):
        line = _strip_inline_code(raw_line)
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
    # - www.ensembl.org: genome browser homepage, intermittently slow in CI
    # - www.mirbase.org: canonical miRBase database, TLS handshake intermittently times out in CI
    # - smart.embl.de: canonical SMART domain database, intermittently times out in CI
    _SKIP_DOMAINS = {
        "bio-pro.mintlify.app",
        "doi.org",
        "proteininformationresource.org",
        "www.ensembl.org",
        "www.mirbase.org",
        "smart.embl.de",
    }
    # Self-references to this project's own repo; no need to validate our own links.
    _SKIP_SUBSTRINGS = ("proto-tools",)

    broken = []
    for lineno, url in urls:
        if any(domain in url for domain in _SKIP_DOMAINS):
            continue
        if any(substring in url for substring in _SKIP_SUBSTRINGS):
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
