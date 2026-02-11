#!/usr/bin/env python3
"""
Documentation generator for Bio Programming Tools.

This script auto-generates Mintlify MDX documentation from tool README.md files.

Run from repository root:
    python docs/generate_docs.py

Dependencies:
    No external dependencies required (uses only stdlib)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

DOCS_DIR = PROJECT_ROOT / "docs"
TOOLS_DIR = DOCS_DIR / "tools"

# Directories to exclude when discovering tool categories
EXCLUDED_TOOL_DIRS = {"__pycache__", "infra", "utils"}


def discover_tool_categories() -> Dict[str, str]:
    """Auto-discover tool categories from bio_programming_tools/tools directory."""
    tools_base = PROJECT_ROOT / "bio_programming_tools" / "tools"
    categories = {}

    for item in sorted(tools_base.iterdir()):
        if not item.is_dir():
            continue
        if item.name.startswith("_") or item.name in EXCLUDED_TOOL_DIRS:
            print(f"  Skipped (excluded): {item.name}")
            continue
        # Check if directory contains tool subdirectories with READMEs
        subdirs_checked = []
        for subdir in item.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("_"):
                subdirs_checked.append(subdir.name)

        has_tool_readmes = any(
            (item / subdir / "README.md").exists()
            for subdir in subdirs_checked
        )
        if has_tool_readmes:
            slug = item.name.replace("_", "-")
            categories[slug] = f"bio_programming_tools/tools/{item.name}"
        else:
            print(f"  Skipped (no READMEs): {item.name}/ (checked: {', '.join(subdirs_checked) or 'no subdirs'})")

    return categories


# =============================================================================
# Helpers
# =============================================================================

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    return text.lower().replace("_", "-").replace(" ", "-")


def escape_mdx(text: str) -> str:
    """Escape characters that have special meaning in MDX.

    In MDX, `<` followed by a letter or number is interpreted as a JSX tag.
    We need to escape `<` using HTML entity `&lt;` when it appears in text
    (not inside code blocks or JSX tags).
    """
    if not text:
        return text
    # Escape < followed by a digit (e.g., <70 becomes &lt;70)
    text = re.sub(r'<(\d)', r'&lt;\1', text)
    # Escape < followed by space then digit (e.g., < 50 becomes &lt; 50)
    text = re.sub(r'< (\d)', r'&lt; \1', text)
    return text


# Language to icon mapping for Mintlify code blocks
CODE_BLOCK_ICONS = {
    "python": "python",
    "bash": "terminal",
    "shell": "terminal",
    "sh": "terminal",
    "json": "brackets",
    "yaml": "file",
    "yml": "file",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
}


def add_code_block_icons(text: str) -> str:
    """Add Mintlify icon syntax to code blocks.

    Converts ```python to ```python python icon="python"
    Converts ```bash to ```bash bash icon="terminal"
    etc.
    """
    if not text:
        return text

    def replace_code_block(match: re.Match) -> str:
        lang = match.group(1)
        icon = CODE_BLOCK_ICONS.get(lang)
        if icon:
            return f'```{lang} {lang} icon="{icon}"'
        return match.group(0)

    # Match ```language that isn't already followed by icon syntax
    # Captures the language name and checks it's not already decorated
    pattern = r'```(' + '|'.join(re.escape(lang) for lang in CODE_BLOCK_ICONS.keys()) + r')(?!\s+\w+\s+icon=)'
    return re.sub(pattern, replace_code_block, text)


# =============================================================================
# Tool README Documentation Generator
# =============================================================================

def find_tool_readmes(category_dir: Path) -> List[Dict[str, Any]]:
    """Find all README.md files in a tool category directory."""
    tools = []

    if not category_dir.exists():
        return tools

    for item in category_dir.iterdir():
        if item.is_dir():
            readme_path = item / "README.md"
            if readme_path.exists():
                tools.append({
                    "name": item.name,
                    "readme_path": readme_path,
                })

    return tools


def extract_readme_title(content: str) -> str:
    """Extract title from README content (first H1)."""
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    return match.group(1) if match else "Untitled"


def extract_readme_description(content: str) -> str:
    """Extract description from README (overview section or first clean paragraph)."""
    # First try to get Overview section
    overview_match = re.search(r'## Overview\s*\n+(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if overview_match:
        overview_text = overview_match.group(1).strip()
        # Get first sentence or line
        first_line = overview_text.split('\n')[0].strip()
        # Clean markdown
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', first_line)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
        if clean:
            return clean[:200]

    # Fallback: first paragraph after title
    content_no_title = re.sub(r'^#\s+.+\n', '', content, count=1)
    paragraphs = content_no_title.strip().split("\n\n")
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith("#") and not p.startswith("```") and not p.startswith("**") and not p.startswith("-"):
            clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', p)
            clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
            clean = re.sub(r'`([^`]+)`', r'\1', clean)
            clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
            clean = clean.replace("\n", " ")
            return clean[:200]

    return ""


def convert_readme_to_mdx(readme_path: Path, output_path: Path) -> None:
    """Convert a README.md to Mintlify MDX format."""
    content = readme_path.read_text()

    # Extract metadata
    title = extract_readme_title(content)
    description = extract_readme_description(content)

    # Clean description for frontmatter (remove quotes, truncate)
    clean_desc = description.replace('"', "'").replace('\n', ' ')[:100]
    if len(description) > 100:
        clean_desc += "..."

    # Remove existing "Last updated" line
    content = re.sub(r'\n*-\s*Last updated:.*$', '', content, flags=re.MULTILINE)

    # Escape MDX special characters in content
    content = escape_mdx(content)

    # Add code block icons
    content = add_code_block_icons(content)

    # Add frontmatter (no timestamp to ensure deterministic output)
    mdx_content = f"""---
title: "{title}"
description: "{clean_desc}"
---

{content.rstrip()}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mdx_content)


def generate_tool_docs() -> Dict[str, List[str]]:
    """Generate MDX documentation from tool READMEs."""
    tool_categories = discover_tool_categories()
    generated_pages = {}

    for category_slug, source_dir in tool_categories.items():
        category_path = PROJECT_ROOT / source_dir
        output_dir = TOOLS_DIR / category_slug

        tools = find_tool_readmes(category_path)
        if not tools:
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        category_pages = []

        for tool in tools:
            tool_slug = slugify(tool["name"])
            output_path = output_dir / f"{tool_slug}.mdx"

            convert_readme_to_mdx(tool["readme_path"], output_path)

            page_path = f"tools/{category_slug}/{tool_slug}"
            category_pages.append(page_path)
            print(f"  Generated: {output_path.relative_to(PROJECT_ROOT)}")

        generated_pages[category_slug] = category_pages

    return generated_pages


# =============================================================================
# Navigation Updater
# =============================================================================

def slug_to_label(slug: str) -> str:
    """Convert a slug to a human-readable label."""
    acronyms = {"orf": "ORF", "rna": "RNA", "dna": "DNA", "api": "API"}
    words = slug.split("-")
    result = []
    for word in words:
        if word.lower() in acronyms:
            result.append(acronyms[word.lower()])
        else:
            result.append(word.capitalize())
    return " ".join(result)


def update_docs_json(tool_pages: Dict[str, List[str]]) -> None:
    """Update docs.json navigation with generated tool pages."""
    docs_json_path = DOCS_DIR / "docs.json"

    if not docs_json_path.exists():
        print("  Warning: docs.json not found, skipping navigation update")
        return

    docs = json.loads(docs_json_path.read_text())

    # Validate expected structure
    if "navigation" not in docs:
        print("  Error: docs.json missing 'navigation' key")
        return

    if "tabs" not in docs["navigation"]:
        print("  Error: docs.json missing 'navigation.tabs' key")
        return

    # Find the Tools tab
    tools_tab = None
    for tab in docs["navigation"]["tabs"]:
        if tab.get("tab") == "Tools":
            tools_tab = tab
            break

    if not tools_tab:
        print("  Warning: No 'Tools' tab found in docs.json")
        return

    # Build groups for each category
    new_groups = []
    for category_slug in sorted(tool_pages.keys()):
        group_label = slug_to_label(category_slug)
        pages = sorted(tool_pages[category_slug])

        new_groups.append({
            "group": group_label,
            "pages": pages
        })

    # Update the Tools tab's groups
    tools_tab["groups"] = new_groups

    # Write back to docs.json with pretty formatting
    docs_json_path.write_text(json.dumps(docs, indent="\t") + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for documentation generation."""
    print("=" * 60)
    print("Bio Programming Tools Documentation Generator")
    print("=" * 60)

    # Ensure docs directories exist
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Generating tool documentation...")
    tool_pages = generate_tool_docs()
    total_tools = sum(len(pages) for pages in tool_pages.values())
    print(f"  Total: {total_tools} tools across {len(tool_pages)} categories")

    print("\n[2/2] Updating docs.json navigation...")
    update_docs_json(tool_pages)
    print("  Updated: docs/docs.json")

    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"  - Tools: {total_tools}")
    print(f"  - Categories: {len(tool_pages)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
