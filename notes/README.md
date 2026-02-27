# Notes

Team-shared development docs for bio-programming-tools. These files capture platform compatibility reports, tool-specific gotchas, and architecture decisions — knowledge that **every developer** needs.

For personal discoveries (debugging patterns, tool quirks found during a session), use Claude's auto-memory instead of adding to these files. Only add to notes/ when the knowledge benefits the whole team.

## Directory Structure

- `environments/` — Machine-generated Markdown compatibility reports (see `environments/README.md`)
- `huggingface_token.md` — HuggingFace token setup for gated models (ESM3, AlphaGenome)

## Docs Ownership

- This repository owns parsing and generation for tool docs via `docs/generate_docs.py`.
- Generated tool pages in `docs/tools/` are treated as source artifacts for the unified outer docs site.
- Tool docs are auto-generated on pushes to `main` when tool README/source files change.

## Docs Parsing Fallbacks

- JSON schema extraction remains primary for API reference generation.
- For arbitrary model field types (for example `pandas.DataFrame`, `numpy.ndarray`), the generator falls back to Pydantic field introspection and emits readable type aliases (`DataFrame`, `ndarray`).
- Manual README input/config/output sections are stripped only when corresponding API sections are generated successfully, so hand-written output docs remain as a safety net when schema extraction still fails.
