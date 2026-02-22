# Notes

Dynamic development notes for bio-programming-tools. These files capture platform quirks, tool-specific gotchas, and other context that evolves as the repo grows.

Claude: consult these notes when working in this repo and update them when you discover new information worth preserving.

## Directory Structure

- `environments/` — Machine-generated Markdown compatibility reports (see `environments/README.md`)

## Docs Ownership

- This repository owns parsing and generation for tool docs via `docs/generate_docs.py`.
- Generated tool pages in `docs/tools/` are treated as source artifacts for the unified outer docs site.
- Tool docs are auto-generated on pushes to `main` when tool README/source files change.

## Docs Parsing Fallbacks

- JSON schema extraction remains primary for API reference generation.
- For arbitrary model field types (for example `pandas.DataFrame`, `numpy.ndarray`), the generator falls back to Pydantic field introspection and emits readable type aliases (`DataFrame`, `ndarray`).
- Manual README input/config/output sections are stripped only when corresponding API sections are generated successfully, so hand-written output docs remain as a safety net when schema extraction still fails.
