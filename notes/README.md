# Notes

Team-shared development docs for bio-programming-tools. These files capture platform compatibility reports, tool-specific gotchas, and architecture decisions — knowledge that **every developer** needs.

For personal discoveries (debugging patterns, tool quirks found during a session), use Claude's auto-memory instead of adding to these files. Only add to notes/ when the knowledge benefits the whole team.

## Directory Structure

- `sherlock-setup.md` — Stanford Sherlock cluster-specific setup (temporary, for beta testers)
- `tool-environments.md` — Standalone env setup, compute deps, GCC/nvcc, caches, binaries, `to_device()` protocol
- `testing.md` — Test structure, assertions, markers, naming conventions
- `usage-guide.md` — Claude Code script patterns, batch persistence, GPU tools, citations
- `environments/` — Machine-generated Markdown compatibility reports (see `environments/README.md`)

## Documentation

User-facing documentation reference pages are auto-generated from Python docstrings and field descriptions in the source code.

Developer reference docs (`tool-environments.md`, `testing.md`, `usage-guide.md`) live here in `notes/` as the canonical source for internal development guidance.
