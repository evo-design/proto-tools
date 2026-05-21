# Notes

Team-shared development docs for proto-tools. These files capture platform compatibility reports, tool-specific gotchas, and architecture decisions; knowledge that **every developer** needs.

For personal discoveries (debugging patterns, tool quirks found during a session), use Claude's auto-memory instead of adding to these files. Only add to notes/ when the knowledge benefits the whole team.

## Directory Structure

- `beta-welcome.md`: Onboarding for beta users and testers
- `error-handling.md`: `@tool` raise-vs-capture policy, `PROTO_CAPTURE_ERRORS`, `MissingAssetError` carve-out
- `homology-search-design.md`: Design doc for the mmseqs2-based homology-search family (MSA generation, paired MSAs, dataset registry)
- `logging.md`: Worker logging architecture, status updates, verbosity control, third-party progress bar handling
- `seeding.md`: Seed management for stochastic tools, cache behavior with/without seeds, per-item RNG advancement
- `sherlock-setup.md`: Stanford Sherlock cluster-specific setup (temporary, for beta testers)
- `storage.md`: `PROTO_HOME`, `PROTO_MODEL_CACHE`, shared weights, per-tool overrides, storage layout
- `testing.md`: Test structure, assertions, markers, naming conventions
- `tool-environments.md`: Standalone env setup, compute deps, GCC/nvcc, caches, binaries, `to_device()` protocol
- `environments/`: Machine-generated Markdown compatibility reports (see `environments/README.md`)

## Documentation

User-facing documentation reference pages are auto-generated from Python docstrings and field descriptions in the source code.

Developer reference docs (`tool-environments.md`, `testing.md`) live here in `notes/` as the canonical source for internal development guidance.
