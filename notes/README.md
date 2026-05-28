# Notes

Team-shared development docs for proto-tools. These files capture
tool-specific gotchas, setup guidance, and architecture decisions; knowledge
that **every developer** needs.

For personal discoveries (debugging patterns, tool quirks found during a
session), use Claude's auto-memory instead of adding to these files. Only add
to notes/ when the knowledge benefits the whole team.

## Directory Structure

- `beta-welcome.md`: Onboarding for beta users and testers
- `error-handling.md`: `@tool` raise-vs-capture policy,
  `PROTO_CAPTURE_ERRORS`, `MissingAssetError` carve-out
- `logging.md`: worker logging architecture, status updates, verbosity control,
  third-party progress bar handling
- `runtime-api.md`: `ToolRegistry`, CLI, identifier resolution, docs
  extraction, schemas, JSON surfaces, gated weights, and calling tools
- `seeding.md`: seed management for stochastic tools, cache behavior
  with/without seeds, per-item RNG advancement
- `sherlock-setup.md`: Stanford Sherlock cluster-specific setup (temporary, for beta testers)
- `storage.md`: `PROTO_HOME`, `PROTO_MODEL_CACHE`, shared weights, per-tool
  overrides, storage layout
- `testing.md`: Test structure, assertions, markers, naming conventions
- `tool-environments.md`: standalone env setup, compute deps, GCC/nvcc, caches,
  binaries, `to_device()` protocol
- `model-taste.md`: expanded model/tool selection guidance for biological
  design classes, validator choice, model failure modes, and confidence labels

## Documentation

User-facing documentation reference pages are auto-generated from Python
docstrings and field descriptions in the source code.

Developer reference docs live here in `notes/` as the canonical source for
internal development guidance.
