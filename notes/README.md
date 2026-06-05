# Introduction to proto-tools

This directory holds the developer reference notes for proto-tools, one per area of the codebase.

proto-tools is a library of typed bioinformatics tool wrappers. It gives every
tool, from sequence and structure prediction to ligand, genomic, and
publication/database lookups, a uniform Python API with validated input and
output schemas, generated documentation, citations and licenses, and an
isolated execution environment so heavy or conflicting dependencies never
collide.

Each note below is the canonical source for one area. Start here to find where a
topic is covered, then read the source and tests for the final word on
signatures and behavior.

## Setup and environments

- `storage.md`: `PROTO_HOME`, `PROTO_MODEL_CACHE`, where model weights live, shared weights, and per-tool overrides.
- `tool-environments.md`: how isolated tool environments are built, plus compute dependency management (GCC/nvcc), caches, binaries, overriding a tool's packaged env, and the `to_device()` protocol.
- `troubleshooting.md`: working around cluster-specific problems (failed env builds, old glibc/containers, storage quotas, GPU scheduling).

## Runtime and behavior

- `finding-tools.md`: discovering, inspecting, and calling tools through the `ToolRegistry` and `proto-tools` CLI surface, covering identifier resolution, schemas, docs extraction, JSON surfaces, gated weights, and run functions.
- `error-handling.md`: the `@tool` raise-by-default policy, opt-in capture mode (`PROTO_CAPTURE_ERRORS`), and the `MissingAssetError` carve-out.
- `logging.md`: the worker logging architecture, status updates, verbosity control, and third-party progress-bar handling.
- `seeding.md`: seed management for stochastic tools, how seeds interact with caching and dedup, and per-item RNG advancement.
- `model-taste.md`: choosing models and validators for biological design tasks, common model failure modes, and confidence labels.

## Contributing

- `testing.md`: test structure, assertions, markers, and naming conventions.
