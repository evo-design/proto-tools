# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bio_tools is a modular bioinformatics tool library providing Python wrappers for 25+ biological sequence and structure analysis tools. It is the tool layer of the [bio-programming](https://github.com/evo-design/bio-programming/tree/main) project. The repo was recently restructured from the parent repo; some tests may not yet pass on main.

## Build & Development Setup

Assume `bio_tools` is already installed in the current Python environment. Do **not** create or activate a virtual environment before running tools — just use `python3` directly.

For first-time manual setup only:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Common Commands

```bash
# Run tests (slow tests skipped by default)
pytest

# Run a single test file or filter by name
pytest tests/test_blast.py
pytest -k "blast"

# Test flags
pytest --cpu              # CPU-only tests
pytest --gpu              # GPU-only tests
pytest --all              # Include slow tests
pytest --slow             # Only slow tests
pytest --no-log-console   # Suppress console logging

# Linting (CI only checks F401 unused imports and F841 unused variables)
flake8 bio_tools tests

# Formatting
black bio_tools tests
isort bio_tools tests

# Type checking
mypy bio_tools

# Pre-commit hooks (isort, trailing whitespace, doc generation)
pre-commit run --all-files
```

## Architecture

### Two Main Packages

- **`bio_tools/tools/`** — Wrappers around bioinformatics algorithms (BLAST, AlphaFold3, ESM2, etc.)
- **`bio_tools/entities/`** — Core data structures: `Structure`/`StructureEnsemble` (protein/RNA) and `Ligands` (small molecules)

### Tool Pattern (Input/Config/Output)

Every tool follows a standardized pattern using Pydantic models:
- **Input** (`BaseToolInput`) — primary data (sequences, structures)
- **Config** (`BaseConfig`) — parameters (evalue, num_threads)
- **Output** (`BaseToolOutput`) — results + metadata (tool_id, execution_time, success, errors)

Tools are registered via the `@tool()` decorator in `tool_registry.py`, which enables automatic schema generation, metadata tracking, and unified error handling.

### Environment Isolation (EnvManager)

Tools with complex or conflicting dependencies use isolated venvs managed by `EnvManager` (`bio_tools/tools/infra/env_manager.py`). Each such tool has a `standalone/` subdirectory containing `setup.sh`, optionally `requirements.txt`, and a `run.py` entry point. Venvs are created in `.venvs/{model_name}_env/`.

### Binary Tools

Tools wrapping external C/C++ binaries (BLAST+, MMseqs2, MAFFT) use a `standalone/binary_config.py` that specifies platform-specific download URLs and extraction logic. The shared installer at `infra/install_binary.py` handles downloading. Binaries live in the venv's `bin/` directory.

### Tool Categories

| Category | Examples |
|---|---|
| Gene Annotation | BLAST, MMseqs2, PyHMMER |
| ORF Prediction | Orfipy, Prodigal |
| Sequence Alignment | MAFFT, ColabFold Search |
| Sequence Scoring | Enformer, Borzoi, AlphaGenome |
| Inverse Folding | ProteinMPNN, LigandMPNN |
| Structure Prediction | AlphaFold3, Boltz2, Chai1, ESMFold, Protenix, ViennaRNA |
| Structure Design | RFDiffusion3 |
| Structure Dynamics | BioEmu |
| Masked LMs | ESM2, ESM3 |
| Causal LMs | ProGen2, Evo2 |

### Infrastructure Modules (`bio_tools/tools/infra/`)

- `env_manager.py` — Isolated venv creation and script execution
- `install_binary.py` — Platform-aware binary downloading
- `tool_cache.py` — Decorator-based result caching with TTL (`@tool_cache`)
- `tool_io.py` — Base I/O classes (`BaseToolInput`, `BaseToolOutput`)

### Key Configuration

- Python >=3.12, Pydantic >=2.0
- Black/isort line length: 88
- Flake8 only checks: F401 (unused imports), F841 (unused variables)
- Pytest markers: `uses_gpu`, `uses_cpu`, `slow`, `integration`, `skip_ci`, `asyncio`
- Tests auto-mark as `uses_cpu` unless explicitly marked `uses_gpu`

## Using bio_tools with Claude Code

### Tool Discovery

When a user asks to run a bioinformatics tool:
1. Check the category table above to find the right directory under `bio_tools/tools/`
2. Read the tool's `README.md` for biological context, parameters, and examples
3. Read the tool's Pydantic `Input`/`Config`/`Output` classes for the exact API
4. All public symbols are re-exported from `bio_tools/tools/__init__.py`

### Universal Call Pattern

Every tool follows the same pattern:

```python
from bio_tools.tools.{category}.{tool} import run_{tool}, {Tool}Input, {Tool}Config

inputs = {Tool}Input(...)   # Primary data: sequences, structures, files
config = {Tool}Config(...)  # Parameters: evalue, num_threads, seeds
result = run_{tool}(inputs, config)

# result.success, result.execution_time, result.errors, plus tool-specific fields
```

### Output Rule

Infer from context whether to **write a script** or **execute directly**:

**Write a script** to `./analyses/{descriptive_name}_{YYYY-MM-DD}.py` when:
- The user says "write", "create", "set up", "notebook", or similar authoring language
- The task is a multi-step pipeline or expensive GPU job
- The user will likely iterate on parameters or review before running
- When unclear — default to writing a script (safer, reproducible)

**Execute directly** (run inline via Bash, include code in the response) when:
- The user says "what is", "how many", "show me", "quick", "find", or similar query language
- The task is a simple one-off lookup or quick check
- The answer is more important than the script

In either case, always show the equivalent Python code so the user can reproduce the result. See `analyses/examples/` for reference scripts.

### Script Structure

Generated scripts should follow this structure:

```python
"""
Brief description of what this analysis does.
Generated: {date}
"""
from bio_tools.tools.{category}.{tool} import ...

# --- Configuration (review these) ---
# All parameters in one place with comments explaining choices

# --- Run ---
# Tool execution

# --- Results ---
# Parse and display output
```

For multi-step pipelines, use one script with `# === Step N: Description ===` section headers.

### GPU Tools

Some tools require GPU access. Tools that need GPU include: AlphaFold3, Boltz2, Chai1, ESMFold, Protenix, ESM2, ESM3, Evo2, ProGen2, Enformer, Borzoi, AlphaGenome, ProteinMPNN, LigandMPNN, RFDiffusion3, BioEmu. When writing scripts for these tools, note the GPU requirement in a comment at the top of the script.
