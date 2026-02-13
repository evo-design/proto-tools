# Development Guide

This guide covers the development workflow for `bio_programming_tools`, including pre-commit hooks and CI checks.

## Quick Reference

```bash
# Important commands to know
python docs/generate_docs.py          # Auto-generate docs (done automatically by pre-commit hooks)
flake8 bio_programming_tools tests    # Run by Lint Check CI to check code style
pytest                                # Run all tests
pytest --skip-ci                      # Run tests excluding those marked with skip_ci (mimics CI behavior)
pytest -m "not slow"                  # Run tests excluding slow tests
pytest -m integration                 # Run only integration tests
```

## Table of Contents
- [Initial Setup](#initial-setup)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Continuous Integration (CI) Checks](#continuous-integration-ci-checks)
- [Running Tests](#running-tests)
- [Documentation Generation](#documentation-generation)

---

## Initial Setup

```bash
# Create conda environment
conda create -n bio_programming_tools python=3.12 -y
conda activate bio_programming_tools

# Install package in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Pre-commit Hooks

Pre-commit hooks run automatically before every commit to ensure code quality and keep documentation in sync.

### What the Hooks Do

1. **Auto-generate documentation** - Converts tool READMEs to Mintlify MDX format
2. **Import sorting** - Runs `isort` to organize imports
3. **Basic checks** - Removes trailing whitespace, fixes end-of-file issues, validates YAML, checks for large files, checks for merge conflicts

### Running Hooks Manually

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py

# Run a specific hook
pre-commit run generate-docs --all-files
pre-commit run isort --all-files
```

### Bypassing Hooks (Not Recommended)

```bash
git commit --no-verify
```

**Note:** CI will still catch issues if you bypass hooks.

---

## Continuous Integration (CI) Checks

### Automatic CIs

These CIs run automatically on pull requests:

#### Lint Check
**File:** `.github/workflows/flake8_check.yml`
**Triggers:** On all PR pushes and main branch
**What it does:** Checks code style with flake8

**Run locally:**
```bash
flake8 bio_programming_tools tests
```

#### Test Pip Install
**File:** `.github/workflows/test-pip-install.yml`
**Triggers:** On PR pushes
**What it does:** Tests that the package can be installed via pip and runs basic tests

**Run locally:**
```bash
pip install -e .
pytest
```

#### Auto-Generate Documentation Check
**File:** `.github/workflows/docs_check.yml`
**Triggers:** When doc-related files change (tool READMEs, generate_docs.py)
**What it does:** Verifies that generated docs are in sync with source files

This should be covered automatically by the pre-commit hooks, but you can also manually run:

```bash
python docs/generate_docs.py
git add docs/
git commit -m "docs: Auto-generate documentation"
```

#### Mintlify Documentation Validation
**File:** `.github/workflows/docs.yml`
**Triggers:** On pushes to main branch that modify docs
**What it does:** Validates documentation and checks for broken links

**Run locally:**
```bash
cd docs
npm install -g mintlify
mintlify broken-links
```

---

## Running Tests

The test suite includes various markers to categorize and filter tests:

```bash
# Run all tests
pytest

# Run tests excluding slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only CPU tests
pytest -m uses_cpu

# Run tests excluding those marked for CI skip (mimics CI behavior)
pytest --skip-ci

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_blast.py

# Run a specific test function
pytest tests/test_blast.py::test_function_name
```

### Test Markers

- `@pytest.mark.uses_gpu` - Tests requiring GPU
- `@pytest.mark.uses_cpu` - CPU-only tests
- `@pytest.mark.slow` - Tests that may take several minutes
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.skip_ci` - Skip test in GitHub Actions CI (e.g., remote API tests that may hit rate limits)

---

## Documentation Generation

Documentation is auto-generated from tool READMEs in `bio_programming_tools/tools/*/README.md`.

### Documentation Structure

```
docs/
├── tools/              # Auto-generated from tool READMEs
└── docs.json           # Navigation structure (auto-updated)
```

### Adding Documentation for Tools

1. Create a `README.md` in your tool directory: `bio_programming_tools/tools/category/tool_name/README.md`
2. Follow the existing README structure from other tools
3. Commit (docs will auto-generate via pre-commit)

### Manual Documentation Generation

```bash
python docs/generate_docs.py
```

This will:
- Convert tool READMEs to MDX format
- Update `docs.json` navigation

---

## Binary Installation

Tools that need external binaries (not available via pip) use `utils/install_binary.py`. Never download binaries with raw `curl`/`wget` in `setup.sh`.

### Adding a binary to a tool

1. Create `standalone/binary_config.py` with `URLS` dict and `extract()` function (see `blast` or `mmseqs` for examples)
2. Add the standard `install_binary.py` call block to `setup.sh` (same block used by blast, mafft, mmseqs, segmasker)

Existing tools using `install_binary`: blast, mafft, mmseqs, segmasker, minced.

---

## Code Quality

The project uses several tools to maintain code quality:

- **black** (code formatter) - Not currently in pre-commit but available: `black bio_programming_tools tests`
- **isort** (import sorter) - Runs automatically in pre-commit
- **flake8** (linter) - Currently configured to check for unused imports (F401) and unused variables (F841)
- **mypy** (type checker) - Available for type checking: `mypy bio_programming_tools`

### Running Code Quality Tools Manually

```bash
# Format code
black bio_programming_tools tests

# Sort imports
isort bio_programming_tools tests

# Check linting
flake8 bio_programming_tools tests

# Type checking
mypy bio_programming_tools
```
