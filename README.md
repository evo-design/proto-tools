<img src="https://evodesign.org/images/logo.svg" alt="Evo Design Logo" width="120" align="left" style="margin-right: 15px;">

# Tools Hub for Biological AI Models and Computational Biology
[![Lint Check](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml)
[![Run Unit Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml)
[![Run Integration Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-integration-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-integration-tests.yml)
[![docs_autogen](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_autogen.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_autogen.yml)

## Related Repositories

### Backend
* [`bio-programming`](https://github.com/evo-design/bio-programming) – The primary backend repository (includes this repo as a submodule).

### Frontend
* [`bio-programming-tools-ui`](https://github.com/evo-design/bio-programming-tools-ui) – Mock UI specifically designed for demonstrating and testing these tools.
* [`bio-programming-lang`](https://github.com/evo-design/bio-programming-lang) – The dedicated frontend interface for the biological programming language.

## Installation

### Recommended: Conda environment file

Using the provided `environment.yml` installs Python, compilers, build tools, and system
libraries that tool environments need when compiling from source. This is the most stable
setup, especially on HPC and GPU nodes.

```bash
conda env create -f environment.yml
conda activate bio-tools

# For development
pip install -e ".[dev]"
pre-commit install
```

### Minimal setup

If you already have compilers and system libraries available (e.g., via modules), a
plain conda or venv environment works too:

```bash
# Using conda
conda create -n bio-tools python=3.12 -y
conda activate bio-tools
pip install -e "."

# Or using venv
python -m venv .venv
source .venv/bin/activate
pip install -e "."

# For development
pip install -e ".[dev]"
pre-commit install
```

## HuggingFace Authentication

Some models are hosted in gated HuggingFace repositories that require both authentication and accepting the model's license/terms. The following models require this:

| Model | HuggingFace Repo | Notes |
|-------|-----------------|-------|
| ESM3 | [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) | Requires accepting EvolutionaryScale license |
| AlphaGenome | [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds) | Requires accepting Google DeepMind terms |

To use these models:

1. Create a [HuggingFace](https://huggingface.co) account
2. Visit each model page above and **accept the license/terms**
3. Create an [access token](https://huggingface.co/settings/tokens)
4. Install [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   ```

5. Set the token in your environment:
   ```bash
   export HF_TOKEN=hf_...
   # Or log in with: hf auth login
   ```

The setup scripts for gated models will check for access and provide a clear error if authentication is missing.

## Usage

```python
from bio_programming_tools.tools.gene_annotation.blast import (
    run_blast_search, BlastSearchInput, BlastSearchConfig,
)

inputs = BlastSearchInput(query="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK...")
config = BlastSearchConfig(program="blastp", database="swissprot")
result = run_blast_search(inputs, config)

print(result.num_hits)
print(result.results_df.head())
```

### Persistent Execution

For batch workloads, keep the model loaded across calls to avoid redundant load times:

```python
from bio_programming_tools.utils.tool_instance import ToolInstance
from bio_programming_tools.tools.structure_prediction.esmfold import (
    run_esmfold, ESMFoldInput, ESMFoldConfig,
)

# Model loads once, reused across all calls in the block
with ToolInstance.persist():
    for seq in sequences:
        result = run_esmfold(ESMFoldInput(complexes=[seq]), ESMFoldConfig())
```

See `notes/tool_instance_example.ipynb` for a full walkthrough with timing comparisons.

## Using with Claude Code

This repo includes [Claude Code](https://docs.anthropic.com/en/docs/claude-code) integration for both users running tools and developers extending the library. Launch `claude` from the repo root:

```bash
claude
```

Ask Claude things like:

```
> BLAST this sequence against swissprot
> Predict the structure of MKFLIL... with ESMFold
> Fold this sequence, redesign it with inverse folding, and compare the original and designed sequences
```

It will write runnable scripts to `analyses/` or execute directly depending on context. See `CLAUDE.md` for the full workflow guide.

### For developers (extending the tool library)

Commands (invoked with `/command-name`):

- **`/fix-issue <number>`** — full GitHub issue fix lifecycle (read issue, explore, reproduce, fix, test, verify)
- **`/implement-tool`** — step-by-step guide for implementing a new tool wrapper (architecture, templates, export chain, examples, tests)

Every tool follows the same `Input` / `Config` / `run_{tool}()` / `Output` pattern.

## Development

To run tests, linting, and other code quality tools, install with the dev extras:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
pytest                          # CPU unit tests (skips GPU, slow, integration)
pytest --gpu                    # GPU tests only (skip CPU tests)
pytest --integration            # Add integration tests (external APIs) + GPU if available
pytest --all                    # Everything: GPU + slow + integration
pytest --all --cpu              # Slow + integration, but skip GPU
pytest --env-report             # Generate environment compatibility report
pytest --env-report --cpu       # CPU tools only
```
