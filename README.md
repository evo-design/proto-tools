<img src="https://evodesign.org/images/logo.svg" alt="Evo Design Logo" width="120" align="left" style="margin-right: 15px;">

# Tools Hub for Biological AI Models and Computational Biology
[![Lint Check](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml)
[![Run Unit Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml)
[![Verify Documentation autogeneration synced](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_check.yml)
[![Validate Docs](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs.yml)

## Related Repositories

### Backend
* [`bio-programming`](https://github.com/evo-design/bio-programming) – The primary backend repository (includes this repo as a submodule).

### Client
* [`bio-programming-tools-ui`](https://github.com/evo-design/bio-programming-tools-ui) – Mock UI specifically designed for demonstrating and testing these tools.
* [`bio-programming-lang`](https://github.com/evo-design/bio-programming-lang) – The dedicated client interface for the biological programming language.

## Installation

```bash
# Using conda (recommended)
conda create -n bio_tools python=3.12 -y
conda activate bio_tools

# Or using venv
python -m venv .venv
source .venv/bin/activate

pip install .

# For development
pip install ".[dev]"
pre-commit install
```

## HuggingFace Authentication

Some models are hosted in gated HuggingFace repositories that require both authentication and accepting the model's license/terms. The following models require this:

| Model | HuggingFace Repo | Notes |
|-------|-----------------|-------|
| ESM3 | [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) | Requires accepting EvolutionaryScale license |
| AlphaGenome | [google/alphagenome](https://huggingface.co/google/alphagenome) | Requires accepting Google DeepMind terms |

To use these models:

1. Create a [HuggingFace](https://huggingface.co) account
2. Visit each model page above and **accept the license/terms**
3. Create an [access token](https://huggingface.co/settings/tokens)
4. Set the token in your environment:
   ```bash
   export HF_TOKEN=hf_...
   # Or log in with: huggingface-cli login
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

See `bio_programming_tools/tools/tool_instance_example.ipynb` for a full walkthrough with timing comparisons.

## Using with Claude Code

This repo includes [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills for both users running tools and developers extending the library. Launch `claude` from the repo root:

```bash
claude
```

### For users (running tools)

- **bio-tools** — workflow for running, analyzing, and writing scripts for any bioinformatics tool (discovery, script generation, GPU handling, output conventions)

Ask Claude things like:

```
> BLAST this sequence against swissprot
> Predict the structure of MKFLIL... with ESMFold
> Fold this sequence, redesign it with inverse folding, and compare the original and designed sequences
```

It will write runnable scripts to `analyses/` or execute directly depending on context. See [`analyses/examples/`](analyses/examples/) for reference scripts.

### For developers (extending the tool library)

Commands (invoked with `/command-name`):

- **`/fix-issue <number>`** — full GitHub issue fix lifecycle (read issue, explore, reproduce, fix, test, verify)
- **`/implement-tool`** — step-by-step guide for implementing a new tool wrapper (architecture, templates, export chain, examples, tests)


Every tool follows the same `Input` / `Config` / `run_{tool}()` / `Output` pattern. See [`analyses/examples/`](analyses/examples/) for complete runnable scripts.

## Development

To run tests, linting, and other code quality tools, install with the dev extras:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
pytest                          # Fast tests (skips slow)
pytest --cpu                    # CPU tests only
pytest --all                    # Include slow tests
pytest --run-all-venvs          # One smoke test per standalone venv tool
pytest --run-all-venvs --cpu    # CPU venv smoke tests only
```
