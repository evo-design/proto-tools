# `bio_programming_tools`

[![Lint Check](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml)
[![Run Unit Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml)
[![Verify Documentation autogeneration synced](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_check.yml)
[![Validate Docs](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs.yml)

This repo contains the tool layer of the [`bio-programming`](https://github.com/evo-design/bio-programming/tree/main) project.


> [!NOTE]
> We currently in the process of transferring all of the infra from the `bio-programming` repo to this one. Note that some tests may not be passing on main at the moment.


## Installation

```bash
# Using conda
conda create -n bio_programming_tools python=3.12 -y
conda activate bio_programming_tools

# Or using venv
python -m venv .venv
source .venv/bin/activate


pip install -e .
```


## Usage

```python
from bio_tools.tools.gene_annotation.blast.online_blast import (
    run_online_blast_search, OnlineBlastInput, OnlineBlastConfig,
)

inputs = OnlineBlastInput(query="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK...")
config = OnlineBlastConfig(program="blastp", database="swissprot")
result = run_online_blast_search(inputs, config)

print(result.num_hits)
print(result.results_df.head())
```


## Using with Claude Code

This repo includes a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill that helps Claude discover and utilize the bio tools. Just launch `claude` from the repo root after installing:

```bash
claude
```

Claude automatically picks up the project instructions (`CLAUDE.md`) and the bio-tools skill. Ask it things like:

```
> BLAST this sequence against swissprot
> Predict the structure of MKFLIL... with ESMFold
> Fold this sequence, redesign it with inverse folding, and compare the original and designed sequences
```

It will write runnable scripts to `analyses/` or execute directly depending on context. See [`analyses/examples/`](analyses/examples/) for reference scripts.


Every tool follows the same `Input` / `Config` / `run_{tool}()` / `Output` pattern. See [`analyses/examples/`](analyses/examples/) for complete runnable scripts.

## Development

To run tests, linting, and other code quality tools, install with the dev extras:

```bash
pip install -e ".[dev]"
```
