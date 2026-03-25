<img src="https://evodesign.org/images/logo.svg" alt="Evo Design Logo" width="120" align="left" style="margin-right: 15px;">

# 🧬 Biological Programming Language Project - Tools Layer 🛠️
[![Lint Check](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/flake8_check.yml)
[![Run Unit Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-unit-tests.yml)
[![Run Integration Tests](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-integration-tests.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/run-integration-tests.yml)
[![docs_autogen](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_autogen.yml/badge.svg)](https://github.com/evo-design/bio-programming-tools/actions/workflows/docs_autogen.yml)

Welcome! This repository contains the **tools layer** of the biological programming language project. It puts **60+ computational biology and biological AI tools** at your fingertips through a single, consistent Python interface. Protein language models, structure predictors, inverse folding, sequence analysis, gene annotation, conformational dynamics, genomic scoring, and more are all just one function call away.

Every tool runs in its own automatically managed isolated environment, so you never have to fight dependency conflicts or build complex conda setups. Just install, call, and get results. Run tools locally on your own hardware, or dispatch to remote compute.

You can use it as a standalone Python library or as part of the broader [bio-programming](https://github.com/evo-design/bio-programming) system.

## Setup 🚧

> [!NOTE]
> **On Stanford Sherlock?** The cluster's old glibc requires a container-based setup. See the [Sherlock HPC Setup Guide](notes/sherlock-setup.md) for step-by-step instructions.

### Step 1: Install the package 🐍

We recommend using the provided `environment.yml`, which sets up Python, compilers, build tools, and system libraries that some tool environments need when compiling from source:

```bash
conda env create -f environment.yml
conda activate bio-tools
pip install -e "."
```

Most newer systems will already have compilers and build tools available at a system level, which should allow you to just `pip install -e "."` in any Python 3.10+ environment.

For development (tests, linting, pre-commit hooks):

```bash
pip install -e ".[dev]"
pre-commit install
```

### Step 2: Model weights cache (optional) 🗂️

Model weights are stored in `model_cache/` at the repo root by default (gitignored). This keeps weights outside tool environments so they survive env rebuilds.

To share weights across a lab or redirect to larger storage, create a `.bpt.env` file (see `.bpt.env.example`):

```bash
cp .bpt.env.example .bpt.env
# Edit .bpt.env:
BPT_MODEL_CACHE=/path/to/shared/weights
```

Or set the environment variable directly: `export BPT_MODEL_CACHE=/path/to/shared/weights`. See [docs/tool-environments.md](docs/tool-environments.md) for all options.

### Step 3: HuggingFace authentication (optional) 🔑

Some tools use gated models that require a HuggingFace account and accepting the model's license/terms:

| Model | HuggingFace Repo | Notes |
|-------|-----------------|-------|
| ESM3 | [EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1) | Requires accepting EvolutionaryScale license |
| AlphaGenome | [google/alphagenome-all-folds](https://huggingface.co/google/alphagenome-all-folds) | Requires accepting Google DeepMind terms |

To use these models:

1. Create a [HuggingFace](https://huggingface.co) account
2. Visit each model page above and **accept the license/terms**
3. Install the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and log in:
   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   hf auth login
   ```
   Or set the token directly in your environment:
   ```bash
   export HF_TOKEN=hf_...
   ```

## Available Tools 🔬

<pre>
<a href="bio_programming_tools/tools/causal_models/">causal_models/</a>                 # Autoregressive sequence models
├── <a href="bio_programming_tools/tools/causal_models/evo1/">evo1/</a>
├── <a href="bio_programming_tools/tools/causal_models/evo2/">evo2/</a>
└── <a href="bio_programming_tools/tools/causal_models/progen2/">progen2/</a>
<a href="bio_programming_tools/tools/database_retrieval/">database_retrieval/</a>             # Sequence and structure database access
├── <a href="bio_programming_tools/tools/database_retrieval/ncbi/">ncbi/</a>
├── <a href="bio_programming_tools/tools/database_retrieval/pdb/">pdb/</a>
├── <a href="bio_programming_tools/tools/database_retrieval/sequence_fetch/">sequence_fetch/</a>
└── <a href="bio_programming_tools/tools/database_retrieval/uniprot/">uniprot/</a>
<a href="bio_programming_tools/tools/gene_annotation/">gene_annotation/</a>                # Sequence annotation and homology search
├── <a href="bio_programming_tools/tools/gene_annotation/blast/">blast/</a>
├── <a href="bio_programming_tools/tools/gene_annotation/crispr_tracr/">crispr_tracr/</a>
├── <a href="bio_programming_tools/tools/gene_annotation/minced/">minced/</a>
├── <a href="bio_programming_tools/tools/gene_annotation/mmseqs/">mmseqs/</a>
└── <a href="bio_programming_tools/tools/gene_annotation/pyhmmer/">pyhmmer/</a>
<a href="bio_programming_tools/tools/inverse_folding/">inverse_folding/</a>                # Sequence design from structures
├── <a href="bio_programming_tools/tools/inverse_folding/ligandmpnn/">ligandmpnn/</a>
└── <a href="bio_programming_tools/tools/inverse_folding/proteinmpnn/">proteinmpnn/</a>
<a href="bio_programming_tools/tools/masked_models/">masked_models/</a>                  # Masked language models
├── <a href="bio_programming_tools/tools/masked_models/esm2/">esm2/</a>
└── <a href="bio_programming_tools/tools/masked_models/esm3/">esm3/</a>
<a href="bio_programming_tools/tools/mutagenesis/">mutagenesis/</a>                    # Random sequence mutagenesis
├── <a href="bio_programming_tools/tools/mutagenesis/random_nucleotide/">random_nucleotide/</a>
└── <a href="bio_programming_tools/tools/mutagenesis/random_protein/">random_protein/</a>
<a href="bio_programming_tools/tools/orf_prediction/">orf_prediction/</a>                 # Open reading frame detection
├── <a href="bio_programming_tools/tools/orf_prediction/orfipy/">orfipy/</a>
└── <a href="bio_programming_tools/tools/orf_prediction/prodigal/">prodigal/</a>
<a href="bio_programming_tools/tools/rna_splicing/">rna_splicing/</a>                   # RNA splice site prediction
└── <a href="bio_programming_tools/tools/rna_splicing/splice_transformer/">splice_transformer/</a>
<a href="bio_programming_tools/tools/sequence_alignment/">sequence_alignment/</a>             # Multiple sequence alignment
├── <a href="bio_programming_tools/tools/sequence_alignment/colabfold_search/">colabfold_search/</a>
└── <a href="bio_programming_tools/tools/sequence_alignment/mafft/">mafft/</a>
<a href="bio_programming_tools/tools/sequence_scoring/">sequence_scoring/</a>               # Genomic and regulatory scoring
├── <a href="bio_programming_tools/tools/sequence_scoring/alphagenome/">alphagenome/</a>
├── <a href="bio_programming_tools/tools/sequence_scoring/borzoi/">borzoi/</a>
├── <a href="bio_programming_tools/tools/sequence_scoring/enformer/">enformer/</a>
└── <a href="bio_programming_tools/tools/sequence_scoring/segmasker/">segmasker/</a>
<a href="bio_programming_tools/tools/structure_alignment/">structure_alignment/</a>            # Structure comparison
├── <a href="bio_programming_tools/tools/structure_alignment/tmalign/">tmalign/</a>
└── <a href="bio_programming_tools/tools/structure_alignment/usalign/">usalign/</a>
<a href="bio_programming_tools/tools/structure_design/">structure_design/</a>               # De novo structure generation
└── <a href="bio_programming_tools/tools/structure_design/rfdiffusion3/">rfdiffusion3/</a>
<a href="bio_programming_tools/tools/structure_dynamics/">structure_dynamics/</a>             # Conformational dynamics
└── <a href="bio_programming_tools/tools/structure_dynamics/bioemu/">bioemu/</a>
<a href="bio_programming_tools/tools/structure_prediction/">structure_prediction/</a>           # 3D structure prediction
├── <a href="bio_programming_tools/tools/structure_prediction/alphafold2/">alphafold2/</a>
├── <a href="bio_programming_tools/tools/structure_prediction/alphafold3/">alphafold3/</a>
├── <a href="bio_programming_tools/tools/structure_prediction/boltz2/">boltz2/</a>
├── <a href="bio_programming_tools/tools/structure_prediction/chai1/">chai1/</a>
├── <a href="bio_programming_tools/tools/structure_prediction/esmfold/">esmfold/</a>
├── <a href="bio_programming_tools/tools/structure_prediction/protenix/">protenix/</a>
├── <a href="bio_programming_tools/tools/structure_prediction/structure_metrics/">structure_metrics/</a>
└── <a href="bio_programming_tools/tools/structure_prediction/viennarna/">viennarna/</a>
</pre>

## Usage

Every tool follows the same `Input` / `Config` / `run_{tool}()` / `Output` pattern. Config is always optional — omit it to use defaults.

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

## MCP Servers

Two MCP servers expose the bio-programming ecosystem to AI agents. Both support stdio (Claude Desktop / Claude Code) and HTTP transports.

### bio-programming-tools (this repo) — tool discovery & schemas

Wraps the ToolRegistry so agents can discover tools, inspect schemas, search by keyword, and read citations.

```bash
# stdio (auto-discovered by Claude Code via .mcp.json)
python -m bio_tools_mcp

# HTTP
python -m bio_tools_mcp --transport http --port 9200
```

Install: `pip install -e ".[mcp]"`

**Example — find structure prediction tools and inspect one:**
```
> list_tools(category="structure_prediction")
> get_tool_schema(key="esmfold-prediction")
> search_tools(query="inverse folding")
```

### bio-programming (parent repo) — DSL, constraints, generators

Lives in [`bio-programming/bio_mcp/`](https://github.com/evo-design/bio-programming). Exposes the language layer: constraints, generators, optimizers, program validation, and doc search.

```bash
# From the bio-programming repo root
python -m bio_mcp
```

**Example — explore the DSL and validate a program:**
```
> list_constraints(category="sequence")
> list_generators()
> get_component_schema(component_type="constraint", key="gc-content")
> validate_program(program_json='{"constructs": [...]}')
```

## Using with Claude Code

Run tools interactively through natural language using [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Launch `claude` from the repo root:

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

## Development & Contributing

Contributing new tools, running tests, and developer workflows. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide, including PR format, code style, and testing conventions.

### Claude Code features for developers

Slash commands for common development tasks (invoked with `/command-name`):
- **`/fix-issue <number>`** — full GitHub issue fix lifecycle (read issue, explore, reproduce, fix, test, verify)
- **`/implement-tool`** — step-by-step guide for implementing a new tool wrapper (architecture, templates, export chain, examples, tests)

Every tool follows the same `Input` / `Config` / `run_{tool}()` / `Output` pattern.

### Testing

Tests are split by resource requirements. GPU and integration tests are skipped by default.

```bash
pytest                          # CPU unit tests (skips GPU, slow, integration)
pytest --gpu                    # GPU tests only (skip CPU tests)
pytest --integration            # Add integration tests (external APIs) + GPU if available
pytest --all                    # Everything: GPU + slow + integration
pytest --all --cpu              # Slow + integration, but skip GPU
pytest --env-report             # Generate environment compatibility report
pytest --env-report --cpu       # CPU tools only
```
