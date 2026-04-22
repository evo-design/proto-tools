> [!IMPORTANT]
> This repo contains an unpublished project codebase. Do not share the source code externally.
> If you are a beta tester, start here: [Beta Tester Instructions](notes/beta-welcome.md)

# 🧬 Proto Tools 🛠️
[![Checks](https://github.com/proto-bio/proto-tools/actions/workflows/checks.yml/badge.svg)](https://github.com/proto-bio/proto-tools/actions/workflows/checks.yml)
[![Unit Tests](https://github.com/proto-bio/proto-tools/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/proto-bio/proto-tools/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/proto-bio/proto-tools/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/proto-bio/proto-tools/actions/workflows/integration-tests.yml)
[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/evs3Unkegv)


Welcome! This repository contains the **tools layer** of the biological programming language project. It puts **60+ computational biology and biological AI tools** at your fingertips through a single, consistent Python interface. Protein language models, structure predictors, inverse folding, sequence analysis, gene annotation, conformational dynamics, genomic scoring, and more are all just one function call away.

Every tool runs in its own automatically managed isolated environment, so you never have to fight dependency conflicts or build complex conda setups. Just install, call, and get results. Run tools locally on your own hardware, or dispatch to remote compute.

> [!TIP]
> **COMING SOON:** Cloud dispatch will allow you to run GPU jobs even without local compute.

You can use it as a standalone Python library or as part of the broader [proto-language](https://github.com/proto-bio/proto-language) system.

## Related Repositories

- [`proto-language`](https://github.com/proto-bio/proto-language) – Core language framework (constraints, generators, optimizers)
- [`proto-client`](https://github.com/proto-bio/proto-client) – Python SDK for Proto Bio APIs

## Setup 🚧

> [!NOTE]
> **On Stanford Sherlock?** The cluster's old glibc requires a container-based setup. See the [Sherlock HPC Setup Guide](notes/sherlock-setup.md) for step-by-step instructions.

### Step 0: Clone the repository 📦

```bash
git clone https://github.com/proto-bio/proto-tools.git
cd proto-tools
```

> [!NOTE]
> In the future, we plan to enable a direct PyPI install (`pip install proto-tools`), but prior to the public release we will be using this local install approach.

### Step 1: Install the package 🐍

All you need is Python 3.10+ and pip:

```bash
pip install .
```

System tools that standalone tool environments need (git, curl, gcc) are automatically provisioned on first use via a shared foundation environment — no manual setup required.

> **Note:** If you are developing or contributing to this project, follow the setup instructions in [CONTRIBUTING.md](CONTRIBUTING.md) instead.

### Step 2: Configure storage (optional) 🗂️

All persistent data (model weights, tool environments, micromamba) is stored under `PROTO_HOME` (defaults to `~/.proto/`).

To customize the storage location (recommended for labs/HPC):

```bash
# Add to your ~/.bashrc:
export PROTO_HOME=/path/to/your/proto_home
```

To override just model weights separately: `export PROTO_MODEL_CACHE=/path/to/shared/weights`. See [notes/storage.md](notes/storage.md) for all options.

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

> [!TIP]
> **You're all set up!** To learn what features are available in the library, check out the [tutorial series](tutorials/) — four short notebooks covering the universal tool pattern, persistent execution, device management, and parallel multi-GPU runs.

## Available Tools 🔬

<pre>
<a href="proto_tools/tools/causal_models/">causal_models/</a>                 # Autoregressive sequence models
├── <a href="proto_tools/tools/causal_models/evo1/">evo1/</a>
├── <a href="proto_tools/tools/causal_models/evo2/">evo2/</a>
├── <a href="proto_tools/tools/causal_models/progen2/">progen2/</a>
└── <a href="proto_tools/tools/causal_models/progen3/">progen3/</a>
<a href="proto_tools/tools/database_retrieval/">database_retrieval/</a>             # Sequence and structure database access
├── <a href="proto_tools/tools/database_retrieval/ncbi/">ncbi/</a>
├── <a href="proto_tools/tools/database_retrieval/pdb/">pdb/</a>
├── <a href="proto_tools/tools/database_retrieval/sequence_fetch/">sequence_fetch/</a>
└── <a href="proto_tools/tools/database_retrieval/uniprot/">uniprot/</a>
<a href="proto_tools/tools/gene_annotation/">gene_annotation/</a>                # Sequence annotation and homology search
├── <a href="proto_tools/tools/gene_annotation/blast/">blast/</a>
├── <a href="proto_tools/tools/gene_annotation/crispr_tracr/">crispr_tracr/</a>
├── <a href="proto_tools/tools/gene_annotation/minced/">minced/</a>
├── <a href="proto_tools/tools/gene_annotation/mmseqs/">mmseqs/</a>
└── <a href="proto_tools/tools/gene_annotation/pyhmmer/">pyhmmer/</a>
<a href="proto_tools/tools/inverse_folding/">inverse_folding/</a>                # Sequence design from structures
├── <a href="proto_tools/tools/inverse_folding/esm_if1/">esm_if1/</a>
├── <a href="proto_tools/tools/inverse_folding/fampnn/">fampnn/</a>
├── <a href="proto_tools/tools/inverse_folding/ligandmpnn/">ligandmpnn/</a>
└── <a href="proto_tools/tools/inverse_folding/proteinmpnn/">proteinmpnn/</a>
<a href="proto_tools/tools/masked_models/">masked_models/</a>                  # Masked language models
├── <a href="proto_tools/tools/masked_models/ablang/">ablang/</a>
├── <a href="proto_tools/tools/masked_models/esm2/">esm2/</a>
├── <a href="proto_tools/tools/masked_models/esm3/">esm3/</a>
├── <a href="proto_tools/tools/masked_models/esmc/">esmc/</a>
└── <a href="proto_tools/tools/masked_models/masking/">masking/</a>
<a href="proto_tools/tools/mutagenesis/">mutagenesis/</a>                    # Random sequence mutagenesis
├── <a href="proto_tools/tools/mutagenesis/random_nucleotide/">random_nucleotide/</a>
└── <a href="proto_tools/tools/mutagenesis/random_protein/">random_protein/</a>
<a href="proto_tools/tools/orf_prediction/">orf_prediction/</a>                 # Open reading frame detection
├── <a href="proto_tools/tools/orf_prediction/orfipy/">orfipy/</a>
└── <a href="proto_tools/tools/orf_prediction/prodigal/">prodigal/</a>
<a href="proto_tools/tools/rna_splicing/">rna_splicing/</a>                   # RNA splice site prediction
└── <a href="proto_tools/tools/rna_splicing/splice_transformer/">splice_transformer/</a>
<a href="proto_tools/tools/sequence_alignment/">sequence_alignment/</a>             # Multiple sequence alignment
├── <a href="proto_tools/tools/sequence_alignment/colabfold_search/">colabfold_search/</a>
└── <a href="proto_tools/tools/sequence_alignment/mafft/">mafft/</a>
<a href="proto_tools/tools/sequence_scoring/">sequence_scoring/</a>               # Genomic and regulatory scoring
├── <a href="proto_tools/tools/sequence_scoring/alphagenome/">alphagenome/</a>
├── <a href="proto_tools/tools/sequence_scoring/borzoi/">borzoi/</a>
├── <a href="proto_tools/tools/sequence_scoring/enformer/">enformer/</a>
└── <a href="proto_tools/tools/sequence_scoring/segmasker/">segmasker/</a>
<a href="proto_tools/tools/structure_alignment/">structure_alignment/</a>            # Structure comparison
├── <a href="proto_tools/tools/structure_alignment/tmalign/">tmalign/</a>
└── <a href="proto_tools/tools/structure_alignment/usalign/">usalign/</a>
<a href="proto_tools/tools/structure_design/">structure_design/</a>               # De novo structure generation
└── <a href="proto_tools/tools/structure_design/rfdiffusion3/">rfdiffusion3/</a>
<a href="proto_tools/tools/structure_dynamics/">structure_dynamics/</a>             # Conformational dynamics
└── <a href="proto_tools/tools/structure_dynamics/bioemu/">bioemu/</a>
<a href="proto_tools/tools/structure_prediction/">structure_prediction/</a>           # 3D structure prediction
├── <a href="proto_tools/tools/structure_prediction/alphafold2/">alphafold2/</a>
├── <a href="proto_tools/tools/structure_prediction/alphafold3/">alphafold3/</a>
├── <a href="proto_tools/tools/structure_prediction/boltz2/">boltz2/</a>
├── <a href="proto_tools/tools/structure_prediction/chai1/">chai1/</a>
├── <a href="proto_tools/tools/structure_prediction/esmfold/">esmfold/</a>
├── <a href="proto_tools/tools/structure_prediction/protenix/">protenix/</a>
└── <a href="proto_tools/tools/structure_prediction/viennarna/">viennarna/</a>
<a href="proto_tools/tools/structure_scoring/">structure_scoring/</a>              # Structure quality scoring
├── <a href="proto_tools/tools/structure_scoring/pdockq2/">pdockq2/</a>
├── <a href="proto_tools/tools/structure_scoring/pyrosetta/">pyrosetta/</a>
└── <a href="proto_tools/tools/structure_scoring/structure_metrics/">structure_metrics/</a>
</pre>

## Usage

Every tool follows the same `Input` / `Config` / `run_{tool}()` / `Output` pattern. Config is always optional; omit it to use defaults.

```python
from proto_tools.tools.gene_annotation.blast import (
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
from proto_tools.utils.tool_instance import ToolInstance
from proto_tools.tools.structure_prediction.esmfold import (
    run_esmfold, ESMFoldInput, ESMFoldConfig,
)

# Model loads once, reused across all calls in the block
with ToolInstance.persist():
    for seq in sequences:
        result = run_esmfold(ESMFoldInput(complexes=[seq]), ESMFoldConfig())
```

## Tutorials

Runnable walkthroughs of the core framework features live in [`tutorials/`](tutorials/):

1. [Getting Started](tutorials/01_getting_started.ipynb) — the universal tool pattern (CPU-only)
2. [Tool Persistence](tutorials/02_tool_persistence.ipynb) — keep models warm across calls
3. [Device Management](tutorials/03_device_management.ipynb) — GPU allocation, LRU eviction, CPU offload
4. [Parallel Execution](tutorials/04_parallel_execution.ipynb) — fan out work across every GPU with `ToolPool`

Each specific tool also ships a minimal `examples/example.ipynb` under `proto_tools/tools/{category}/{tool}/examples/`.

## MCP Servers

### proto-tools: tool discovery & schemas

The proto-tools MCP server has been migrated to [proto-tools-api](https://github.com/proto-bio/proto-tools-api). See that repo for setup and usage.

### proto-language: DSL, constraints, generators

A separate MCP server exposes the language layer: constraints, generators, optimizers, program validation, and doc search. See the [proto-language repo](https://github.com/proto-bio/proto-language) for details.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for full developer setup, storage configuration, PR format, code style, and testing conventions.

### Claude Code features for developers

Slash commands for common development tasks (invoked with `/command-name`):
- **`/implement-tool`**: step-by-step guide for implementing a new tool wrapper (architecture, templates, export chain, examples, tests)

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
