# Proto Tools

![Proto Tools](https://proto-bio.github.io/proto-assets/covers/open-wings-code/carousel.png)

[![Checks](https://github.com/evo-design/proto-tools/actions/workflows/checks.yml/badge.svg)](https://github.com/evo-design/proto-tools/actions/workflows/checks.yml)
[![Unit Tests](https://github.com/evo-design/proto-tools/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/evo-design/proto-tools/actions/workflows/unit-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/evo-design/proto-tools/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-proto.evodesign.org-blue)](https://proto.evodesign.org/docs/tools/introduction)

Welcome! This repository contains the open-source implementation of `proto-tools`, a Python package containing a large suite of computational biology and biological AI tools, all accessible through a single, consistent Python interface. Language models, structure predictors, inverse folding, sequence analysis, gene annotation, conformational dynamics, genomic scoring, and more are all available through a single `pip install` command.

Every tool runs in its own automatically managed isolated environment, so all dependency wrangling is handled for you. In addition, `proto-tools` implements extensive infrastructure for features such as device management and GPU fan-out, making it easy to call tools in quick succession. You can use it as a standalone Python library, as part of the broader [proto-language](https://github.com/evo-design/proto-language) optimization system, or through the [proto-client](https://github.com/evo-design/proto-client) Python SDK for hosted access over the Proto Bio API. 

Proto-tools is open source under an MIT license. Contributions are welcome!

## Setup

### Step 1: Install the package

Proto-tools requires Python 3.10+:

```bash
pip install git+https://github.com/evo-design/proto-tools.git
```

> [!NOTE]
> A direct PyPI install (`pip install proto-tools`) will be available soon.

> [!NOTE]
> If you are developing or contributing to this project, follow the setup instructions in [CONTRIBUTING.md](CONTRIBUTING.md) instead.

### Step 2: Configure storage (optional)

All persistent data (model weights and tool environments) is cached under the `PROTO_HOME` directory on first use (defaults to `~/.proto/`).

To customize the storage location, you can specify a path via the following environment variable:

```bash
# Add to your shell profile:
export PROTO_HOME=/path/to/your/proto_home
```

For shared filesystems, model weights can be reused to avoid downloading duplicate copies. The `PROTO_MODEL_CACHE` environment variable lets you point just the weights at that shared location (sharing tool environments is not recommended): `export PROTO_MODEL_CACHE=/path/to/shared/weights`. See [notes/storage.md](notes/storage.md) for all details and options.

### Step 3: Gated model access (optional)

A few tools use gated models or software that require accepting a license / terms-of-use first (e.g. ESM3, AlphaGenome, AlphaFold3, X3DNA). See [notes/gated-models.md](notes/gated-models.md) for the full list and per-model access steps.

> [!TIP]
> **You're all set up!** To learn what features are available in the library, check out the [guides](guides/) — four short notebooks covering tool environments, persistent execution, device management, and parallel multi-GPU runs.

## Available Tools

<pre>
<a href="proto_tools/tools/binder_design/">binder_design/</a>                  # De novo antibody / binder design pipelines
├── <a href="proto_tools/tools/binder_design/bindcraft/">bindcraft/</a>
├── <a href="proto_tools/tools/binder_design/freebindcraft/">freebindcraft/</a>
└── <a href="proto_tools/tools/binder_design/germinal/">germinal/</a>
<a href="proto_tools/tools/causal_models/">causal_models/</a>                 # Autoregressive sequence models
├── <a href="proto_tools/tools/causal_models/evo1/">evo1/</a>
├── <a href="proto_tools/tools/causal_models/evo2/">evo2/</a>
├── <a href="proto_tools/tools/causal_models/progen2/">progen2/</a>
└── <a href="proto_tools/tools/causal_models/progen3/">progen3/</a>
<a href="proto_tools/tools/database_retrieval/">database_retrieval/</a>             # Sequence and structure database access
├── <a href="proto_tools/tools/database_retrieval/alphafold_db/">alphafold_db/</a>
├── <a href="proto_tools/tools/database_retrieval/alphamissense_db/">alphamissense_db/</a>
├── <a href="proto_tools/tools/database_retrieval/ccd_lookup/">ccd_lookup/</a>
├── <a href="proto_tools/tools/database_retrieval/ensembl/">ensembl/</a>
├── <a href="proto_tools/tools/database_retrieval/interproscan/">interproscan/</a>
├── <a href="proto_tools/tools/database_retrieval/ncbi/">ncbi/</a>
├── <a href="proto_tools/tools/database_retrieval/pdb/">pdb/</a>
├── <a href="proto_tools/tools/database_retrieval/pubchem/">pubchem/</a>
├── <a href="proto_tools/tools/database_retrieval/sequence_fetch/">sequence_fetch/</a>
└── <a href="proto_tools/tools/database_retrieval/uniprot/">uniprot/</a>
<a href="proto_tools/tools/gene_annotation/">gene_annotation/</a>                # Sequence annotation
├── <a href="proto_tools/tools/gene_annotation/crispr_tracr_rna/">crispr_tracr_rna/</a>
├── <a href="proto_tools/tools/gene_annotation/meme/">meme/</a>
├── <a href="proto_tools/tools/gene_annotation/minced/">minced/</a>
├── <a href="proto_tools/tools/gene_annotation/miranda/">miranda/</a>
├── <a href="proto_tools/tools/gene_annotation/promoter_calculator/">promoter_calculator/</a>
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
└── <a href="proto_tools/tools/masked_models/esmc/">esmc/</a>
<a href="proto_tools/tools/mutagenesis/">mutagenesis/</a>                    # Random sequence mutagenesis
├── <a href="proto_tools/tools/mutagenesis/random_nucleotide/">random_nucleotide/</a>
└── <a href="proto_tools/tools/mutagenesis/random_protein/">random_protein/</a>
<a href="proto_tools/tools/orf_prediction/">orf_prediction/</a>                 # Open reading frame detection
├── <a href="proto_tools/tools/orf_prediction/orfipy/">orfipy/</a>
└── <a href="proto_tools/tools/orf_prediction/prodigal/">prodigal/</a>
<a href="proto_tools/tools/rna_splicing/">rna_splicing/</a>                   # RNA splice site prediction
├── <a href="proto_tools/tools/rna_splicing/pangolin/">pangolin/</a>
├── <a href="proto_tools/tools/rna_splicing/splice_transformer/">splice_transformer/</a>
└── <a href="proto_tools/tools/rna_splicing/spliceai/">spliceai/</a>
<a href="proto_tools/tools/sequence_alignment/">sequence_alignment/</a>             # Sequence search and multiple sequence alignment
├── <a href="proto_tools/tools/sequence_alignment/blast/">blast/</a>
├── <a href="proto_tools/tools/sequence_alignment/mafft/">mafft/</a>
└── <a href="proto_tools/tools/sequence_alignment/mmseqs2/">mmseqs2/</a>
<a href="proto_tools/tools/sequence_scoring/">sequence_scoring/</a>               # Genomic and regulatory scoring
├── <a href="proto_tools/tools/sequence_scoring/alphagenome/">alphagenome/</a>
├── <a href="proto_tools/tools/sequence_scoring/borzoi/">borzoi/</a>
├── <a href="proto_tools/tools/sequence_scoring/deeppbs_specificity/">deeppbs_specificity/</a>
├── <a href="proto_tools/tools/sequence_scoring/enformer/">enformer/</a>
├── <a href="proto_tools/tools/sequence_scoring/malinois/">malinois/</a>
├── <a href="proto_tools/tools/sequence_scoring/na_mpnn_specificity/">na_mpnn_specificity/</a>
├── <a href="proto_tools/tools/sequence_scoring/puffin/">puffin/</a>
└── <a href="proto_tools/tools/sequence_scoring/segmasker/">segmasker/</a>
<a href="proto_tools/tools/structure_alignment/">structure_alignment/</a>            # Structure comparison
├── <a href="proto_tools/tools/structure_alignment/foldmason/">foldmason/</a>
├── <a href="proto_tools/tools/structure_alignment/foldseek/">foldseek/</a>
├── <a href="proto_tools/tools/structure_alignment/pymol_rmsd/">pymol_rmsd/</a>
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
├── <a href="proto_tools/tools/structure_prediction/esmfold2/">esmfold2/</a>
├── <a href="proto_tools/tools/structure_prediction/protenix/">protenix/</a>
├── <a href="proto_tools/tools/structure_prediction/rf3/">rf3/</a>
├── <a href="proto_tools/tools/structure_prediction/viennarna/">viennarna/</a>
└── <a href="proto_tools/tools/structure_prediction/x3dna/">x3dna/</a>
<a href="proto_tools/tools/structure_scoring/">structure_scoring/</a>              # Structure quality scoring
├── <a href="proto_tools/tools/structure_scoring/dssp/">dssp/</a>
├── <a href="proto_tools/tools/structure_scoring/ipsae/">ipsae/</a>
├── <a href="proto_tools/tools/structure_scoring/pdockq2/">pdockq2/</a>
├── <a href="proto_tools/tools/structure_scoring/pyrosetta/">pyrosetta/</a>
└── <a href="proto_tools/tools/structure_scoring/structure_metrics/">structure_metrics/</a>
</pre>

## Guides

Runnable walkthroughs of the core framework features live in [`guides/`](guides/) and are also available on our [docs page](https://proto.evodesign.org/docs/tools/introduction):

1. [Tool Environments](guides/tool_environments.ipynb) — how isolated environments are built and cached on first call.
2. [Tool Persistence](guides/tool_persistence.ipynb) — keep models warm across calls
3. [Device Management](guides/device_management.ipynb) — GPU allocation, LRU eviction, CPU offload
4. [Parallel Execution](guides/parallel_execution.ipynb) — fan out work across every GPU with `ToolPool`

Each specific tool also ships a minimal `examples/example.ipynb` under `proto_tools/tools/{category}/{tool}/examples/`.

## Using with a coding agent

Run tools through natural language with any coding agent (Claude Code, Gemini CLI, OpenAI Codex CLI, etc.). Point the agent at `proto-tools agent-context`: it prints a primer covering the `Input → Config → run_*() → Output` pattern, the offline CLI discovery verbs, persistence and parallel execution, and links to the long-form notes on GitHub. The command ships in the wheel, so it works on a plain `pip install` with no repo checkout.

If you've cloned the repo for contributing, agents also pick up `CLAUDE.md` (symlinked as `AGENTS.md`/`GEMINI.md`) and the task-specific guides in [`.claude/skills/`](.claude/skills/) automatically.

## Development & Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for full developer setup, storage
configuration, PR format, code style, and testing conventions.
