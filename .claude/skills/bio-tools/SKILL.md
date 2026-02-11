---
name: bio-tools
description: >
  ALWAYS use this skill when the user asks to run, analyze, or write scripts
  for ANY bioinformatics tool — including BLAST, MMseqs2, PyHMMER, MAFFT,
  AlphaFold3, Boltz2, Chai1, ESMFold, Protenix, ProteinMPNN, LigandMPNN,
  ESM2, ESM3, ProGen2, Evo2, ColabFold Search, Enformer, Borzoi, AlphaGenome,
  RFDiffusion3, BioEmu, Orfipy, Prodigal, ViennaRNA, SpliceTransformer, or
  any sequence/structure analysis. Invoke this skill FIRST before reading tool
  source code or writing any analysis script.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# bio-tools skill

You help users run bioinformatics analyses using the `bio_programming_tools` library.

## How to use a tool

1. **Discover:** Glob for `bio_programming_tools/tools/*/*/README.md` to find all tools
2. **Read README:** Read the matching tool's `README.md` — has parameters (with types
   and defaults), thresholds, biological context, and quick-start examples
3. **Read notebook:** Read the tool's `examples/example.ipynb` — has working code with
   exact imports and real output examples
4. **Call pattern:** `Input` → `Config` → `run_{tool}()` → `Output`

Do NOT read tool source code (.py files) for API details — the README and
notebook already contain complete parameter tables and usage examples.

**Important:** Do NOT activate a virtual environment (e.g. `source .venv/bin/activate`)
before running tools. The package is already installed — just use `python3` directly.

## Tool categories

Categories under `bio_programming_tools/tools/`: gene_annotation, orf_prediction,
sequence_alignment, sequence_scoring, inverse_folding, structure_prediction,
structure_design, structure_dynamics, masked_models, causal_models, rna_splicing

## Script vs direct execution

Infer from context:

- **Write a script** (`analyses/{name}_{date}.py`): "write", "create", "set up",
  multi-step pipelines, GPU jobs, or when unclear
- **Execute directly** (inline via Bash): "what is", "how many", "show me",
  "quick", simple lookups

Always show the equivalent Python code either way.

## Key conventions

- Every tool uses the pattern: `Input` → `Config` → `run_{tool}()` → `Output`
- Check the tool's README for GPU requirements — note this in generated scripts
- Multi-step pipelines use `# === Step N: Description ===` headers
- All public symbols are re-exported from `bio_programming_tools.tools`
