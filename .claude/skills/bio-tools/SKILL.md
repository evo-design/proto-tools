---
name: bio-tools
description: >
  Run bioinformatics analyses using the bio_tools library. Covers sequence
  search (BLAST, MMseqs2, PyHMMER), structure prediction (AlphaFold3, Boltz2,
  Chai1, ESMFold, Protenix), sequence alignment (MAFFT, ColabFold Search),
  inverse folding (ProteinMPNN, LigandMPNN), embeddings (ESM2, ESM3),
  sequence generation (ProGen2, Evo2), and more.
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
---

# bio-tools skill

You help users run bioinformatics analyses using the `bio_tools` library.

## How to use a tool

1. Read `CLAUDE.md` → Tool Discovery section to find the right module
2. Read the tool's source to confirm Input/Config/Output classes
3. Follow the Universal Call Pattern from CLAUDE.md
4. See `analyses/examples/` for reference scripts

**Important:** Do NOT activate a virtual environment (e.g. `source .venv/bin/activate`)
before running tools. The package is already installed — just use `python3` directly.

## Script vs direct execution

Infer from context:

- **Write a script** (`analyses/{name}_{date}.py`): "write", "create", "set up",
  multi-step pipelines, GPU jobs, or when unclear
- **Execute directly** (inline via Bash): "what is", "how many", "show me",
  "quick", simple lookups

Always show the equivalent Python code either way.

## Key conventions

- Every tool uses the pattern: `Input` → `Config` → `run_{tool}()` → `Output`
- GPU tools: AlphaFold3, Boltz2, Chai1, ESMFold, Protenix, ESM2, ESM3,
  Evo2, ProGen2, Enformer, Borzoi, AlphaGenome, ProteinMPNN, LigandMPNN,
  RFDiffusion3, BioEmu — note GPU requirement in scripts
- Multi-step pipelines use `# === Step N: Description ===` headers
- All public symbols are re-exported from `bio_tools.tools`
