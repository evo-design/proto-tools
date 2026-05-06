<a href="https://bio-pro.mintlify.app/tools/structure-alignment/foldmason"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# FoldMason Toolkit

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

The FoldMason toolkit wraps the Steinegger Lab's multiple structure alignment tool (Gilchrist et al., *Science* 2026). Two sibling tools, both backed by the same FoldMason binary:

| Tool key | Operation | Modes |
|---|---|---|
| `foldmason-msa` | Multiple structure alignment over ≥2 PDB structures | remote (server) + local (CLI) |
| `foldmason-score-msa` | Score an existing MSA with average + per-column LDDT | local only |

`foldmason-msa` is the canonical "align my list of structures" tool — it builds a structural MSA and a guide tree from raw PDB inputs. `foldmason-score-msa` takes a precomputed MSA and structures, and returns LDDT scores you can use as a quality metric or downstream design constraint.

## Background

**What does this toolkit measure?**
FoldMason aligns multiple protein structures by encoding each one as a sequence over the 3Di structural alphabet (the same alphabet used by Foldseek), then running progressive multiple-sequence alignment with structure-aware scoring. The result is an alignment that captures structural homology even between proteins below the sequence-search "twilight zone." LDDT (Local Distance Difference Test) at the column level quantifies how well the structures superimpose at each aligned position.

**Why is this important?**
- **Structural family analysis:** align a set of homologs to discover which residues are structurally conserved vs. variable.
- **Design-set quality:** score an MSA of designed structures to surface positions where the design diverges from the intended scaffold.
- **Template ensembles:** prepare a multi-structure template for downstream protein design or template-based modelling.

**Scientific foundation:**
FoldMason combines Foldseek's 3Di alphabet with a progressive multiple-alignment engine (`structuremsa`), augmented by neighborhood-aware scoring and optional iterative refinement. `msa2lddt` measures alignment quality column-by-column using LDDT, giving a fast, consistent metric across structurally diverse inputs.

## Tools

### FoldMason MSA (`foldmason-msa`)

Run a FoldMason multiple structure alignment.

Dispatches to the public FoldMason server (remote) or the local FoldMason
CLI's `easy-msa` based on `config.search_mode`.

### FoldMason Score MSA (`foldmason-score-msa`)

Score a structural MSA with FoldMason msa2lddt.

## How It Works

**Remote mode (`foldmason-msa` with `search_mode="remote"`):**
1. POST the list of PDB structures to `search.foldseek.com/api/ticket/foldmason` (multipart/form-data with `fileNames[]` + `queries[]`, one entry per structure).
2. Poll `/api/ticket/{id}` every `poll_interval_seconds` until status reaches `COMPLETE` (or `ERROR`, or timeout).
3. Fetch the JSON result from `/api/result/foldmason/{id}` — contains per-row aligned AA + 3Di sequences plus the Newick guide tree.
4. Reassemble the AA and 3Di alignments as standard FASTA strings and return.

**Local mode (`foldmason-msa` with `search_mode="local"`, `foldmason-score-msa`):**
1. Provision the FoldMason binary via the standalone env (`standalone/setup.sh` calls `proto_tools/utils/install_binary.py foldmason`, which downloads the platform-specific tarball from `mmseqs.com/foldmason` and extracts the binary into the venv's `bin/` directory).
2. Write inputs to a temp dir, invoke `foldmason easy-msa` (or `foldmason createdb` + `msa2lddt`) via `ToolInstance.dispatch`, parse the output files / stdout.

**Key assumptions:**
- Inputs are PDB-format text (mmCIF works locally but the remote server parses PDB).
- Remote mode requires network reachability of `search.foldseek.com` (anonymous access).
- Local modes require the standalone env to have been provisioned (`setup.sh`).

**Limitations:**
- The remote server uses fixed alignment parameters; tune `gap_open`/`gap_extend`/`refine_iters` only in local mode.
- `foldmason-score-msa` is local-only — the server does not expose `msa2lddt`.
- For chained `foldmason-msa` → `foldmason-score-msa`, the FASTA record headers in the MSA must match `structure_ids` so `msa2lddt` can resolve each row to its structure.

**Computational requirements:**
- **Remote modes:** HTTP-only on the wrapper side; CPU only. ~5-30s for typical small inputs against the public server.
- **Local modes:** local CPU; alignment time scales with N² × L (N structures, L mean length). The FoldMason binary is ~50 MB.

## Input Parameters

### `FoldmasonMSAInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structures` | `list[str]` | *required* (≥2) | PDB-format text strings to align. |
| `structure_ids` | `list[str] \| None` | `None` | Optional IDs per structure (default: `'structure_0'`, ...). Length must match `structures`. IDs become FASTA record headers and Newick leaf labels. |

### `FoldmasonScoreMSAInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structures` | `list[str]` | *required* (≥2) | PDB-format text strings whose order matches the rows of `aa_msa_fasta`. |
| `structure_ids` | `list[str] \| None` | `None` | Optional IDs per structure; must match the FASTA record headers in `aa_msa_fasta`. |
| `aa_msa_fasta` | `str` | *required* | Amino-acid MSA in FASTA format (typically the output of `foldmason-msa`). |

## Configuration

### `FoldmasonMSAConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_mode` | `Literal["remote", "local"]` | `"remote"` | `remote` hits the public server; `local` runs the FoldMason CLI. |
| `poll_interval_seconds` | `float` | `5.0` | (advanced, remote-only) Delay between status polls. |
| `timeout_seconds` | `float` | `600.0` | (advanced, remote-only) Max wall-clock time for the alignment. |
| `gap_open` | `int` | `25` | (advanced, local-only) Gap open cost. |
| `gap_extend` | `int` | `2` | (advanced, local-only) Gap extension cost. |
| `refine_iters` | `int` | `0` | (advanced, local-only) Number of alignment-refinement iterations. |
| `precluster` | `bool` | `False` | (advanced, local-only) Pre-cluster structures before MSA construction. Recommended for large datasets (>1k structures). |
| `guide_tree_newick` | `str \| None` | `None` | (advanced, local-only) Newick guide tree to use instead of computing one; leaf labels must match `structure_ids`. |
| `num_threads` | `int` | `4` | (advanced, local-only) CPU threads. |

### `FoldmasonScoreMSAConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pair_threshold` | `float` | `0.0` | (advanced) Minimum fraction of pair sub-alignments with LDDT info to score a column (0-1). |
| `only_scoring_cols` | `bool` | `False` | (advanced) Normalise average LDDT by scoring-column count rather than alignment length. |
| `guide_tree_newick` | `str \| None` | `None` | (advanced) Newick guide tree to score against; leaf labels must match `structure_ids`. |
| `num_threads` | `int` | `4` | (advanced) CPU threads. |

## Output Specification

```python
# foldmason-msa
FoldmasonMSAOutput(
    ticket_id: str,                # Server ticket ID; "" in local mode
    aa_msa_fasta: str,             # Amino-acid MSA in FASTA format
    three_di_msa_fasta: str,       # 3Di-alphabet MSA in FASTA format
    newick_tree: str,              # Newick guide tree
    num_sequences: int,
    alignment_length: int,         # Number of MSA columns
    result_url: str,               # Server result URL; "" in local mode
)

# foldmason-score-msa
FoldmasonScoreMSAOutput(
    average_lddt: float,           # Average MSA LDDT score (0-1)
    columns_considered: int,       # Number of columns that were scored
    alignment_length: int,         # Total MSA columns
    column_scores: list[float],    # Per-column LDDT scores
)
```

**Supported export formats:** `json`

## Interpreting Results

**LDDT score interpretation:**
- `average_lddt > 0.7`: structurally consistent alignment; columns superimpose well.
- `0.4 < average_lddt < 0.7`: moderate alignment quality; some divergent regions.
- `average_lddt < 0.4`: weak structural support; treat as exploratory.

**Column scores:**
Per-column LDDT highlights which alignment positions are structurally conserved (high score) vs. divergent (low score). Use this to identify variable loops in an otherwise conserved fold, or to mask out unreliable regions before downstream analysis.

**Newick tree:**
The guide tree shows FoldMason's structural-similarity-based grouping. Useful for visualizing relationships when aligning ≥3 structures.

## Quick Start Examples

**Example 1: Remote multiple structure alignment.**

```python
from proto_tools.tools.structure_alignment import (
    FoldmasonMSAConfig, FoldmasonMSAInput, run_foldmason_msa,
)
import requests

structures = [
    requests.get(f"https://files.rcsb.org/download/{pdb}.pdb", timeout=30).text
    for pdb in ("1TIM", "8TIM", "1TPF")
]

output = run_foldmason_msa(
    FoldmasonMSAInput(structures=structures, structure_ids=["chicken_TIM", "trypano_TIM", "yeast_TIM"]),
    FoldmasonMSAConfig(),
)
print(f"Aligned {output.num_sequences} structures across {output.alignment_length} columns")
print(f"Tree: {output.newick_tree.strip()}")
```

**Example 2: Local alignment with custom gap penalties.**

```python
output = run_foldmason_msa(
    FoldmasonMSAInput(structures=designs, structure_ids=design_names),
    FoldmasonMSAConfig(search_mode="local", gap_open=15, refine_iters=3, num_threads=8),
)
```

**Example 3: Chained MSA → LDDT scoring.**

```python
from proto_tools.tools.structure_alignment import (
    FoldmasonMSAConfig, FoldmasonMSAInput, run_foldmason_msa,
    FoldmasonScoreMSAConfig, FoldmasonScoreMSAInput, run_foldmason_score_msa,
)

ids = ["a", "b", "c"]
msa_out = run_foldmason_msa(
    FoldmasonMSAInput(structures=structures, structure_ids=ids),
    FoldmasonMSAConfig(search_mode="local"),
)

score = run_foldmason_score_msa(
    FoldmasonScoreMSAInput(structures=structures, structure_ids=ids, aa_msa_fasta=msa_out.aa_msa_fasta),
    FoldmasonScoreMSAConfig(),
)
print(f"average LDDT: {score.average_lddt:.3f}")
```

## Best Practices & Gotchas

1. **IDs must round-trip cleanly between MSA and score-MSA.** When chaining `foldmason-msa` → `foldmason-score-msa`, make sure `structure_ids` is the same in both calls — `msa2lddt` resolves rows by FASTA header, and a mismatch silently produces wrong scores.
2. **The remote server deduplicates by structure content.** Re-submitting the same set of structures returns the same ticket ID and the *original* entry names — user-supplied `structure_ids` are dropped on the server-side cache hit. If you need IDs preserved, use `search_mode="local"`.
3. **Local mode is the only path for tuned alignments.** The remote server uses fixed defaults; if you need custom `gap_open`/`gap_extend`/`refine_iters`, run locally.
4. **`refine_iters` costs time linearly.** Each refinement iteration re-aligns every sequence against the current consensus; default 0 is fine for most use cases.
5. **Cache responsibly.** Both tools are `cacheable=True`; subsequent calls with the same inputs + config skip the work. Polling parameters and threads are correctly excluded from the cache key.

## Related Tools

- [`foldseek-search`](../foldseek/README.md) — pairwise structural search; useful upstream for finding homologs to align with FoldMason.
- [`tmalign`](../tmalign/README.md) — pairwise structural alignment; complementary for two-structure cases.
- [`mafft`](../../sequence_alignment/mafft/README.md) — sequence-based MSA; FoldMason replaces this for structure-aware alignment.

## References

- [FoldMason paper](https://doi.org/10.1126/science.ads6733) — Gilchrist et al., *Science* 2026
- [FoldMason GitHub](https://github.com/steineggerlab/foldmason)
- [FoldMason web server](https://search.foldseek.com/foldmason)
