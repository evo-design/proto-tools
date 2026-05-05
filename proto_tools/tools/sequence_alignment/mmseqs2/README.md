<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/mmseqs2"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# MMseqs2

## Overview

[MMseqs2](https://mmseqs.com/) (Many-against-Many sequence searching) is an ultra-fast tool for searching and clustering huge protein and nucleotide sequence sets. It performs BLAST-like searches at ~100× the speed while maintaining comparable sensitivity. The GPU build ([Kallenborn et al. 2024](https://doi.org/10.1038/s41592-025-02819-8)) brings further speedups on Turing+ NVIDIA GPUs.

This module exposes four registered tools, all backed by a single MMseqs2 install plus the ColabFold pipeline. Pick by what you need:

- **Free-form FASTA / pre-built DB?** → `mmseqs2-search-proteins` or `-search-genomes`. You provide the database path; output is per-query hits as Pydantic models / DataFrames. Default to CPU; flip `use_gpu=True` if you've built a `*.idx_pad` index.
- **Need an MSA for structure prediction?** → `mmseqs2-homology-search`. Database is registry-driven (`uniref30-2302`, etc., provisioned via `setup_databases.py`); output is `MSA` objects (one per chain). GPU by default; the ColabFold iterative pipeline runs internally.
- **Reduce redundancy in a sequence set?** → `mmseqs2-clustering`. CPU only; output is per-sequence cluster assignments.

## Tool Catalog

| Tool | Operation | Devices | Use Case |
|------|-----------|---------|----------|
| `mmseqs2-search-proteins` | `mmseqs easy-search` against a user-supplied protein DB | CPU by default; opt-in `use_gpu=True` (requires `*.idx_pad`) | Ad-hoc protein-vs-database similarity search |
| `mmseqs2-search-genomes` | Full nucleotide `createdb` + `search` + `convertalis` workflow | CPU | Genome-to-genome nucleotide comparisons |
| `mmseqs2-clustering` | `mmseqs cluster` greedy set-cover | CPU | Group similar sequences and extract representatives |
| `mmseqs2-homology-search` | ColabFold-style iterative MSA pipeline against a registry-provisioned DB | GPU by default (Turing+ Linux); CPU fallback | MSA generation feeding structure predictors (AF3, Boltz-2, Chai-1, Protenix, AF2, AlphaFast) |

The four tools share one toolkit (`mmseqs2`) and one standalone env (`mmseqs2_env`). One persistent worker can serve all four operations.

## Background

MMseqs2 ([Steinegger & Söding 2017](https://doi.org/10.1038/nbt.3988)) is the search engine: ~100× faster than BLAST at comparable sensitivity. It uses a cascaded prefilter-align approach to handle massive sequence sets:

1. **K-mer matching** — fast index lookup over short amino acid / nucleotide words.
2. **Ungapped prefilter** — score k-mer matches with ungapped alignment; keep the top hits.
3. **Gapped (Smith–Waterman) alignment** — sensitive local alignment on the filtered set.
4. **Clustering** — greedy [set-cover](https://en.wikipedia.org/wiki/Set_cover_problem) over the alignment graph.

This dramatically reduces search space while preserving sensitivity comparable to BLAST.

The GPU build ([Kallenborn et al. 2024](https://doi.org/10.1038/s41592-025-02819-8)) accelerates the prefilter and alignment stages on Turing+ NVIDIA GPUs. The ColabFold iterative pipeline ([Mirdita et al. 2022](https://doi.org/10.1038/s41592-022-01488-1)) layered on top adds profile-based re-search and forms the basis of `mmseqs2-homology-search`.

## How It Works

Each of the four tools dispatches to the same standalone env via a single persistent worker, but routes to a different MMseqs2 invocation:

- **`mmseqs2-search-proteins`** wraps `mmseqs easy-search`. Inputs: protein sequences (or FASTA path) and a target DB; output: per-query hits with target ID, percent identity, and E-value. With `use_gpu=True`, the dispatch validates that a GPU-padded sibling DB (`*.idx_pad`, built via `mmseqs makepaddedseqdb`) exists, swaps it in as the easy-search target, and adds `--gpu 1`.
- **`mmseqs2-search-genomes`** runs the full nucleotide pipeline: `createdb` for query and target, `createindex`, `search`, then `convertalis` to m8 format. CPU-only.
- **`mmseqs2-clustering`** runs `mmseqs cluster` with greedy set-cover, then `createsubdb` + `createtsv` to extract per-sequence cluster assignments. CPU-only.
- **`mmseqs2-homology-search`** invokes the upstream `colabfold_search` CLI, which iterates `mmseqs search` against UniRef30 (or other ColabFold-format DBs), then merges into A3M files. The standalone env ships `colabfold[alphafold]` for this entry point. GPU-by-default.

The standalone env ships the GPU-capable MMseqs2 binary (`mmseqs-linux-gpu.tar.gz`) as canonical: it's a strict superset of the CPU build, runs CPU subcommands without `--gpu`, and only loads NVIDIA drivers lazily.

## Quick Start Examples

### Protein search (ad-hoc)

```python
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2SearchProteinsConfig,
    Mmseqs2SearchProteinsInput,
    run_mmseqs2_search_proteins,
)

inputs = Mmseqs2SearchProteinsInput(
    query_sequences=["MSKGEELFTGVVPIL", "MVSKGEELFTGVVPI"],
    mmseqs_db="reference_proteins.faa",
)
result = run_mmseqs2_search_proteins(inputs, Mmseqs2SearchProteinsConfig())
print(result.total_hits)
```

GPU mode (when the target DB has a `*.idx_pad` index built via `mmseqs makepaddedseqdb`):

```python
config = Mmseqs2SearchProteinsConfig(use_gpu=True)
result = run_mmseqs2_search_proteins(inputs, config)
```

### Genome-to-genome nucleotide search

```python
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2SearchGenomesConfig,
    Mmseqs2SearchGenomesInput,
    run_mmseqs2_search_genomes,
)

inputs = Mmseqs2SearchGenomesInput(
    query_genomes=["ATGCGT...", "ACCGGG..."],
    target_genomes=["ATGCGT...", "CCCCGGGG..."],
)
result = run_mmseqs2_search_genomes(inputs, Mmseqs2SearchGenomesConfig(sensitivity=7.5))
print(result.total_hits)
```

### Sequence clustering

```python
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2ClusteringConfig,
    Mmseqs2ClusteringInput,
    run_mmseqs2_clustering,
)

inputs = Mmseqs2ClusteringInput(input_sequences=["MVLSPADKTN...", "MVLSPADKTN...", "MKLLVVAAAA..."])
result = run_mmseqs2_clustering(inputs, Mmseqs2ClusteringConfig(min_seq_id=0.95))
print(result.num_clusters)
```

### Homology search (MSA for structure prediction)

```python
from proto_tools.tools.sequence_alignment.mmseqs2 import (
    Mmseqs2HomologySearchConfig,
    Mmseqs2HomologySearchInput,
    run_mmseqs2_homology_search,
)

inp = Mmseqs2HomologySearchInput(queries=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"])
result = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig())
grp = result.results[0]
print(f"{grp.sequence_ids[0]}: {grp.num_homologs_found[0]} homologs found")
```

CPU fallback (skip GPU even when available):

```python
cfg = Mmseqs2HomologySearchConfig(use_gpu=False)
result = run_mmseqs2_homology_search(inp, cfg)
```

## Configuration

**Database provisioning (homology search only).** `mmseqs2-homology-search` does not auto-download databases. Provision UniRef30 once before running:

```bash
python -m proto_tools.tools.sequence_alignment.mmseqs2.setup_databases uniref30-2302
```

See `setup_databases.py --list` for available datasets and presets matching predictor preferences. The dataset cache lives under `$PROTO_MODEL_CACHE/databases/<name>/` and is shared with `colabfold-search` during the migration window.

**GPU prerequisite (`mmseqs2-search-proteins`).** Opting into `use_gpu=True` requires the target DB to have a sibling `*.idx_pad` GPU-padded index. Build it once with `mmseqs makepaddedseqdb <db> <db>.idx_pad`. The tool validates at call time and raises with the exact remediation if missing.

## Important Parameters

**Sensitivity (`-s` / `sensitivity`):**

| Value | Speed | Use Case |
|-------|-------|----------|
| `1.0` | Fastest | Near-identical matches only |
| `4.0` | Balanced | `mmseqs2-clustering` default (matches upstream `easy-cluster`) |
| `5.7` | Standard | `mmseqs2-search-proteins` default (matches upstream `easy-search`) |
| `7.5` | Slower | `mmseqs2-search-genomes` default (wrapper bias for nucleotide search; upstream = 5.7) |

GPU mode (`use_gpu=True` on `mmseqs2-search-proteins`) honors `-s` normally — the binary does not override sensitivity under `--gpu 1`.

**Clustering identity (`min_seq_id`):**

| Value | Use Case |
|-------|----------|
| `0.95` | Remove near-duplicates (sequencing errors, isoforms) |
| `0.60` | Group proteins into functional families (`mmseqs2-clustering` wrapper default; upstream MMseqs2 = `0.0`) |
| `0.30` | Remote homologs |

## Best Practices & Gotchas

- **Database vs FASTA:** `mmseqs2-search-proteins` accepts either a FASTA file or a pre-built MMseqs2 DB. Pre-build for repeated searches.
- **Memory:** if `mmseqs easy-search` runs out of memory, raise `split` (e.g. `split=1`).
- **Mismatched sequence types:** don't search proteins against a nucleotide DB or vice versa. `search_type=3` for nucleotide-only.
- **Top-hits filter:** `mmseqs2-search-proteins` defaults to `only_top_hits=True`; flip it off if you need all hits per query.
- **Clustering representatives:** the representative is *not* necessarily the "best" sequence — it's the first one to cover the cluster during greedy set-cover.
- **GPU prerequisite:** `mmseqs2-search-proteins` opt-in `use_gpu=True` requires a `*.idx_pad` index alongside the DB; build with `mmseqs makepaddedseqdb`. Validation runs at call time and raises with the exact command if missing.
- **Multi-dataset for homology search:** Phase 3 supports exactly one dataset per call. UniRef30 + ColabFoldDB merge support is planned.
- **Escape hatch for niche flags:** every search/cluster tool exposes `extra_args: list[str]` for verbatim mmseqs CLI tokens not exposed as typed fields (e.g. `extra_args=["--alignment-mode", "3"]`). Tokens go through to `mmseqs easy-search` / `mmseqs search` / `mmseqs cluster` after the typed flags. `mmseqs2-homology-search` instead routes registry-driven `extra_args` via the dataset entry's `mmseqs_flags`.

## References

- Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11), 1026–1028. DOI: [10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)
- Kallenborn, F., Chacon, A., Hundt, C., et al. (2024). GPU-accelerated homology search with MMseqs2. *Nature Methods*, in press. DOI: [10.1038/s41592-025-02819-8](https://doi.org/10.1038/s41592-025-02819-8)
- Mirdita, M., Schütze, K., Moriwaki, Y., et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19(6), 679–682. DOI: [10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)

## Related Tools

- **`blast-search`**: smaller-scale searches with NCBI database access; lower throughput than MMseqs2 but a more familiar E-value-driven workflow.
- **`pyhmmer-hmmsearch` / `pyhmmer-hmmscan`**: profile-based search for remote homologs (<30% identity) when MMseqs2 sensitivity isn't enough.
- **`colabfold-search`**: the original wrapper of the homology-search backend; will be deprecated after structure-predictor migrations to `mmseqs2-homology-search` complete (Phase 6 of #581).
- **`foldseek-search`** / **`foldseek-cluster`** / **`foldseek-multimer-search`** / **`foldseek-multimercluster`** / **`foldseek-rbh`**: structure-based search and clustering for remote homologs the sequence layer misses.
