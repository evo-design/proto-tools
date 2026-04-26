<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/mmseqs2-homology-search"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# MMseqs2 Homology Search

## Overview

Generates [Multiple Sequence Alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSAs) for protein sequences by searching MMseqs2-indexed homology databases. The successor to `colabfold-search`: GPU-by-default, registry-driven dataset selection, and a forward-compatible grouped input shape that will support paired multimer MSAs in a future PR (tracked in [#543](https://github.com/proto-bio/proto-tools/issues/543)).

- **Tool key**: `mmseqs2-homology-search`
- **Input**: Protein sequences (singleton groups in this phase; paired groups planned)
- **Output**: Per-group `MSA` objects + paired-MSA placeholders
- **Execution**: GPU-by-default via [MMseqs2-GPU](https://www.nature.com/articles/s41592-025-02819-8); CPU fallback when `use_gpu=False`
- **Caching**: Per-item caching via `cacheable=True`
- **Datasets**: Registered in `proto_tools/tools/sequence_alignment/databases/`; this phase ships with `uniref30-2302`

## Background

MSAs capture the evolutionary history of a protein family by aligning [homologous](https://en.wikipedia.org/wiki/Homology_(biology)) sequences across organisms. Patterns of conservation and covariation across columns encode structural and functional information that downstream tools — most notably AlphaFold and its successors — exploit to predict 3D structure.

MMseqs2 ([Steinegger & Söding 2017](https://doi.org/10.1038/nbt.3988)) is the search engine: ~100× faster than BLAST at comparable sensitivity. The GPU build ([Kallenborn et al. 2024](https://www.biorxiv.org/content/10.1101/2024.11.13.623350)) brings further speedups on Turing+ NVIDIA GPUs.

This tool is the long-term replacement for `colabfold-search`. The motivation is generalization: support multiple molecule types (protein → RNA/DNA in future phases), eliminate hardcoded database paths in favor of a shared registry, and unblock paired MSAs for multimer structure prediction. See `notes/homology-search-design.md` for the full design and migration plan.

## How It Works

1. **Query collection**: Sequences are organized into *groups*. A flat sequence is a singleton group; a list of sequences is a paired group (Phase 4+).
2. **Dataset resolution**: Each `Mmseqs2HomologySearchConfig.datasets` key is looked up in `DatasetRegistry` to get its on-disk cache path, MMseqs2 flags, and capability metadata.
3. **MMseqs2 search**: For each group, the registered dataset's MMseqs2 index is searched using the [ColabFold pipeline](https://github.com/sokrypton/ColabFold) — k-mer prefilter, ungapped/gapped alignment, profile expansion, then realignment.
4. **GPU acceleration**: When `use_gpu=True` (default), the search uses the GPU-padded index (`*.idx_pad`) and runs on the configured CUDA device. The framework reflects this in `Config.devices_per_instance`.
5. **A3M output**: One A3M file per query chain. Files are loaded back into proto-tools `MSA` objects and packaged as group-level results.

## Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `queries` | `List[Mmseqs2HomologySearchQuery \| List[Mmseqs2HomologySearchQuery]]` | Query groups in input order. A flat item is a singleton (unpaired) group; a list is a paired group (rejected in this phase per [#543](https://github.com/proto-bio/proto-tools/issues/543)). |

**Flexible input formats:**

```python
# Plain strings (auto-generated IDs)
inp = Mmseqs2HomologySearchInput(queries=["MVLSPADKTN", "MKTAYIAKQR"])

# Tuples for explicit IDs
inp = Mmseqs2HomologySearchInput(queries=[("MVLSPADKTN", "hba_human"), ("MKTAYIAKQR", "design_001")])

# Explicit Query objects
inp = Mmseqs2HomologySearchInput(queries=[
    Mmseqs2HomologySearchQuery(sequence="MVLSPADKTN", sequence_id="hba_human"),
])
```

`sequence_id` defaults to `seq_<10-char-hash>` from the sequence; uniqueness is enforced globally across all groups.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `datasets` | `List[str]` | `["uniref30-2302"]` | Registered dataset keys. This phase accepts exactly one protein dataset. |
| `use_gpu` | `bool` | `True` | Use MMseqs2-GPU; falls back to CPU when set to `False`. |
| `pairing_strategy` | `Literal["greedy", "complete"]` | `"greedy"` | Reserved for paired MSAs (Phase 4+); ignored here. |
| `sensitivity` | `Optional[float]` | `None` (use registry default) | MMseqs2 `-s` override (1.0–9.0). |
| `num_threads` | `Optional[int]` | `None` (auto-detect) | CPU threads for the search subprocess. |
| `timeout` | `int` | `3600` | Subprocess timeout in seconds. |

### Local Database Setup

Datasets are not auto-downloaded. Provision the default UniRef30 entry once before running with `search_mode="local"` (the only mode in this phase):

```bash
# Resolves to $PROTO_MODEL_CACHE/databases/uniref30_2302/ — same path that
# colabfold-search uses, so a host that's already provisioned for one tool
# doesn't need a second download.
bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \
  "$PROTO_MODEL_CACHE/databases/uniref30_2302" \
  uniref30_2302 \
  colabfold_envdb_202108 \
  1   # SKIP_METAGENOMIC=1; drop this arg to also fetch the ~110 GB envdb
```

A future PR ships `proto-tools provision uniref30-2302` to wrap the same script behind the dataset registry. See `notes/homology-search-design.md` § Bulk Provisioning for the planned interface.

**Disk requirement**: ~650 GB peak / ~630 GB final for UniRef30 alone.
**Time**: ~2 hours on a fast network plus ~30 min indexing.
**GPU prerequisite**: `*.idx_pad` file (built automatically by `setup_databases.sh`); validated at config time when `use_gpu=True`.

If a dataset isn't provisioned, config validation raises with the exact `setup_databases.sh` invocation needed.

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[Mmseqs2HomologySearchResult]` | One result per input *group*, in input order. |

Each `Mmseqs2HomologySearchResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `sequence_ids` | `List[str]` | Identifiers for the chains in this group. |
| `msas` | `List[Optional[MSA]]` | Per-chain unpaired MSA. `None` when no homologs were found. |
| `paired_msas` | `List[Optional[MSA]]` | Per-chain paired MSA. Always `[None]` in this phase; populated in Phase 4. |
| `datasets_searched` | `List[str]` | Dataset registry keys actually searched for this group. |
| `num_homologs_found` | `List[int]` | Count of homologs per chain (excludes the query itself). |

Export formats: `a3m` (default), `fasta`. One file per chain per group.

## Interpreting Results

| MSA Depth | Interpretation | Impact on Structure Prediction |
|-----------|----------------|-------------------------------|
| 0 | No homologs found (`msa is None`) | Structure prediction relies on single-sequence features; expect lower confidence |
| 1–30 | Shallow MSA | Limited coevolutionary signal; predictions may be unreliable for novel folds |
| 30–100 | Moderate MSA | Reasonable signal for most protein families |
| 100–1000 | Deep MSA | Strong coevolutionary signal; high-confidence predictions expected |
| > 1000 | Very deep MSA | Excellent for structure prediction; diminishing returns above ~5000 |

`msa is None` typically means the query is highly divergent (de novo design, isolated short fragment, or a novel fold).

## Quick Start Examples

**Basic GPU search:**

```python
from proto_tools.tools.sequence_alignment.mmseqs2_homology_search import (
    Mmseqs2HomologySearchInput, Mmseqs2HomologySearchConfig, run_mmseqs2_homology_search,
)

# Plain string sugar — auto-generated sequence_id, defaults to UniRef30 + GPU.
inp = Mmseqs2HomologySearchInput(queries=["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"])
result = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig())

grp = result.results[0]
print(f"{grp.sequence_ids[0]}: {grp.num_homologs_found[0]} homologs found")
```

**Batch with explicit IDs:**

```python
inp = Mmseqs2HomologySearchInput(queries=[
    ("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK", "hba_human"),
    ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVG", "design_001"),
])
result = run_mmseqs2_homology_search(inp, Mmseqs2HomologySearchConfig())
for grp in result:
    print(f"{grp.sequence_ids[0]}: {grp.num_homologs_found[0]} homologs")
```

**CPU mode (skip GPU even when available):**

```python
cfg = Mmseqs2HomologySearchConfig(use_gpu=False)
result = run_mmseqs2_homology_search(inp, cfg)
# cfg.devices_per_instance == 0 — framework GPU accounting reflects this.
```

**Export A3M files:**

```python
result.export("/path/to/out_dir", file_format="a3m")
# Or per-chain:
grp.msas[0].to_a3m_file("/path/to/seq.a3m")
```

## Best Practices & Gotchas

- **Provision once.** This tool does not auto-download the UniRef30 DB. The first `run_mmseqs2_homology_search` against a missing dataset raises with the exact `setup_databases.sh` invocation needed. The dataset cache lives under `$PROTO_MODEL_CACHE/databases/<name>/` and is shared with `colabfold-search`.
- **GPU-by-default is intentional.** The default `use_gpu=True` matches AlphaFast and the upstream MMseqs2-GPU recommendation. Set `use_gpu=False` for CPU-only hosts.
- **`devices_per_instance` reflects actual GPU usage.** Returns `1` when `use_gpu=True`, `0` otherwise. ToolPool batching, scheduling, and resource accounting use this.
- **Paired groups are deferred.** A list-of-lists in `queries` raises a validation error pointing at [#543](https://github.com/proto-bio/proto-tools/issues/543). For now, pass each chain of a multimer as a singleton — the resulting MSAs are unpaired but still useful for many predictors.
- **Multi-dataset is deferred.** Only one dataset key in `Config.datasets` for now. UniRef30 + ColabFoldDB merge support comes with the bulk provisioning PR.
- **Dataset cache is shared with colabfold-search.** During the migration window both tools read from the same `databases/<name>/` paths. After `colabfold-search` is removed (Phase 6 of the migration plan), this tool owns the path exclusively.

## References

- Steinegger, M. & Söding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11), 1026–1028. DOI: [10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)
- Kallenborn, F., Chacon, A., Hundt, C., et al. (2024). GPU-accelerated homology search with MMseqs2. *Nature Methods*, in press. DOI: [10.1038/s41592-025-02819-8](https://doi.org/10.1038/s41592-025-02819-8)
- Mirdita, M., Schütze, K., Moriwaki, Y., et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19(6), 679–682. DOI: [10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)

## Related Tools

**Often used together:**
- **AlphaFold 3** (`alphafold3-prediction`), **Boltz-2** (`boltz2-prediction`), **Chai-1** (`chai1-prediction`), **Protenix** (`protenix-prediction`), **AlphaFold 2** (`alphafold2-prediction`) — All consume MSAs from this tool. Per-predictor migrations from `colabfold-search` to `mmseqs2-homology-search` are tracked in #581.
- **ESMFold** (`esmfold-prediction`) — Single-sequence predictor; use when MSAs are too shallow.

**Alternatives for sequence search:**
- **`colabfold-search`** — Earlier wrapper of the same backend; will be deprecated after migration completes (Phase 6 of #581). Functionally equivalent for protein-only unpaired MSAs today.
- **BLAST** (`blast-search`) — Traditional sequence search with E-value statistics; better for targeted homolog identification.
- **Foldseek** (`foldseek-search`) — Structure-based search; finds remote homologs sequence search misses.
