<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/colabfold-search"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ColabFold Search

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

> [!NOTE]
> **Deprecation in progress.** `colabfold-search` is being replaced by [`mmseqs2-homology-search`](../mmseqs2/README.md), a generalized successor that adds GPU-by-default execution, a dataset registry covering AF3/AlphaFast/Lightning-Boltz/Chai-1/Protenix protein and RNA databases, and a forward-compatible grouped input shape for paired multimer MSAs. The migration is phased: structure-prediction wrappers will switch over one at a time, and this tool will be removed once every consumer is migrated. New code should target `mmseqs2-homology-search`.

## Overview

ColabFold MSA Search generates [Multiple Sequence Alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSAs) for protein sequences by searching large sequence databases for homologs. MSAs are a foundational input for structure prediction (AlphaFold, ESMFold), co-evolutionary analysis, and conservation scoring. This tool wraps [MMSeqs2](https://mmseqs.com/) for fast local search and the ColabFold API for remote search.

- **Tool key**: `colabfold-search`
- **Input**: Protein sequences (with optional identifiers)
- **Output**: MSA objects per query sequence
- **Execution**: CPU (local venv via `ToolInstance`), optional GPU for MMSeqs2
- **Caching**: Per-item caching via `cacheable=True` on `@tool()`

## Background

Multiple Sequence Alignments capture the evolutionary history of a protein family by aligning [homologous](https://en.wikipedia.org/wiki/Homology_(biology)) sequences found across organisms. Each column in the alignment represents a structural position, and the patterns of conservation and covariation encode information about:

- **Structural constraints**: Positions buried in the protein core are highly conserved
- **Functional residues**: Active site and binding site residues show conservation
- **[Coevolution](https://en.wikipedia.org/wiki/Coevolution)**: Residue pairs that co-vary indicate spatial contacts, which is the key signal AlphaFold2 uses for structure prediction
- **Evolutionary rate**: The depth (number of homologs) of the MSA correlates with prediction confidence

ColabFold uses MMSeqs2 (Many-against-Many sequence searching) for fast homology detection. MMSeqs2 is ~100x faster than BLAST with comparable sensitivity, making it practical to search databases with billions of sequences.

## Tools

### ColabFold MSA Search (`colabfold-search`)

Generate MSAs for protein sequences using ColabFold search, with options.

for online and local execution.

Local Execution:
Searches a local MMSeqs2 database to find homologous sequences and generates
Multiple Sequence Alignments (MSAs).

Online Execution:
Uses the ColabFold online API to search for homologous sequences and generates
Multiple Sequence Alignments (MSAs). Rate limited.

## How It Works

1. **Query submission**: Protein sequences are formatted as FASTA and submitted for search
2. **Database search**: MMSeqs2 identifies homologous sequences using a three-stage cascade: k-mer matching, ungapped alignment, and gapped alignment
3. **MSA construction**: Homologous sequences are aligned to the query using the [A3M format](https://github.com/soedinglab/hh-suite/wiki#a3m-format) (a compressed alignment format where insertions relative to the query are lowercase)
4. **Result packaging**: Each query gets an MSA object containing aligned sequences, which can be exported as A3M or FASTA files

**Search modes:**
- **Remote** (default): Queries the ColabFold API server. No local database required, but subject to rate limits. Best for small batches.
- **Local**: Searches a locally installed MMSeqs2 database. **Requires manually provisioning the [UniRef30](https://www.uniprot.org/help/uniref) database** before first use — see [Local Database Setup](#local-database-setup) under Configuration. Best for large-scale or repeated searches.

## Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `queries` | `List[ColabfoldSearchQuery]` | Protein sequences to search. Accepts multiple formats (see below). |

**Flexible input formats:**

```python
# Simple: list of sequence strings
inputs = ColabfoldSearchInput(queries=["MVLSPADKTN", "MKTAYIAKQR"])

# With explicit IDs
inputs = ColabfoldSearchInput(queries=[
    ColabfoldSearchQuery(sequence="MVLSPADKTN", sequence_id="hemoglobin_alpha"),
    ColabfoldSearchQuery(sequence="MKTAYIAKQR", sequence_id="my_design"),
])

# Tuple format: (sequence, id)
inputs = ColabfoldSearchInput(queries=[
    ("MVLSPADKTN", "hemoglobin_alpha"),
    ("MKTAYIAKQR", "my_design"),
])
```

Sequence IDs are auto-generated from a hash of the sequence if not provided.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_mode` | `"local"` \| `"remote"` | `"remote"` | Search mode. Remote uses ColabFold API; local uses MMSeqs2 database. |
| `use_metagenomic_db` | `bool` | `False` | Include metagenomic sequences in search. Increases MSA depth but slows search. |
| `sensitivity` | `Optional[float]` | `None` | MMSeqs2 `-s` (1.0-9.0). Local mode only; **ignored on GPU** (colabfold_search forces ungapped prefilter under `--gpu 1`). When `None` on CPU, falls back to colabfold's k-score path (server-equivalent). |
| `database_name` | `str` | `"uniref30_2302_db"` | Database name for local search. |
| `num_threads` | `Optional[int]` | `None` (auto) | CPU threads for local search. Auto-detects available cores. |
| `output_dir` | `Optional[str]` | `None` | Output directory for MSA files. Defaults to `~/.cache/proto-language/colabfold_search`. |
| `msa_db_dir` | `Optional[str]` | `None` (resolves to `$PROTO_MODEL_CACHE/databases/uniref30_2302/`) | Path to local MMSeqs2 database directory. Local mode only. |
| `verbose` | `bool` | `False` | Print progress messages during execution. |

### Parameter Guides

**Sensitivity (local mode):**

| Value | Speed | Sensitivity | Use Case |
|-------|-------|-------------|----------|
| 1.0-4.0 | Fast | Low | Quick screening, close homologs only |
| 5.0-7.0 | Moderate | Moderate | General-purpose search |
| 7.5-8.0 | Slow | High | Default ColabFold behavior; good for structure prediction |
| 8.5-9.0 | Very slow | Maximum | Exhaustive search for remote homologs |

**Metagenomic database:**

| Setting | MSA Depth | Speed | When to Use |
|---------|-----------|-------|-------------|
| `False` | Lower | Faster | Most use cases; sufficient for well-studied protein families |
| `True` | Higher | Slower | Orphan proteins, de novo designs, or when MSA depth is critical for structure prediction |

### Local Database Setup

Local search (`search_mode="local"`) requires the UniRef30 MMSeqs2 database to be downloaded and indexed on the host. **The tool does not download databases automatically** — running `colabfold-search` against a missing local database raises a `ValueError` with a hint pointing at this script. Provision once:

```bash
# Default location: $PROTO_MODEL_CACHE/databases/uniref30_2302/
# (resolves to $PROTO_HOME/proto_model_cache/databases/uniref30_2302/ when PROTO_MODEL_CACHE is unset)
bash proto_tools/tools/sequence_alignment/colabfold_search/setup_databases.sh \
  "$PROTO_MODEL_CACHE/databases/uniref30_2302" \
  uniref30_2302 \
  colabfold_envdb_202108 \
  1   # SKIP_METAGENOMIC=1; drop this arg to also fetch the ~110 GB envdb
```

What the script does:

1. Downloads `uniref30_2302.tar.gz` (~99 GB) and the companion taxonomy tarball
2. Extracts the prebuilt MMSeqs2 database files
3. Runs `mmseqs createindex` to build the CPU-side `.idx` (~290 GB)
4. Runs `mmseqs makepaddedseqdb` to build the GPU-padded `.idx_pad` (~8 GB; requires the GPU-capable MMseqs2 binary, included in the tool's standalone env)
5. Touches a `UNIREF30_READY` marker

**Disk requirement**: ~650 GB free at peak (downloads + intermediate tar + indexed DB). Final indexed size is ~630 GB.

**Time**: ~2 hours on a fast network (download dominates; indexing is ~30 minutes).

**Custom location**: pass any path as the first argument and set `msa_db_dir` to the same path on `ColabfoldSearchConfig`. Multiple proto-tools (now and planned) read from `$PROTO_MODEL_CACHE/databases/{name}/`, so following the default keeps databases shareable.

**GPU search**: requires the `.idx_pad` file produced in step 4 above. The tool validates this at config time when `use_gpu=True`.

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[ColabfoldSearchResult]` | One result per input query, in input order |

Each `ColabfoldSearchResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `msa` | `Optional[MSA]` | The MSA object, or `None` if no homologs were found |
| `sequence_id` | `str` | Identifier for the searched sequence |
| `num_homologs_found` | `int` (property) | Number of homologs (excludes the query sequence itself) |

**MSA object properties:**

| Property | Type | Description |
|----------|------|-------------|
| `num_sequences` | `int` | Total sequences including query |
| `alignment_length` | `int` | Number of columns in the alignment |
| `sequence_ids` | `List[str]` | Identifiers for all sequences |
| `average_gap_fraction` | `float` | Mean fraction of gaps across sequences |

Export formats: `a3m`, `fasta`

## Interpreting Results

| MSA Depth (homologs) | Interpretation | Impact on Structure Prediction |
|----------------------|---------------|-------------------------------|
| 0 | No homologs found | Structure prediction will rely solely on single-sequence features; expect lower confidence |
| 1-30 | Shallow MSA | Limited coevolutionary signal; predictions may be unreliable for novel folds |
| 30-100 | Moderate MSA | Reasonable signal for most protein families |
| 100-1000 | Deep MSA | Strong coevolutionary signal; high-confidence predictions expected |
| > 1000 | Very deep MSA | Excellent for structure prediction; diminishing returns above ~5000 |

When `msa` is `None`, the search found no homologs beyond the query itself. This is common for:
- Very short sequences (< 30 residues)
- De novo designed proteins with no natural homologs
- Highly divergent or novel protein folds

## Quick Start Examples

**Remote search (simplest):**
```python
from proto_tools.tools.sequence_alignment.colabfold_search import (
    ColabfoldSearchInput, ColabfoldSearchConfig, run_colabfold_search,
)

inputs = ColabfoldSearchInput(queries=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"])
config = ColabfoldSearchConfig(search_mode="remote")
result = run_colabfold_search(inputs, config)

msa = result.results[0].msa
if msa is not None:
    print(f"Found {result.results[0].num_homologs_found} homologs")
    print(f"Alignment: {msa.num_sequences} sequences x {msa.alignment_length} columns")
else:
    print("No homologs found")
```

**Batch search with custom IDs:**
```python
from proto_tools.tools.sequence_alignment.colabfold_search import (
    ColabfoldSearchInput, ColabfoldSearchConfig, ColabfoldSearchQuery,
    run_colabfold_search,
)

queries = [
    ColabfoldSearchQuery(sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK", sequence_id="hba_human"),
    ColabfoldSearchQuery(sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVG", sequence_id="design_001"),
]

inputs = ColabfoldSearchInput(queries=queries)
config = ColabfoldSearchConfig(search_mode="remote", use_metagenomic_db=True)
result = run_colabfold_search(inputs, config)

for res in result.results:
    print(f"{res.sequence_id}: {res.num_homologs_found} homologs")
```

**Local search with custom database:**
```python
inputs = ColabfoldSearchInput(queries=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"])
config = ColabfoldSearchConfig(
    search_mode="local",
    msa_db_dir="/path/to/colabfold/databases",
    database_name="uniref30_2302_db",
    sensitivity=8.0,
    num_threads=16,
)
result = run_colabfold_search(inputs, config)
```

**Export MSAs to files:**
```python
# Export all MSAs as A3M files
result.export("/path/to/output_dir", file_format="a3m")

# Or access individual MSA objects
msa = result.results[0].msa
msa.to_a3m_file("/path/to/output.a3m")
msa.to_fasta_file("/path/to/output.fasta")
```

**Analyze MSA conservation:**
```python
msa = result.results[0].msa
if msa is not None:
    for pos in range(min(10, msa.alignment_length)):
        conservation = msa.get_conservation(pos)
        freqs = msa.get_position_frequencies(pos)
        top_aa = max(freqs, key=freqs.get)
        print(f"Position {pos}: conservation={conservation:.2f}, top={top_aa} ({freqs[top_aa]:.2f})")
```

## Best Practices & Gotchas

- **Remote mode is rate-limited.** The ColabFold API server has usage limits. For large batches (>50 sequences), use local mode with a downloaded database.
- **Results are cached per sequence.** The `cacheable=True` flag on `@tool()` enables per-item caching, so re-running with the same sequences skips the search. This works even if you add new sequences to the batch.
- **MSA is `None` when no homologs are found.** Always check `result.msa is not None` before accessing MSA properties. The `num_homologs_found` property safely returns 0 in this case.
- **Sequence IDs must be unique.** If you provide duplicate `sequence_id` values, validation will fail. Auto-generated IDs are derived from sequence hashes and are guaranteed unique for distinct sequences.
- **Local databases require manual setup.** Local mode does not auto-download. See [Local Database Setup](#local-database-setup) for the one-time provisioning script; UniRef30 is ~99 GB to download and ~630 GB after indexing.
- **A3M vs FASTA format.** A3M files use lowercase letters for insertions relative to the query, making them more compact. FASTA files pad with gaps for a rectangular alignment. Most structure predictors accept A3M directly.
- **MSA depth matters for structure prediction.** If your MSA is shallow (<30 sequences), consider enabling `use_metagenomic_db=True` or increasing `sensitivity` to find more distant homologs.
- **Escape hatch for niche colabfold flags.** Local mode exposes `extra_args: list[str]` for verbatim `colabfold_search` CLI tokens not surfaced as typed fields (e.g. `["--max-accept", "500"]`, `["--qsc", "0.0"]`). Tokens are appended after the typed flags. Remote mode does not honor `extra_args` since it goes through the API.

## References

- Mirdita, M., Schutze, K., Moriwaki, Y., et al. (2022). ColabFold: making protein folding accessible to all. *Nature Methods*, 19(6), 679-682. DOI: [10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- Steinegger, M. & Soding, J. (2017). MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. *Nature Biotechnology*, 35(11), 1026-1028. DOI: [10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)
- Mirdita, M., von den Driesch, L., Galiez, C., et al. (2017). Uniclust databases of clustered protein sequences and alignments. *Nucleic Acids Research*, 45(D1), D206-D215. DOI: [10.1093/nar/gkw1081](https://doi.org/10.1093/nar/gkw1081)

## Related Tools

**Often used together:**
- **AlphaFold** (`alphafold3-prediction`) -- MSAs from ColabFold are a primary input for AlphaFold structure prediction
- **ESMFold** (`esmfold-prediction`) -- Single-sequence structure predictor; use when MSAs are too shallow
- **Conservation scoring** -- Use MSA column statistics to identify conserved positions in designed proteins

**Alternatives for sequence search:**
- **BLAST** (`blast-search`) -- Traditional sequence search with E-value statistics; better for targeted homolog identification
- **Foldseek** (`foldseek-search`) -- Structure-based search; finds remote homologs that sequence search misses
- **HHblits** -- Iterative profile-based search; higher sensitivity for remote homologs but slower
