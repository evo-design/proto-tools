# MMseqs2

## Overview
MMseqs2 (Many-against-Many sequence searching) is an ultra-fast tool for searching and clustering huge protein and nucleotide sequence sets. It performs BLAST-like searches 100x faster while maintaining similar sensitivity, making it ideal for large-scale sequence analysis. This module provides interfaces for *Protein Search*, *Genome Search*, and *Sequence Clustering* operations using MMseqs2.

## When to Use This Tool

**Primary use cases:**
- Large-scale sequence similarity searches (millions of sequences)
- Clustering protein families to reduce redundancy
- Fast homology detection in metagenomics pipelines
- Building and searching custom sequence databases
- All-vs-all sequence comparisons

**When to use MMseqs2 vs BLAST:**
- MMseqs2: Best for high-throughput pipelines, large databases (>10,000 sequences), clustering, or when speed is critical.
- BLAST: Best for small-scale queries where you need access to NCBI's curated databases online, or when maximum sensitivity is required for very distant homologs.

**When NOT to use this tool:**
- Single Sequence Queries: If you're searching one sequence against NCBI, use `blast-search` instead.
- Very Short Sequences: MMseqs2 may struggle with sequences <30 amino acids or <100 nucleotides.
- Maximum Sensitivity Required: For detecting extremely distant homologs (<20% identity), profile-based tools like HMMER or HHblits are more appropriate.

## Biological Background

**What does this tool do?**
MMseqs2 finds regions of similarity between biological sequences by using a two-stage cascaded search algorithm. It first uses k-mer matching to quickly identify candidate sequences, then performs sensitive alignment on the filtered set.

**Why is this important?**
Modern sequencing generates datasets too large for traditional BLAST. MMseqs2 enables:
- Functional annotation: Assign function to millions of predicted genes from metagenomes.
- Database deduplication: Cluster redundant sequences to create non-redundant databases.
- Comparative genomics: Find homologous regions across many genomes simultaneously.
- Protein family classification: Group sequences into evolutionary families at scale.

**Scientific foundation:**
MMseqs2 uses a prefilter-align approach:

1. **K-mer Matching**: Extracts short amino acid/nucleotide words and uses a fast k-mer index to find candidate matches.
2. **Ungapped Prefiltering**: Scores candidate matches with ungapped alignment, keeping only promising hits.
3. **Gapped Alignment**: Performs Smith-Waterman-like local alignment on the filtered set.
4. **Clustering**: Uses a greedy set-cover algorithm to group sequences by similarity thresholds.

This cascade dramatically reduces the search space while preserving sensitivity comparable to BLAST.

## Tool Catalog

| Tool | Input | Target | Use Case |
|------|-------|--------|----------|
| `mmseqs-search-proteins` | Protein sequences or FASTA | Protein database or FASTA | Fast protein-vs-database similarity search |
| `mmseqs-search-genomes` | Nucleotide sequences or FASTA | Nucleotide sequences or FASTA | Genome-to-genome nucleotide comparisons |
| `mmseqs-clustering` | Sequences or FASTA | N/A | Group similar sequences and extract representatives |

## How It Works

**Method overview:**
The module exposes three distinct functions:

`mmseqs-search-proteins`: Wraps `mmseqs easy-search` for protein-vs-database searches. Handles FASTA input, database formatting, searching, and result parsing automatically.

`mmseqs-search-genomes`: Implements a full nucleotide search workflow using `mmseqs createdb`, `createindex`, `search`, and `convertalis` for genome-to-genome comparisons.

`mmseqs-clustering`: Wraps `mmseqs cluster` to group similar sequences and optionally extract representative sequences for each cluster.

**The Search Type Matrix:**
For `mmseqs-search-genomes`, the `search_type` parameter controls the search mode:
| search_type | Description | Use Case |
|-------------|-------------|----------|
| `0` | Auto-detect | Let MMseqs2 decide based on input |
| `1` | Amino acid | Protein vs protein |
| `2` | Translated | Nucleotide query translated vs protein DB |
| `3` | Nucleotide | Nucleotide vs nucleotide (default for genomes) |

**Computational Requirements:**

Speed: 100-1000x faster than BLAST for large-scale searches.

CPU: Scales well with multiple threads (default: 96 threads).

RAM: Depends on database size; uses memory-mapped files for efficiency.

Storage: Databases are typically 2-3x the size of the input FASTA.

## Input Parameters

| Tool | Parameter | Type | Description |
|------|-----------|------|-------------|
| `mmseqs-search-proteins` | `query_sequences` | `str` or `List[str]` | Path to FASTA file OR list of protein sequence strings. |
| `mmseqs-search-proteins` | `mmseqs_db` | `str` | Path to target database (FASTA or pre-built MMseqs2 DB). |
| `mmseqs-search-genomes` | `query_genomes` | `str` or `List[str]` | Path to FASTA file OR list of nucleotide sequences. |
| `mmseqs-search-genomes` | `target_genomes` | `str` or `List[str]` | Path to FASTA file OR list of nucleotide sequences. |
| `mmseqs-clustering` | `input_sequences` | `str` or `List[str]` | Path to FASTA file OR list of sequences to cluster. |

## Configuration

**Protein Search (`mmseqs-search-proteins`)**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_dir` | `str` | *Required* | Directory for storing intermediate and final results. |
| `sensitivity` | `float` | `4.0` | Search sensitivity (1.0=fastest, 7.5=most sensitive). |
| `threads` | `int` | `96` | Number of CPU threads for parallel processing. |
| `split` | `int` | `0` | Memory mode (0=auto, higher values use less memory). |
| `only_top_hits` | `bool` | `True` | If True, keep only the best hit per query. |

**Genome Search (`mmseqs-search-genomes`)**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out_dir` | `str` | *Required* | Output directory for databases and results. |
| `sensitivity` | `float` | `7.5` | Search sensitivity (higher for genome searches). |
| `search_type` | `int` | `3` | Search type (3=nucleotide vs nucleotide). |
| `threads` | `int` | `96` | Number of CPU threads. |
| `results_filename` | `str` | `mmseqs_results.m8` | Name for the output .m8 file. |

**Clustering (`mmseqs-clustering`)**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | *Required* | Directory for clustering results. |
| `min_seq_id` | `float` | `0.60` | Minimum sequence identity threshold (0.0-1.0). |
| `extract_representatives` | `bool` | `True` | If True, extract representative sequences to FASTA. |

### Parameter Guides

**Sensitivity tuning:**
| Sensitivity | Speed | Use Case |
|-------------|-------|----------|
| `1.0` | Fastest | Near-identical matches only |
| `4.0` | Default | Good balance for most protein searches |
| `7.5` | Slowest | Maximum sensitivity, genome searches, distant homologs |

**Clustering identity thresholds:**
| min_seq_id | Description | Use Case |
|------------|-------------|----------|
| `0.95` | Near-identical | Removing duplicates, sequencing errors, isoforms |
| `0.60` | Same family | Grouping proteins into functional families (default) |
| `0.30` | Remote homologs | Very loose clustering for distant relationships |

### Sweep Priorities

1. `sensitivity` — Most impactful for search quality; higher values find more distant hits at the cost of speed
2. `min_seq_id` (clustering) — Directly controls cluster granularity
3. `only_top_hits` — Set to `False` when multiple hits per query are needed for downstream analysis

## Output Specification

All tools return an `MmseqsOutput` object with two attributes:
- `results_df`: A Pandas DataFrame with search/clustering results
- `num_hits`: Total number of hits or clusters found

**Search DataFrame Columns (Protein & Genome Search):**

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `query` | Query sequence ID | The ID of your input sequence. |
| `target` | Target sequence ID | The ID of the match in the database. |
| `pident` | Percent Identity | % of exact matches in the alignment. |
| `evalue` | Expect Value | Number of hits expected by chance. Lower is better. |

**Clustering DataFrame Columns:**

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `representative` | Cluster representative ID | The sequence chosen to represent this cluster. |
| `member` | Cluster member ID | A sequence belonging to this cluster. |

When `extract_representatives=True`, the DataFrame instead contains:
| Column | Description |
|--------|-------------|
| `id_prompt` | Representative sequence ID |
| `sequence` | Amino acid/nucleotide sequence |

## Interpreting Results

**For Search Results:**
- **High Confidence:** `evalue < 1e-50`, `pident > 90%`
- **Homology Likely:** `evalue < 1e-5`, `pident > 30%` (for proteins)
- **Noise/Chance:** `evalue > 0.01` (usually disregarded)

**For Clustering:**
- **Near-identical:** `min_seq_id = 0.95` (e.g., removing duplicates)
- **Same protein family:** `min_seq_id = 0.50-0.70`
- **Remote homologs:** `min_seq_id = 0.30` (loose clustering)

**Clustering representative selection:**
The representative sequence is not necessarily the "best" sequence—it's the first sequence that covered the cluster during greedy set-cover. Do not assume it is the longest or highest-quality member.

## Quick Start Examples

**Example 1: Search protein sequences against a database**
```python
from bio_programming_tools.tools.gene_annotation import (
    run_mmseqs_search_proteins,
    MmseqsSearchProteinsInput,
    MmseqsSearchProteinsConfig
)

# Search using a FASTA file
inputs = MmseqsSearchProteinsInput(
    query_sequences="my_proteins.faa",
    mmseqs_db="/path/to/target_database"
)
config = MmseqsSearchProteinsConfig(
    results_dir="mmseqs_output",
    sensitivity=4.0
)
result = run_mmseqs_search_proteins(inputs, config)

print(f"Found {result.num_hits} matches!")
print(result.results_df.head())
```

**Example 2: Search using sequence strings directly**
```python
from bio_programming_tools.tools.gene_annotation import (
    run_mmseqs_search_proteins,
    MmseqsSearchProteinsInput,
    MmseqsSearchProteinsConfig
)

# Pass sequences as a list (no file needed)
inputs = MmseqsSearchProteinsInput(
    query_sequences=["MSKGEELFTGVVPIL", "MVSKGEELFTGVVPI"],
    mmseqs_db="reference_proteins.faa"
)
config = MmseqsSearchProteinsConfig(results_dir="results")
result = run_mmseqs_search_proteins(inputs, config)

# Filter for high-confidence hits
good_hits = result.results_df[result.results_df['evalue'] < 1e-10]
print(good_hits)
```

**Example 3: Genome-to-genome nucleotide search**
```python
from bio_programming_tools.tools.gene_annotation import (
    run_mmseqs_search_genomes,
    MmseqsSearchGenomesInput,
    MmseqsSearchGenomesConfig
)

inputs = MmseqsSearchGenomesInput(
    query_genomes="query_genomes.fna",
    target_genomes="target_genomes.fna"
)
config = MmseqsSearchGenomesConfig(
    out_dir="genome_search_results",
    sensitivity=7.5
)
result = run_mmseqs_search_genomes(inputs, config)

print(f"Found {result.num_hits} genomic regions!")
print(result.results_df.head())
```

**Example 4: Cluster sequences to reduce redundancy**
```python
from bio_programming_tools.tools.gene_annotation import (
    run_mmseqs_clustering,
    MmseqsClusteringInput,
    MmseqsClusteringConfig
)

inputs = MmseqsClusteringInput(input_sequences="all_proteins.faa")
config = MmseqsClusteringConfig(
    output_dir="clustering_results",
    min_seq_id=0.90  # 90% identity threshold
)
result = run_mmseqs_clustering(inputs, config)

print(f"Reduced to {result.num_hits} representative sequences")

# Get cluster size distribution
cluster_sizes = result.results_df.groupby('representative').size()
print(f"Largest cluster: {cluster_sizes.max()} sequences")
print(f"Average cluster size: {cluster_sizes.mean():.1f}")
```

## Best Practices & Gotchas

**Parameter Tuning:**

1. `sensitivity`:
   - Default `4.0` is good for most protein searches.
   - Use `7.5` for genome searches or finding distant homologs.
   - Use `1.0` for very fast, low-sensitivity searches (e.g., near-identical matches only).

2. `min_seq_id` (clustering):
   - `0.95`: Removes near-duplicates (sequencing errors, isoforms).
   - `0.60`: Groups proteins into functional families.
   - `0.30`: Very loose clustering for remote homolog detection.

3. `threads`:
   - Set to the number of available CPU cores for best performance.
   - MMseqs2 scales nearly linearly with thread count.

**Common Mistakes:**

1. **Database vs FASTA Confusion:** MMseqs2 can accept either a FASTA file or a pre-built database. For repeated searches, pre-build the database once with `mmseqs createdb` for efficiency.

2. **Memory Issues:** If you run out of memory, increase the `split` parameter (e.g., `split=1` or `split=2`) to process the database in smaller chunks.

3. **Mismatched Sequence Types:** Don't search protein sequences against a nucleotide database or vice versa. Use `search_type=3` for nucleotide searches.

4. **Ignoring `only_top_hits`:** By default, only the best hit per query is returned. Set `only_top_hits=False` if you need all hits above threshold.

5. **Clustering Interpretation:** The representative sequence is not necessarily the "best" sequence—it's the first sequence that covered the cluster during greedy set-cover.

## References

**Primary Citation:**
- Steinegger, M. & Söding, J. (2017). "MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets." *Nature Biotechnology* 35, 1026–1028. [DOI: 10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)

**Documentation:**
- GitHub: [https://github.com/soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2)
- User Guide: [https://mmseqs.com/latest/userguide.pdf](https://mmseqs.com/latest/userguide.pdf)

## Related Tools

**Tools often used together:**
- `blast-search`: Use BLAST for initial small-scale exploration, then switch to MMseqs2 for large-scale analysis.
- `pyhmmer-hmmsearch` / `pyhmmer-hmmscan`: Use PyHMMER for profile-based follow-up on MMseqs2 hits requiring deeper homology analysis.

**Alternative tools:**
- `blast-search`: Use for smaller-scale searches or when you need NCBI database access.
- `pyhmmer-hmmsearch`: Use for profile-based searches when detecting remote homologs (<30% identity).
