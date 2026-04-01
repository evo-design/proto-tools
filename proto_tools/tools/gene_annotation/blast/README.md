<a href="https://bio-pro.mintlify.app/tools/gene-annotation/blast"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# BLAST

## Overview
BLAST (Basic Local Alignment Search Tool) finds regions of similarity between biological sequences. It compares nucleotide or protein sequences to sequence databases and calculates the statistical significance of matches. This module provides a unified interface for both *Online BLAST* (querying NCBI servers remotely) and *Local BLAST+* (running searches against custom or downloaded databases on your own hardware), as well as utilities for creating custom BLAST databases.

## When to Use This Tool

**Primary use cases:**
- Finding homologous sequences in databases
- Identifying protein families and domains
- Annotating newly sequenced genes

**When to use Online vs. Local BLAST:**
- Online BLAST (`search_mode="online"`): Best for occasional, low-volume queries where you need access to the massive NCBI databases (nt/nr) without downloading terabytes of data.
- Local BLAST (`search_mode="local"`): Best for high-throughput pipelines, privacy (proprietary sequences), or searching against custom/smaller databases (e.g., searching a specific genome).

**When NOT to use this tool:**
- Whole Genome Alignment: Use tools like [MUMmer](https://mummer4.github.io/) or Minimap2 for aligning entire chromosomes.
- Next-Gen Sequencing (NGS) Mapping: Use BWA or Bowtie2 for mapping millions of short reads to a reference.
- Deep Homology Detection: If sequences are very distantly related (<20% identity), Hidden Markov Model tools like [HMMER](http://hmmer.org/) or HHblits are more sensitive.
- Large-Scale Searches: For searching millions of sequences, use [MMseqs2](https://mmseqs.com/) which is 100-1000x faster.

## Biological Background

**What does this tool do?**
[BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) finds regions of local similarity between sequences by comparing nucleotide or protein sequences to sequence databases like [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/) and calculates the statistical significance of matches.

**Why is this important?**
Sequence alignment is the first step in almost all bioinformatics workflows. It allows researchers to:
- Infer functional relationships: If a new sequence resembles a known gene, they likely share a function.
- Identify species: Map unknown DNA reads to specific organisms (metagenomics).
- Detect off-targets: Ensure a primer or CRISPR guide only binds to the intended target.
- Find evolutionary origins: Trace the phylogeny of a gene across species.

**Scientific foundation:**
BLAST uses a heuristic algorithm that seeks high-scoring segment pairs (HSPs). It does not perform a full [Smith-Waterman](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) alignment (which is accurate but slow). Instead, it does:

1. **Seeding**: Breaks the query into short "words" (k-mers) and finds exact matches in the database.
2. **Extension**: Extends these matches in both directions until the alignment score drops below a threshold.
3. **Evaluation**: Calculates an [E-value](https://en.wikipedia.org/wiki/BLAST_(biotechnology)#Algorithm) (Expect value) based on the Karlin-Altschul statistics, representing the number of hits one can expect to see by chance.

## Tool Catalog

| Tool | Description | Use Case |
|------|-------------|----------|
| `blast-search` | Unified BLAST search supporting both online (NCBI) and local modes | Sequence homology search against any BLAST database |
| `blast-create-db` | Creates a local BLAST database from a FASTA file | Prerequisite for local `blast-search` |

## How It Works

**Method overview:**
The module exposes two tools:

`blast-search`: A unified search tool that dispatches to either online (NCBI QBLAST) or local (BLAST+ CLI) search based on the `search_mode` config parameter. Online mode submits requests to NCBI and parses XML results. Local mode executes BLAST+ binaries and parses tabular output.

`blast-create-db`: Wraps the makeblastdb command line tool. It takes a FASTA file and indexes it into binary files (.nhr, .nin, .nsq etc.) optimized for search.

**The BLAST Program Matrix:**
You must select the correct program based on your inputs:
| Program | Query Type | Database Type | Use Case |
|-----------|------|---------|-------------|
| `blastn` | Nucleotide | Nucleotide | Gene identification, mapping oligonucleotides |
| `blastp` | Protein | Protein | Protein function prediction, finding homologs |
| `blastx` | Nucleotide | Protein | Coding sequence finding (translates query in 6 frames) |
| `tblastn` | Protein | Nucleotide | Finding unannotated genes in genomic DNA (translates DB) |
| `tblastx` | Nucleotide | Nucleotide | Gene prediction in divergent species (translates both) |

i.e., if you are searching a nucleotide sequence against a protein database, use `blastx`.

**Computational Requirements:**

Online: Minimal local resources, but high latency (seconds to minutes per query) and rate-limited by NCBI.

Local: CPU-intensive. RAM usage depends on database size.

Storage: Local databases can be large (NCBI 'nt' is >100GB compressed; custom DBs are small).

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Your sequence as a string (e.g., `"ATGCGTAAA..."`) OR a path to a FASTA file. Automatically detected. |

The input has strict type validation: if the value is an existing file path, it's treated as a FASTA file; otherwise it's validated as a raw sequence string containing only valid sequence characters.

## Configuration

**Core Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_mode` | `str` | `"online"` | `"online"` for NCBI servers, `"local"` for BLAST+ CLI |
| `program` | `str` | `"blastn"` | BLAST algorithm (see Program Matrix above) |

**Online-Only Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `database` | `str` | `"nt"` | NCBI database to search against |
| `hitlist_size` | `int` | `None` | Number of hits to return (NCBI default: 50) |
| `entrez_query` | `str` | `None` | Restrict search with an Entrez query |
| `megablast` | `bool` | `None` | Use MegaBLAST algorithm (blastn only) |

**Local-Only Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `local_db` | `str` | *Required* | Path to local BLAST database (created by `blast-create-db`) |
| `num_threads` | `int` | `4` | Number of CPU threads |

**Scoring Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evalue` | `float` | `None` | E-value threshold (BLAST default: 10.0) |
| `word_size` | `int` | `None` | Word size for initial matches |
| `gapopen` | `int` | `None` | Cost to open a gap |
| `gapextend` | `int` | `None` | Cost to extend a gap |
| `matrix` | `str` | `None` | Scoring matrix ([BLOSUM62](https://en.wikipedia.org/wiki/BLOSUM), PAM30, etc.) |
| `reward` | `int` | `None` | Nucleotide match reward (blastn only) |
| `penalty` | `int` | `None` | Nucleotide mismatch penalty (blastn only) |

**Filtering Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_target_seqs` | `int` | `None` | Max aligned sequences to keep |
| `max_hsps` | `int` | `None` | Max HSPs per query-subject pair |
| `perc_identity` | `float` | `None` | Min percent identity (0-100) |
| `qcov_hsp_perc` | `float` | `None` | Min query coverage per HSP (0-100) |

**Search Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `str` | `None` | Task preset (megablast, blastn-short, blastp-fast, etc.) |
| `ungapped` | `bool` | `None` | Perform ungapped alignment only |
| `strand` | `str` | `None` | Query strand: both, plus, minus |

See `BlastSearchConfig` class for the full list of supported parameters.

### Parameter Guides

**Program selection:**
| Query Type | Database Type | Use Program |
|------------|---------------|-------------|
| Nucleotide | Nucleotide | `blastn` |
| Protein | Protein | `blastp` |
| Nucleotide | Protein | `blastx` |
| Protein | Nucleotide | `tblastn` |
| Nucleotide (translated) | Nucleotide (translated) | `tblastx` |

**E-value interpretation:**
| E-value | Meaning |
|---------|---------|
| `< 1e-50` | Near-certain homolog |
| `< 1e-5` | Strong evidence of homology |
| `< 0.01` | Possible homolog (inspect manually) |
| `> 0.01` | Likely noise/chance |

### Sweep Priorities

1. `evalue`; Most impactful; controls sensitivity vs noise tradeoff
2. `program`; Must match query/database types; incorrect choice yields no results
3. `word_size`; Decrease to find shorter/more divergent matches (slower), increase for speed

## Output Specification

`blast-search` returns a `BlastSearchOutput` object with two attributes:
- `results_df`: A Pandas DataFrame with all hits (or empty if no matches)
- `num_hits`: Total number of hits found

**DataFrame Columns (standard "Outfmt 6"):**

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `qseqid` | Query ID | The ID of your input sequence. |
| `sseqid` | Subject ID | The ID of the match in the database. |
| `pident` | Percent Identity | % of exact matches. >95% usually implies same species/gene. |
| `length` | Alignment Length | How long the matching region is. |
| `mismatch` | Mismatches | Number of non-matching positions. |
| `gapopen` | Gap Openings | Number of gaps introduced in the alignment. |
| `qstart` | Query Start | Position in your sequence where the match begins. |
| `qend` | Query End | Position in your sequence where the match ends. |
| `sstart` | Subject Start | Position in the database sequence where the match begins. |
| `send` | Subject End | Position in the database sequence where the match ends. |
| `evalue` | Expect Value | **Crucial statistic.** The number of hits expected by chance. Lower is better. |
| `bitscore` | Bit Score | Alignment score normalized for database size. Higher is better. |

## Interpreting Results

- **High Confidence:** `evalue < 1e-50`, `pident > 90%`
- **Homology Likely:** `evalue < 1e-5`, `pident > 30%` (for proteins)
- **Noise/Chance:** `evalue > 0.01` (usually disregarded unless looking for very short motifs)

**Key considerations:**
- A 100% identity match over only 20 base pairs usually has a poor E-value and may be biologically meaningless. Always check `length` alongside `pident`.
- E-values are database-size dependent: the same alignment score produces different E-values against different databases.
- Bit scores are normalized and comparable across databases.

## Quick Start Examples

**Example 1: Search NCBI online (simplest)**
```python
from proto_tools.tools.gene_annotation import run_blast_search, BlastSearchInput, BlastSearchConfig

# Your DNA sequence
inputs = BlastSearchInput(query="ATGCGTAAACGATTGCAGTACGATCGATCG")

# Search the nucleotide database (online by default)
config = BlastSearchConfig(program="blastn", database="nt")

# Run the search (may take 30+ seconds)
result = run_blast_search(inputs, config)

print(f"Found {result.num_hits} matches!")
print(result.results_df.head())  # View top hits
```

**Example 2: Create a local database and search it**
```python
from proto_tools.tools.gene_annotation import (
    run_create_blast_db, CreateBlastDbInput, CreateBlastDbConfig,
    run_blast_search, BlastSearchInput, BlastSearchConfig
)

# Step 1: Create a database from your reference sequences
db_input = CreateBlastDbInput(fasta="my_reference_genes.fasta")
db_config = CreateBlastDbConfig(dbtype="nucl", title="My Gene Database")
db_result = run_create_blast_db(db_input, db_config)

print(f"Database created at: {db_result.db_path}")

# Step 2: Search against your new database
search_input = BlastSearchInput(query="my_query.fasta")
search_config = BlastSearchConfig(
    search_mode="local",
    program="blastn",
    local_db=db_result.db_path,
    num_threads=4
)
result = run_blast_search(search_input, search_config)

# Filter for high-confidence hits
good_hits = result.results_df[result.results_df['evalue'] < 1e-10]
print(good_hits)
```

## Best Practices & Gotchas

**Parameter Tuning:**

1. `evalue` (Expect Threshold):
   - Default is usually 10.0 (very loose).
   - Set to `1e-5` or `1e-3` to filter out random noise.

2. `word_size`:
   - Decrease this to find shorter/more divergent matches (increases runtime).
   - Increase to speed up search for near-identical matches.

3. `dust` / `soft_masking`:
   - BLAST automatically masks low-complexity regions (e.g., "AAAAA"). If you are searching for repetitive motifs, you may need to disable masking.

**Common Mistakes:**

1. **Mismatched DB/Program:** Trying to run `blastp` (protein query) against a Nucleotide database. You must use `tblastn` for this.

2. **Online Latency:** Do not use `blast-search` with `search_mode="online"` inside a loop for 10,000 sequences. It will take days and NCBI may block your IP. Use `search_mode="local"` for batch processing.

3. **Interpreting Short Hits:** A 100% identity match over only 20 base pairs usually has a poor E-value and may be biologically meaningless. Always check `length` alongside `pident`.

## References

**Primary Citation:**
- Altschul, S.F., Gish, W., Miller, W., Myers, E.W. & Lipman, D.J. (1990) "Basic local alignment search tool." *J. Mol. Biol.* 215:403-410.

**Documentation:**
- BLAST+ Manual: [https://www.ncbi.nlm.nih.gov/books/NBK279690/](https://www.ncbi.nlm.nih.gov/books/NBK279690/)
- Biopython BLAST: [https://biopython.org/DIST/docs/tutorial/Tutorial.html#chapter:blast](https://biopython.org/DIST/docs/tutorial/Tutorial.html#chapter:blast)

## Related Tools

**Tools often used together:**
- `blast-create-db`: Create custom databases for local `blast-search`.
- `muscle` / `mafft`: Use these to align the sequences *after* BLAST finds them (Multiple Sequence Alignment).
- `pyhmmer-hmmscan`: Follow up BLAST hits with domain annotation for functional characterization.

**Alternative tools:**
- `mmseqs-search-proteins` / `mmseqs-search-genomes`: Faster alternative for very large-scale searches (100-1000x faster).
- `pyhmmer-hmmsearch` / `pyhmmer-phmmer`: More sensitive for detecting remote homologs using profile HMMs.
