<a href="https://bio-pro.mintlify.app/tools/orf-prediction/prodigal"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Prodigal

## Overview
Prodigal (Prokaryotic Dynamic Programming Genefinding Algorithm) is a fast, reliable gene prediction tool specifically designed for [prokaryotic](https://en.wikipedia.org/wiki/Prokaryote) genomes (bacteria and archaea). It identifies protein-coding genes using dynamic programming, including partial genes at sequence ends, and provides detailed annotations including ribosome binding sites and start codon types.

## When to Use This Tool

**Primary use cases:**
- Gene prediction in bacterial/archaeal genomes
- ORF finding in [metagenomic](https://en.wikipedia.org/wiki/Metagenomics) contigs
- Annotating draft genome assemblies
- Identifying coding sequences in plasmids

**When NOT to use this tool:**
- For eukaryotic gene prediction: Use Augustus, GeneMark-ES, or BRAKER instead.
- For simple ORF finding (any frame): Use ORFipy instead.
- For virus annotation: Consider specialized tools like Pharokka instead.
- For RNA genes: Use Infernal/RNAmmer for rRNA, tRNAscan for tRNA instead.

## Biological Background

**Prokaryotic gene structure:**
Prokaryotic genes are simpler than eukaryotic genes:
- No introns (continuous coding sequence)
- [Ribosome binding site](https://en.wikipedia.org/wiki/Ribosome-binding_site) (RBS) upstream of start codon
- Start codons: ATG (most common), GTG, TTG
- Stop codons: TAA, TAG, TGA

**What Prodigal predicts:**
- Gene boundaries (start and end positions)
- Reading frame and strand
- Start codon type (ATG, GTG, TTG)
- Ribosome binding site motif and spacing
- Partial gene status (truncated at sequence edges)

**Meta vs single-genome mode:**
- **Meta mode**: Uses pre-trained parameters, works on short contigs and mixed samples
- **Single-genome mode**: Trains on input sequence, requires >100kb for reliable training

## How It Works

**Algorithm:**
1. Find all potential start and stop codons
2. Score each potential gene using:
   - Coding potential (hexamer frequencies)
   - RBS motif strength
   - Start codon type
3. Use [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) to find optimal gene set
4. Handle overlapping genes and partial genes at edges

**Parallel processing:**
When multiple sequences are provided, Prodigal processes them in parallel using multiple CPU threads.

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_sequences` | `str` or `List[str]` | DNA sequence(s) to analyze |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `meta_mode` | `bool` | `True` | Use meta mode (True) or single-genome mode (False) |
| `translation_table` | `int` | `11` | NCBI genetic code (11 = standard bacterial) |
| `closed_ends` | `bool` | `False` | Prevent partial genes at sequence ends |
| `num_threads` | `int` | auto | CPU threads for parallel processing |

### Parameter Guides

**Mode selection:**
| Mode | When to Use |
|------|-------------|
| Meta mode (`True`) | Contigs, draft assemblies, metagenomic data, mixed samples |
| Single-genome mode (`False`) | Complete/near-complete genomes (>100kb) |

**Translation table options:**
| Code | Description |
|------|-------------|
| `11` | Bacterial, archaeal, plant plastid (default) |
| `4` | Mycoplasma/Spiroplasma |
| `25` | Candidate division SR1, Gracilibacteria |

**Closed ends:**
| Setting | When to Use |
|---------|-------------|
| `False` (default) | Linear contigs, draft assemblies (allows partial genes at edges) |
| `True` | Complete circular genomes (chromosomes, plasmids) |

### Sweep Priorities

1. `meta_mode`; Most impactful; determines whether model trains on input or uses pre-trained parameters
2. `translation_table`; Must match organism; incorrect code produces wrong translations
3. `closed_ends`; Only relevant for complete circular genomes

## Output Specification

**ProdigalOutput**

| Field | Type | Description |
|-------|------|-------------|
| `predicted_orfs` | `List[List[ORF]]` | List of ORF results per input sequence |
| `num_orfs` | `int` | Total genes predicted across all sequences (computed) |
| `num_orfs_per_sequence` | `List[int]` | Gene count per sequence (computed) |
| `results_df` | `DataFrame` | All ORFs as a pandas DataFrame (computed) |

**ORF / DataFrame columns**

| Column | Type | Description |
|--------|------|-------------|
| `parent_id` | `str` | Parent sequence ID (seq_0, seq_1, etc.) |
| `orf_id` | `str` | ORF identifier within parent (gene_1, gene_2, etc.) |
| `amino_acid_sequence` | `str` | Translated protein sequence |
| `nucleotide_sequence` | `str` | DNA sequence of the gene |
| `amino_acid_length` | `int` | Protein length in amino acids |
| `nucleotide_length` | `int` | Gene length in nucleotides |
| `nucleotide_start` | `int` | Start position (1-indexed) |
| `nucleotide_end` | `int` | End position (1-indexed) |
| `strand` | `str` | '+' or '-' |
| `frame` | `int` | Reading frame (1, 2, or 3) |
| `partial` | `str` | Partial status (00_00 = complete, computed) |
| `partial_begin` | `int` | 0 = complete, 1 = truncated at 5' |
| `partial_end` | `int` | 0 = complete, 1 = truncated at 3' |
| `gc_content` | `float` | GC content (0.0-1.0) |
| `start_type` | `str` | Start codon (ATG, GTG, TTG) |
| `rbs_motif` | `str` | RBS motif detected |
| `rbs_spacer` | `str` | Spacing between RBS and start |
| `description` | `str` | Full Prodigal annotation string |

## Interpreting Results

**Sequence length recommendations:**
- **Meta mode**: Works on any length, but very short sequences (<500bp) may miss genes
- **Single-genome mode**: Requires >100kb for reliable training (>20kb minimum)

**Partial gene handling:**
| Partial Code | Meaning |
|-------------|---------|
| `00_00` | Complete gene |
| `10_00` | Truncated at 5' end |
| `00_01` | Truncated at 3' end |
| `10_01` | Both ends truncated |

Partial genes are real genes that extend beyond contig boundaries; do not automatically discard them.

**Start codon distribution:**
In typical bacterial genomes, ~80% of genes use ATG, ~10-15% use GTG, and ~5-10% use TTG. Significant deviations may indicate the wrong translation table.

**RBS motifs:**
Strong RBS motifs (e.g., AGGAG) with 5-10 bp spacers indicate high-confidence gene starts. Weak or absent RBS may indicate leaderless mRNAs or alternative initiation.

## Quick Start Examples

**Example 1: Basic gene prediction**
```python
from proto_tools.tools.orf_prediction.prodigal import (
    run_prodigal_prediction, ProdigalInput, ProdigalConfig
)

sequence = "ATGAAACGTAAACTGGATCGTAACTAGATGCGTAAATAA..."  # Your DNA

inputs = ProdigalInput(input_sequences=sequence)
config = ProdigalConfig(meta_mode=True)

result = run_prodigal_prediction(inputs, config)

print(f"Found {result.num_orfs} genes")
for orfs in result.predicted_orfs:
    for orf in orfs:
        print(f"{orf.orf_id}: {orf.nucleotide_start}-{orf.nucleotide_end} ({orf.strand}), {orf.amino_acid_length} aa")
```

**Example 2: Process multiple sequences**
```python
from proto_tools.tools.orf_prediction.prodigal import (
    run_prodigal_prediction, ProdigalInput, ProdigalConfig
)

sequences = [
    "ATGAAACGTAAACTGGATCGTAACTAG...",
    "ATGCCCGTTAAAGGGCCCAAATGA...",
    "ATGGGGTTTCCCAAAGGGTTTTAG..."
]

inputs = ProdigalInput(input_sequences=sequences)
config = ProdigalConfig(
    meta_mode=True,
    num_threads=4
)

result = run_prodigal_prediction(inputs, config)

for i, (count, orfs) in enumerate(zip(result.num_orfs_per_sequence,
                                       result.predicted_orfs)):
    print(f"Sequence {i}: {count} genes")
```

**Example 3: Complete circular genome**
```python
from proto_tools.tools.orf_prediction.prodigal import (
    run_prodigal_prediction, ProdigalInput, ProdigalConfig
)

# Complete circular bacterial chromosome
with open("chromosome.fasta") as f:
    sequence = "".join(line.strip() for line in f if not line.startswith(">"))

inputs = ProdigalInput(input_sequences=sequence)
config = ProdigalConfig(
    meta_mode=False,  # Single-genome mode for complete genome
    closed_ends=True,  # No partial genes (circular)
    translation_table=11
)

result = run_prodigal_prediction(inputs, config)
print(f"Predicted {result.num_orfs} complete genes")
```

**Example 4: Extract proteins for downstream analysis**
```python
from proto_tools.tools.orf_prediction.prodigal import (
    run_prodigal_prediction, ProdigalInput, ProdigalConfig
)

inputs = ProdigalInput(input_sequences="ATGAAACGT...")
config = ProdigalConfig()

result = run_prodigal_prediction(inputs, config)

# Get all proteins as FASTA
for orf in result.predicted_orfs[0]:
    print(f">{orf.parent_id}_{orf.orf_id}")
    print(orf.amino_acid_sequence)

# Or use the DataFrame for tabular access
df = result.results_df
print(df[['parent_id', 'orf_id', 'amino_acid_length', 'strand']])
```

## Best Practices & Gotchas

**Mode selection:**

1. **Default to meta mode**: For contigs, draft assemblies, or metagenomic data.

2. **Single-genome mode**: Only for complete/near-complete genomes (>100kb).

3. **Mixed samples**: Always use meta mode for metagenomes.

**Handling short sequences:**

1. **Very short contigs (<500bp)**: May not contain complete genes.

2. **Edge genes**: In meta mode, partial genes are allowed by default.

3. **Circular genomes**: Set closed_ends=True for complete circular genomes.

**Post-processing:**

1. **Filter by length**: Very short predicted proteins may be spurious.

2. **Check partial status**: Be aware of truncated genes at contig edges.

3. **Translation table**: Use correct genetic code for your organism.

**Common mistakes:**

1. **Wrong mode for short contigs**: Using single-genome mode on <100kb sequences.

2. **closed_ends on linear fragments**: Losing real genes at contig ends.

3. **Ignoring partial genes**: Partial genes are real; don't automatically discard them.

## References

**Primary publication:**
- Hyatt et al. (2010). "Prodigal: prokaryotic gene recognition and translation initiation site identification". *BMC Bioinformatics*. DOI: 10.1186/1471-2105-11-119

**Resources:**
- Pyrodigal documentation: https://pyrodigal.readthedocs.io/
- Original Prodigal: https://github.com/hyattpd/Prodigal

## Related Tools

**Tools often used together:**
- `blast-search`: Annotate predicted proteins against sequence databases.
- `pyhmmer-hmmscan`: Annotate predicted proteins with protein domain profiles from Pfam.
- `esm2`: Score predicted proteins with a protein language model.

**Alternative tools:**
- `orfipy`: General ORF prediction: more flexible for custom start/stop codons but less accurate for prokaryotic gene calling.
