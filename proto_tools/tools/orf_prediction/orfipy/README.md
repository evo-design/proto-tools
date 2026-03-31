<a href="https://bio-pro.mintlify.app/tools/orf-prediction/orfipy"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ORFipy

## Overview
ORFipy is a fast, flexible ORF (Open Reading Frame) prediction tool that identifies potential coding regions in DNA sequences based on start and stop codons. Unlike gene prediction tools like Prodigal, ORFipy performs simple ORF finding without machine learning - it reports all ORFs matching your criteria, making it ideal for exploratory analysis and custom filtering.

## When to Use This Tool

**Primary use cases:**
- Finding all possible ORFs in any DNA sequence
- Exploratory analysis before specialized gene prediction
- Custom ORF filtering with specific start/stop codons
- ORF finding in eukaryotic sequences (with custom parameters)

**When NOT to use this tool:**
- For accurate prokaryotic gene prediction: Use Prodigal instead.
- For eukaryotic gene prediction (with introns): Use Augustus or GeneMark instead.
- When you need RBS/promoter predictions: Use Prodigal instead.
- When you need ML-based gene scoring: Use Prodigal instead (it scores coding potential via hexamer frequencies).

## Biological Background

**What is an ORF?**
An [Open Reading Frame](https://en.wikipedia.org/wiki/Open_reading_frame) is a stretch of DNA between a [start codon](https://en.wikipedia.org/wiki/Start_codon) and an in-frame [stop codon](https://en.wikipedia.org/wiki/Stop_codon):
- **Start codons**: ATG (standard), GTG, TTG (alternative)
- **Stop codons**: TAA (ochre), TAG (amber), TGA (opal)
- **[Reading frames](https://en.wikipedia.org/wiki/Reading_frame)**: 3 forward (+1, +2, +3) and 3 reverse (-1, -2, -3)

**ORF vs Gene:**
- **ORF**: Any sequence matching start/stop pattern (may include non-coding)
- **Gene**: Biologically functional coding sequence (subset of ORFs)

ORFipy finds ORFs; gene predictors like Prodigal use additional signals to distinguish real genes.

## How It Works

**Algorithm:**
1. Scan sequence for start codons on specified strand(s)
2. Find first in-frame stop codon after each start
3. Apply length filters
4. Output both nucleotide and amino acid sequences

**Six-frame translation:**
- Forward strand: Frames +1, +2, +3
- Reverse strand: Frames -1, -2, -3

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `str` or `List[str]` | DNA sequence(s) to scan |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_codons` | `str` | `ATG,GTG,TTG` | Comma-separated start codons |
| `stop_codons` | `str` | `TAA,TAG,TGA` | Comma-separated stop codons |
| `strand` | `str` | `b` | Strand to scan: `f` (forward), `r` (reverse), `b` (both) |
| `min_len` | `int` | `0` | Minimum ORF length in nucleotides |
| `max_len` | `int` | `10000` | Maximum ORF length in nucleotides |
| `include_stop` | `bool` | `True` | Include stop codon in reported sequence |
| `translation_table` | `int` | `None` | NCBI genetic code (1-33), None = standard |
| `threads` | `int` | `4` | CPU threads per sequence |

### Parameter Guides

**Start codon selection:**
| Setting | Codons | Use Case |
|---------|--------|----------|
| Standard | `ATG` | Eukaryotic sequences, strict filtering |
| Prokaryotic (default) | `ATG,GTG,TTG` | Bacterial/archaeal sequences |
| Custom | Any valid codons | Specialized genetic codes |

**Length filtering guidelines:**
| min_len | Approximate proteins | Use case |
|---------|---------------------|----------|
| `0` | All ORFs | Comprehensive analysis |
| `150` | >50 aa | Filter very short ORFs |
| `300` | >100 aa | Typical small proteins |
| `600` | >200 aa | Substantial proteins only |
| `900` | >300 aa | Large proteins only |

### Sweep Priorities

1. `min_len` — Most impactful for controlling output volume and filtering noise
2. `start_codons` — Controls whether alternative start codons are included
3. `strand` — Reduce to `f` or `r` when coding strand is known

## Output Specification

**ORFipyOutput**

| Field | Type | Description |
|-------|------|-------------|
| `predicted_orfs` | `List[List[Orf]]` | ORFs per input sequence |
| `num_orfs` | `int` | Total ORFs across all sequences (computed) |
| `results_df` | `DataFrame` | All ORFs as DataFrame (computed) |

**Orf object / DataFrame columns**

| Field | Type | Description |
|-------|------|-------------|
| `parent_id` | `str` | Parent sequence ID (seq_0, seq_1, etc.) |
| `orf_id` | `str` | ORF identifier within parent |
| `strand` | `str` | '+' or '-' |
| `frame` | `int` | Reading frame (1, 2, or 3) |
| `amino_acid_sequence` | `str` | Translated protein |
| `nucleotide_sequence` | `str` | DNA sequence |
| `amino_acid_length` | `int` | Protein length |
| `nucleotide_length` | `int` | ORF length in bp |
| `nucleotide_start` | `int` | Start position (1-indexed) |
| `nucleotide_end` | `int` | End position (1-indexed) |

## Interpreting Results

**Length-based filtering:**
- ORFs < 150 nt (~50 aa) are usually spurious in prokaryotic contexts.
- ORFs > 300 nt (~100 aa) are more likely to be real coding sequences.
- The longest ORF per reading frame is often the best candidate for the real gene.

**Codon variations:**
- Standard: ATG only as start
- Prokaryotic: ATG, GTG, TTG as starts (GTG and TTG account for ~15-20% of bacterial genes)
- Custom: Any valid codons for specialized genetic codes

**ORF vs gene prediction:**
ORFipy reports all ORFs matching your criteria — it does not score or rank them. For distinguishing real genes from spurious ORFs, use Prodigal or annotate ORF translations against protein databases with BLAST/HMMER.

## Quick Start Examples

**Example 1: Basic ORF finding**
```python
from proto_tools.tools.orf_prediction.orfipy import (
    run_orfipy_prediction, OrfipyInput, OrfipyConfig
)

sequence = "ATGAAACGTAAACTGGATCGTAACTAGATGCGTAAATAA"

inputs = OrfipyInput(sequences=sequence)
config = OrfipyConfig(min_len=30)  # At least 10 codons

result = run_orfipy_prediction(inputs, config)

print(f"Found {result.num_orfs} ORFs")
print(result.results_df)
```

**Example 2: ATG-only starts**
```python
from proto_tools.tools.orf_prediction.orfipy import (
    run_orfipy_prediction, OrfipyInput, OrfipyConfig
)

inputs = OrfipyInput(sequences="ATGCGTAAACTGATGTAA...")
config = OrfipyConfig(
    start_codons="ATG",  # Only standard start
    min_len=150
)

result = run_orfipy_prediction(inputs, config)
print(f"Found {result.num_orfs} ATG-initiated ORFs")
```

**Example 3: Forward strand only**
```python
from proto_tools.tools.orf_prediction.orfipy import (
    run_orfipy_prediction, OrfipyInput, OrfipyConfig
)

inputs = OrfipyInput(sequences="ATGCGTAAACTG...")
config = OrfipyConfig(
    strand="f",  # Forward only
    min_len=300
)

result = run_orfipy_prediction(inputs, config)
```

**Example 4: Process multiple sequences**
```python
from proto_tools.tools.orf_prediction.orfipy import (
    run_orfipy_prediction, OrfipyInput, OrfipyConfig
)

sequences = [
    "ATGAAACGTAAACTGGATCGTAACTAG",
    "ATGCCCGTTAAAGGGCCCAAATGA",
]

inputs = OrfipyInput(sequences=sequences)
config = OrfipyConfig(min_len=12)

result = run_orfipy_prediction(inputs, config)

# Access per-sequence results
for i, orfs in enumerate(result.predicted_orfs):
    print(f"Sequence {i}: {len(orfs)} ORFs")
    for orf in orfs:
        print(f"  {orf.orf_id}: {orf.amino_acid_length} aa, frame {orf.frame}")
```

**Example 5: Extract longest ORF per sequence**
```python
from proto_tools.tools.orf_prediction.orfipy import (
    run_orfipy_prediction, OrfipyInput, OrfipyConfig
)

inputs = OrfipyInput(sequences=["ATGAAACGT...", "ATGCCCGTT..."])
config = OrfipyConfig()

result = run_orfipy_prediction(inputs, config)

# Get longest ORF per sequence
for i, orfs in enumerate(result.predicted_orfs):
    if orfs:
        longest = max(orfs, key=lambda x: x.amino_acid_length)
        print(f"Seq {i}: Longest ORF is {longest.amino_acid_length} aa")
```

## Best Practices & Gotchas

**Start codon selection:**

1. **Standard (ATG only)**: Use for eukaryotic sequences or strict filtering.

2. **Prokaryotic (ATG,GTG,TTG)**: Default, includes alternative bacterial starts.

3. **Custom codons**: Can specify any valid codons.

**Length filtering:**

1. **Very short ORFs**: min_len=0 includes tiny ORFs that are usually spurious.

2. **Reasonable minimum**: min_len=150 (~50 aa) filters most noise.

3. **Balance sensitivity/specificity**: Adjust based on your analysis goals.

**Strand selection:**

1. **Both strands (b)**: Default, finds ORFs on forward and reverse.

2. **Single strand (f/r)**: Use when you know the coding strand.

**Common mistakes:**

1. **Confusing ORFs with genes**: Not all ORFs are functional genes.

2. **No length filter**: Getting thousands of tiny spurious ORFs.

3. **Wrong genetic code**: Using standard code for mitochondrial sequences.

## References

**Primary publication:**
- Singh & Wurtele (2021). "orfipy: a fast and flexible tool for extracting ORFs". *Bioinformatics*. DOI: 10.1093/bioinformatics/btab090

**Resources:**
- GitHub: https://github.com/urmi-21/orfipy

## Related Tools

**Tools often used together:**
- `blast-search`: Annotate ORF translations against sequence databases to identify gene function.
- `pyhmmer-hmmscan`: Annotate ORF translations with protein domain profiles from Pfam.
- `esm2`: Score ORF translations with a protein language model to assess protein-likeness.

**Alternative tools:**
- `prodigal`: ML-based prokaryotic gene prediction — more accurate for bacteria/archaea but less flexible for custom ORF criteria.
