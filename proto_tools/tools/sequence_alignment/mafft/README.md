<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/mafft"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# MAFFT

## Overview
MAFFT (Multiple Alignment using [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform)) is a widely used tool for [multiple sequence alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSA). It offers various strategies ranging from fast approximate alignments for large datasets to highly accurate iterative methods for smaller sets of sequences. This module provides a standardized interface for performing MSA using MAFFT.

## Background

**What does this tool do?**
MAFFT aligns multiple biological sequences (protein or nucleotide) by inserting gap characters to maximize the similarity between sequences. The resulting alignment reveals conserved regions, evolutionary relationships, and functional domains.

**Why is this important?**
Multiple sequence alignment is fundamental to many bioinformatics analyses:
- **[Phylogenetics](https://en.wikipedia.org/wiki/Phylogenetics)**: Inferring evolutionary relationships between sequences
- **Conservation analysis**: Identifying functionally important residues
- **[Homology modeling](https://en.wikipedia.org/wiki/Homology_modeling)**: Building protein structures based on related sequences
- **Motif discovery**: Finding conserved patterns across protein families
- **Variant analysis**: Understanding the impact of mutations in context

**Scientific foundation:**
MAFFT uses several algorithmic approaches:

1. **FFT-NS-i**: Uses Fast Fourier Transform to rapidly identify homologous segments, followed by iterative refinement.
2. **L-INS-i** (localpair): Local pairwise alignment with iterative refinement. Best for sequences with one alignable domain.
3. **G-INS-i** (globalpair): Global pairwise alignment with iterative refinement. Best for sequences of similar length.
4. **E-INS-i** (genafpair): Local alignment considering multiple conserved domains. Best for sequences with large unalignable regions.

## How It Works

MAFFT builds alignments in several stages:

1. **All-to-all comparison**: Rapidly estimates pairwise distances using k-mer counting or FFT
2. **Guide tree construction**: Builds a tree to determine alignment order
3. **Progressive alignment**: Aligns sequences following the guide tree
4. **Iterative refinement**: Repeatedly improves the alignment (for iterative methods)

**Key assumptions:**
- Input sequences are homologous (evolutionarily related)
- Sequences can be meaningfully aligned (share common ancestry)
- Gap penalties are appropriate for the sequence type

**Limitations:**
- Very divergent sequences (<20% identity) may not align well
- Alignments of >10,000 sequences may require significant time/memory
- Quality decreases as sequence similarity decreases

**Computational requirements:**
- **Hardware:** CPU only, scales with threads
- **Runtime:** ~1 second for 10 sequences, minutes for thousands
- **Scalability:** Excellent parallelization with `--thread`

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `List[str]` | List of sequence strings to align (minimum 2 required) |
| `sequence_ids` | `Optional[List[str]]` | Optional sequence identifiers. If not provided, defaults to `seq_0`, `seq_1`, etc. These IDs are preserved in the output MSA |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `align_method` | `str` | `"auto"` | Alignment method: `auto`, `localpair`, `globalpair`, or `genafpair` |
| `max_iterations` | `int` | `0` | Maximum iterative refinement cycles (0 = method default) |
| `threads` | `int` | `1` | Number of CPU threads for parallel processing |

### Parameter Guides

**Alignment method selection:**

| `align_method` | Method | Best For | Max Sequences |
|----------|--------|----------|---------------|
| `auto` | Auto-select | General use | Any |
| `localpair` | L-INS-i | One alignable domain with flanking regions | ~200 |
| `globalpair` | G-INS-i | Similar-length sequences, global homology | ~200 |
| `genafpair` | E-INS-i | Multiple domains, large unalignable regions | ~200 |

## Output Specification

**Return type:** `MafftOutput`

```python
MafftOutput(
    msa=MSA(
        aligned_sequences=["MVLSPADKTN", "MVLSAADKTN"],
        sequence_ids=["seq_0", "seq_1"],
    )
)
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `msa` | `MSA` | Multiple sequence alignment object with aligned sequences |

**MSA fields:**

| Field | Type | Description |
|-------|------|-------------|
| `aligned_sequences` | `List[str]` | Aligned sequences with gap characters ('-') |
| `sequence_ids` | `List[str]` | Sequence identifiers (e.g., "seq_0", "seq_1") |
| `original_sequences` | `List[str]` | Original input sequences without gaps |

**MSA properties:**
- `num_sequences`: Number of sequences in alignment
- `alignment_length`: Length of alignment (all sequences have this length)
- `total_gaps`: Total gap characters across all sequences
- `average_gap_fraction`: Average fraction of gaps
- `get_column(pos)`: Get all residues at a position
- `get_conservation(pos)`: Calculate conservation score at a position
- `get_position_frequencies(pos)`: Get character frequencies at a position
- `to_fasta()`: Convert alignment to FASTA format

## Interpreting Results

**Alignment quality indicators:**
- **High quality:** Average gap fraction < 20%, conservation > 0.7 at key positions
- **Moderate quality:** Average gap fraction 20-40%
- **Poor quality:** Average gap fraction > 40%, may indicate non-homologous sequences

**When to increase iterations:**
- Default (`max_iterations=0`): Good for most cases
- `max_iterations=100-1000`: For difficult alignments needing refinement

## Quick Start Examples

**Example 1: Basic alignment with auto method**
```python
from proto_tools.tools.sequence_alignment import (
    run_mafft_align,
    MafftInput,
    MafftConfig
)

# Align protein sequences with custom IDs
inputs = MafftInput(
    sequences=[
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
        "MVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
        "MVLTPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
    ],
    sequence_ids=["human_hbb", "mouse_hbb", "rat_hbb"]
)
config = MafftConfig(align_method="auto")
result = run_mafft_align(inputs, config)

msa = result.msa
print(f"Alignment length: {msa.alignment_length}")
print(f"Number of sequences: {msa.num_sequences}")

for seq_id, aligned_seq in zip(msa.sequence_ids, msa.aligned_sequences):
    print(f"{seq_id}: {aligned_seq}")
# Output: human_hbb: MVLSPADKTN..., mouse_hbb: MVLSAADKTN..., etc.
```

**Example 2: High-accuracy alignment for small dataset**
```python
from proto_tools.tools.sequence_alignment import (
    run_mafft_align,
    MafftInput,
    MafftConfig
)

# Use L-INS-i for accurate alignment
inputs = MafftInput(
    sequences=[
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
        "MVLSAADKTNVKAAWSKVGAHAGEYGAEALERMFLSFPTTK",
        "MVLTPADKTNVKAAWGKVGAHAGEYGAEALERMFLGFPTTK",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
    ]
)
config = MafftConfig(
    align_method="localpair",
    max_iterations=1000,
    threads=4
)
result = run_mafft_align(inputs, config)

# Analyze conservation at position 5
conservation = result.msa.get_conservation(5)
print(f"Conservation at position 5: {conservation:.2f}")

# Export to FASTA
fasta_output = result.msa.to_fasta()
print(fasta_output)
```

**Example 3: Analyze gap distribution**
```python
from proto_tools.tools.sequence_alignment import (
    run_mafft_align,
    MafftInput,
    MafftConfig
)

inputs = MafftInput(
    sequences=[
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
        "MVLSAADKVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",  # Shorter
        "MVLTPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKEXTRA",  # Longer
    ]
)
config = MafftConfig()
result = run_mafft_align(inputs, config)

msa = result.msa
print(f"Average gap fraction: {msa.average_gap_fraction:.2%}")
print(f"Total gaps: {msa.total_gaps}")

for seq_id, aligned_seq in zip(msa.sequence_ids, msa.aligned_sequences):
    gap_count = aligned_seq.count("-")
    gap_fraction = gap_count / len(aligned_seq)
    print(f"{seq_id}: {gap_count} gaps ({gap_fraction:.1%})")
```

## Best Practices & Gotchas

**Parameter tuning:**

1. **`align_method`**:
   - Start with `auto` for most cases
   - Use `localpair` for proteins with conserved domains flanked by variable regions
   - Use `globalpair` for full-length homologs of similar size
   - Use `genafpair` for multi-domain proteins with insertions

2. **`max_iterations`**:
   - Keep at 0 for quick alignments
   - Set to 1000 for publication-quality alignments of <200 sequences

3. **`threads`**:
   - Set to available CPU cores for large alignments
   - Single thread sufficient for <100 sequences

**Common mistakes:**

1. **Aligning non-homologous sequences**: MAFFT will produce an alignment even for unrelated sequences, but it won't be meaningful.

2. **Using iterative methods on large datasets**: `localpair`, `globalpair`, and `genafpair` are O(N^2) and slow for >200 sequences. Use `auto` for large datasets.

3. **Ignoring gap patterns**: Many gaps in one region may indicate alignment errors or genuinely variable regions.

4. **Over-interpreting low-identity alignments**: Alignments with <30% identity should be interpreted cautiously.

**Tips for optimal results:**
- Pre-filter sequences to remove obvious outliers
- Remove redundant sequences (>95% identity) for cleaner alignments
- Manually inspect alignments at conserved positions
- Consider trimming poorly aligned regions for downstream analysis

## References

**Primary publication:**
- Katoh, K. & Standley, D.M. (2013). "MAFFT Multiple Sequence Alignment Software Version 7: Improvements in Performance and Usability." *Molecular Biology and Evolution* 30(4):772-780. [DOI: 10.1093/molbev/mst010](https://doi.org/10.1093/molbev/mst010)

**Implementation:**
- GitHub: [https://github.com/GSLBiotech/mafft](https://github.com/GSLBiotech/mafft)
- Documentation: [https://mafft.cbrc.jp/alignment/software/](https://mafft.cbrc.jp/alignment/software/)

**Additional resources:**
- MAFFT online server: [https://mafft.cbrc.jp/alignment/server/](https://mafft.cbrc.jp/alignment/server/)
- Algorithm comparison: [https://mafft.cbrc.jp/alignment/software/algorithms/algorithms.html](https://mafft.cbrc.jp/alignment/software/algorithms/algorithms.html)

## Related Tools

**Tools often used together:**
- `mmseqs-clustering`: Cluster sequences before alignment to reduce redundancy
- `pyhmmer`: Build HMM profiles from alignments
- `esmfold`: Predict structures from aligned sequences

**Alternative tools:**
- Clustal Omega: Alternative MSA tool, available in Biopython
- MUSCLE: Fast MSA, good for large datasets
- T-Coffee: Higher accuracy for small sets, significantly slower
