# MinCED

## Overview
MinCED (Mining CRISPRs in Environmental Datasets) is a tool for detecting CRISPR arrays in nucleotide sequences. It identifies repeat-spacer arrays using a heuristic k-mer search algorithm derived from the CRISPR Recognition Tool (CRT), returning structured information about each array's repeats, spacers, and their genomic positions.

## Biological Background

**What does this tool measure/predict?**
MinCED detects CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) arrays in DNA sequences. CRISPR arrays consist of short, conserved repeat sequences separated by unique spacer sequences derived from past viral infections.

**Why is this important?**
CRISPR arrays are the adaptive immune memory of prokaryotes. Each spacer records a past viral encounter, and the repeats serve as structural elements recognized by CRISPR-associated (Cas) proteins. Detecting CRISPR arrays is essential for:
- Identifying CRISPR-Cas defense systems in genomes
- Characterizing spacer repertoires for phage-host interaction studies
- Validating that generated DNA sequences contain functional CRISPR loci
- Engineering new CRISPR systems by understanding natural array architecture

**Scientific foundation:**
CRISPR arrays have a characteristic structure: direct repeats of 23-47 nucleotides separated by unique spacers of 26-50 nucleotides. MinCED exploits this regularity by searching for repeated k-mers at consistent intervals, then extending and refining candidate arrays. The algorithm is based on CRT (CRISPR Recognition Tool), which uses a seed-and-extend approach analogous to BLAST but tuned for the specific repeat-spacer pattern of CRISPR loci.

## When to Use This Tool

**Primary use cases:**
- Detecting CRISPR arrays in newly sequenced prokaryotic genomes
- Validating that generated DNA sequences (e.g., from Evo1) contain CRISPR arrays
- Extracting spacer sequences for downstream phage-host analysis
- Counting and characterizing CRISPR arrays in metagenomic assemblies

**When NOT to use this tool:**
- For identifying Cas protein genes: use Prodigal + PyHMMER with Cas HMM profiles instead
- For predicting tracrRNA: use `crispr-tracr` (CRISPRtracrRNA) instead
- For CRISPR guide RNA design: use specialized guide design tools
- For eukaryotic sequences: MinCED is designed for prokaryotic CRISPR arrays

**Comparison with alternatives:**
- **CRISPRFinder/CRISPRCasFinder**: Web-based tools with more comprehensive annotation but not available as local Python libraries
- **PILER-CR**: Alternative local tool; MinCED generally has higher sensitivity for short arrays
- **MinCED**: Fast, command-line friendly, good sensitivity for arrays with >=3 repeats

## How It Works

**Method overview:**
MinCED uses a heuristic k-mer search to find candidate CRISPR arrays. It scans the input sequence for exact k-mer matches occurring at regular intervals (consistent with repeat-spacer structure), then extends candidate arrays by searching for additional repeats flanking the initial seeds. Each candidate array is validated by checking that repeats are sufficiently similar and spacers fall within expected length ranges.

**Key assumptions:**
- Input sequences are prokaryotic or phage DNA
- CRISPR repeats are 23-47 nt in length (by default, minimum 27 nt)
- Arrays contain at least 3 repeats (by default; minimum 2 allowed)
- Spacers between repeats are 26-50 nt

**Limitations:**
- May miss degenerate or highly diverged CRISPR arrays
- Cannot distinguish active from inactive (degraded) arrays
- Does not identify Cas genes or system type -- use complementary tools for full CRISPR-Cas annotation
- Very short arrays (2 repeats) have higher false-positive rates

**Computational requirements:**
- **Hardware:** CPU only; no GPU required
- **Runtime:** Seconds for typical bacterial genomes (~5 Mb)
- **Scalability:** Processes sequences independently; scales linearly with input size

## Important Parameters

**Input parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `List[str]` | *Required* | Nucleotide sequence(s) to search for CRISPR arrays |
| `sequence_ids` | `Optional[List[str]]` | `None` | Optional sequence identifiers (defaults to seq_0, seq_1, ...) |

**Configuration parameters:**

| Parameter | Type | Default | Sweep Range | Description |
|-----------|------|---------|-------------|-------------|
| `min_num_repeats` | `int` | `3` | `2 - 5` | Minimum number of repeats required for a CRISPR array |
| `min_repeat_length` | `int` | `27` | `20 - 35` | Minimum length of a repeat sequence in nucleotides |

**Parameters to prioritize for sweeps:**
1. **`min_num_repeats`**: Lowering to 2 increases sensitivity but also false positives. Default of 3 balances sensitivity and specificity for most applications.
2. **`min_repeat_length`**: Lowering below 23 may detect non-CRISPR tandem repeats. Increasing above 30 may miss arrays with shorter repeats.

---

**Output specification:**

```python
# Return type: MincedOutput
{
    "results": [
        {
            "sequence_id": str,
            "crispr_arrays": [
                {
                    "repeats_and_spacers": [
                        {
                            "position": int,       # Genomic position
                            "repeat": str,         # Repeat sequence
                            "spacer": str | None,  # Spacer (None for last repeat)
                            "repeat_length": int,
                            "spacer_length": int | None,
                        },
                        ...
                    ]
                },
                ...
            ]
        },
        ...
    ]
}
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `results` | `List[MincedSequenceResult]` | Per-sequence CRISPR array detection results |
| `sequence_id` | `str` | Identifier for the input sequence |
| `crispr_arrays` | `List[CrisprArray]` | CRISPR arrays detected in this sequence |
| `repeats_and_spacers` | `List[CrisprRepeatSpacer]` | Ordered repeat-spacer units in each array |
| `position` | `int` | Genomic position of each repeat |
| `repeat` / `spacer` | `str` | The actual repeat and spacer sequences |

**Properties:**
- `MincedSequenceResult.has_crispr`: Whether any CRISPR arrays were found
- `MincedSequenceResult.num_arrays`: Number of arrays detected
- `CrisprArray.num_repeats`: Number of repeats in the array
- `CrisprArray.spacers`: List of spacer sequences

**Supported export formats:** `csv`, `json`

**Thresholds & decision boundaries:**
- **Typical CRISPR array**: 3+ repeats of 23-47 nt, spacers of 26-50 nt
- **High-confidence array**: >=4 repeats with consistent repeat lengths (+/-2 nt)
- **Questionable array**: 2 repeats only, or highly variable repeat lengths

## Best Practices & Gotchas

**Parameter tuning:**
- **`min_num_repeats`**:
  - Default (3): Good balance for most genomes
  - Lower (2): Use when searching for partial or degraded arrays; expect more false positives
  - Higher (4-5): Use for high-confidence detection only

**Common mistakes:**
1. **Expecting Cas gene annotation**: MinCED only detects the CRISPR array (repeats + spacers). It does not identify Cas proteins or classify the CRISPR system type. Use Prodigal + PyHMMER for Cas gene identification.
2. **Using protein sequences as input**: MinCED operates on nucleotide sequences only.
3. **Interpreting spacer count as immunity breadth**: Spacers may target the same phage or be inactive remnants.

**Tips for optimal results:**
- For generated sequences, run MinCED as a first-pass filter to confirm CRISPR array presence before more expensive analyses
- Cross-reference detected spacers with phage databases to identify targets
- Use `sequence_ids` to maintain traceability in batch processing

**Edge cases to watch for:**
- Tandem repeats that are not CRISPR arrays may be detected with low `min_repeat_length`
- Very long sequences (>10 Mb) may take longer but should complete without issues
- Sequences with no CRISPR arrays return results with `has_crispr=False` and empty `crispr_arrays`

## References

**Primary publication:**
- Bland, C., Ramsey, T.L., Sabree, F., Lowe, M., Brown, K., Kyrpides, N.C. & Hugenholtz, P. (2007). "CRISPR Recognition Tool (CRT): a tool for automatic detection of clustered regularly interspaced palindromic repeats." *BMC Bioinformatics* 8:209. [DOI: 10.1186/1471-2105-8-209](https://doi.org/10.1186/1471-2105-8-209)
- Summary: Describes the CRT algorithm that MinCED is based on, which uses a k-mer seed-and-extend approach to identify CRISPR repeat-spacer arrays in genomic sequences.

**Implementation:**
- GitHub: [https://github.com/ctSkennerton/minced](https://github.com/ctSkennerton/minced)

## Quick Start Examples

**Example 1: Detect CRISPR arrays in a sequence**
```python
from bio_programming_tools.tools.gene_annotation.minced import (
    run_minced, MincedInput, MincedConfig
)

inputs = MincedInput(
    sequences=["ATGCGT...your_genomic_sequence...GCATCG"],
    sequence_ids=["my_genome"],
)
config = MincedConfig()

result = run_minced(inputs, config)

for seq_result in result.results:
    print(f"{seq_result.sequence_id}: {seq_result.num_arrays} CRISPR arrays found")
    for i, array in enumerate(seq_result.crispr_arrays):
        print(f"  Array {i+1}: {array.num_repeats} repeats")
        for rs in array.repeats_and_spacers:
            print(f"    Repeat at {rs.position}: {rs.repeat[:20]}...")
            if rs.spacer:
                print(f"    Spacer: {rs.spacer[:20]}...")
```

**Example 2: Batch processing with sensitive settings**
```python
from bio_programming_tools.tools.gene_annotation.minced import (
    run_minced, MincedInput, MincedConfig
)

inputs = MincedInput(
    sequences=[seq1, seq2, seq3],
    sequence_ids=["genome_A", "genome_B", "genome_C"],
)
config = MincedConfig(
    min_num_repeats=2,      # More sensitive
    min_repeat_length=23,   # Allow shorter repeats
)

result = run_minced(inputs, config)
print(f"{result.num_sequences_with_crispr} of {len(result.results)} sequences have CRISPR arrays")
```

## Related Tools

**Tools often used together:**
- **`crispr-tracr`**: Predict tracrRNA sequences in CRISPR loci (run after MinCED confirms array presence)
- **`prodigal`**: Find ORFs near CRISPR arrays to identify Cas genes
- **`pyhmmer-hmmsearch`**: Search for Cas protein domains using HMM profiles
- **`local-blast`**: Search spacer sequences against phage databases

**Alternative tools (similar function):**
- **CRISPRCasFinder**: More comprehensive (detects Cas genes too) but web-based
- **PILER-CR**: Alternative local CRISPR finder; MinCED is generally more sensitive for short arrays

---
**Maintenance notes:**
- Last updated: 2025-06-01
