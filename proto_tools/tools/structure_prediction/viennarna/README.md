<a href="https://bio-pro.mintlify.app/tools/structure-prediction/viennarna"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ViennaRNA

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

ViennaRNA is a fast RNA secondary structure prediction tool that uses thermodynamic parameters to compute the [minimum free energy](https://en.wikipedia.org/wiki/Nucleic_acid_thermodynamics) (MFE) structure from nucleotide sequences.

## Background

**What does this tool measure/predict?**
ViennaRNA predicts the secondary structure of RNA molecules; the pattern of [Watson-Crick](https://en.wikipedia.org/wiki/Watson%E2%80%93Crick_base_pair) and wobble base pairs that form within a single RNA strand. It outputs the structure in dot-bracket notation along with the minimum free energy (MFE).

**Why is this important?**
RNA secondary structure is fundamental to understanding RNA function:
- **mRNA stability:** Secondary structures in UTRs affect translation efficiency and mRNA half-life
- **Regulatory elements:** Riboswitches, thermosensors, and other regulatory RNAs depend on specific structures
- **Guide RNA design:** CRISPR gRNAs and other synthetic RNAs require specific structural features
- **RNA therapeutics:** siRNA, ASO, and mRNA vaccine design all consider secondary structure
- **Viral RNA:** Understanding viral RNA structures aids in understanding replication and drug targeting

**Scientific foundation:**
ViennaRNA uses a dynamic programming algorithm based on the [nearest-neighbor](https://en.wikipedia.org/wiki/Nucleic_acid_thermodynamics#Nearest-neighbor_method) thermodynamic model. Energy parameters (Turner 2004 for RNA, Mathews 2004 for DNA) describe the free energy contributions of:
- **Base pair stacking:** The stabilizing interactions between adjacent base pairs
- **Loop energies:** Penalties for hairpin loops, internal loops, bulges, and multiloops
- **Dangling ends:** Contributions from unpaired nucleotides adjacent to helices

The algorithm finds the structure with the lowest total free energy (most stable).

## Tools

### ViennaRNA Secondary Structure Prediction (`viennarna-prediction`)

Predict RNA secondary structures using ViennaRNA's MFE algorithm.

This function uses ViennaRNA's minimum free energy (MFE) algorithm to
predict the most thermodynamically stable secondary structure for each
input RNA sequence.

## How It Works

**Method overview:**
ViennaRNA implements the Zuker algorithm with McCaskill's partition function:
1. **Dynamic programming:** Fills matrices for all possible substructures
2. **Traceback:** Reconstructs the minimum free energy structure
3. **Energy model:** Uses nearest-neighbor parameters for thermodynamic calculations

**Key assumptions:**
- RNA folds to thermodynamic equilibrium (MFE structure)
- Standard Watson-Crick and GU wobble pairs
- No pseudoknots in the default algorithm
- No tertiary interactions or long-range contacts

**Limitations:**
- **No pseudoknots:** Default algorithm excludes pseudoknotted structures
- **Single sequence:** Predicts one structure per sequence (not multi-strand)
- **Equilibrium assumption:** May not reflect kinetically trapped structures
- **Parameter uncertainty:** Energy parameters have experimental error (~1 kcal/mol)

**Computational requirements:**
- **Time complexity:** O(n^3) for sequence length n
- **Space complexity:** O(n^2)
- **Runtime:** ~1 second for 1000 nt on modern CPU
- **No GPU required:** Runs efficiently on CPU

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `List[str]` | *required* | List of RNA/DNA sequences to fold |
| `temperature` | `float` | `37.0` | Temperature in Celsius; affects thermodynamic stability |
| `use_dna_params` | `bool` | `False` | Use DNA parameters instead of RNA |
| `no_lonely_pairs` | `bool` | `False` | Disallow isolated base pairs |

## Configuration

### Parameter Guides

| Parameter | Sweep Range | Notes |
|-----------|-------------|-------|
| `temperature` | `25 - 42` | Lower temperatures stabilize structures; higher destabilize |
| `no_lonely_pairs` | `True, False` | Can reduce artifacts from isolated base pairs |

### Sweep Priorities

1. **`temperature`**: Most impactful for thermosensor designs or temperature-sensitive applications. Sweep 25C, 30C, 37C, 42C.
2. **`no_lonely_pairs`**: Can reduce artifacts; try both True and False for final designs.

## Output Specification

```python
# Return type: ViennaRNAOutput
ViennaRNAOutput(
    results: List[RNAFoldResult],
    metadata: dict
)

RNAFoldResult(
    sequence: str,           # Input sequence (normalized to RNA)
    structure: str,          # Dot-bracket notation
    mfe: float,              # Minimum free energy (kcal/mol)
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `structure` | `str` | N/A | '.' = unpaired, '(' = 5' pair, ')' = 3' pair |
| `mfe` | `float` | typically -50 to 0 | More negative = more stable; depends on length |

## Interpreting Results

**Thresholds & decision boundaries:**
MFE interpretation depends heavily on sequence length. Use normalized metrics:
- **Per-nucleotide MFE:** `mfe / len(sequence)` should be ~ -0.3 to -0.5 kcal/mol/nt for stable structures
- **Unstable regions:** Per-nt MFE > -0.2 suggests less structured regions
- **Highly stable:** Per-nt MFE < -0.5 indicates very stable structure (GC-rich)

**Tips for interpreting output:**
- Always normalize MFE by sequence length for cross-sequence comparisons
- Similar MFE values (within 1-2 kcal/mol) represent similar stability: don't over-interpret small differences
- The MFE structure is not always the biologically relevant structure; compare suboptimal structures when possible
- Dot-bracket notation is 2D only: it does not encode 3D spatial information

## Quick Start Examples

```python
from proto_tools.tools.structure_prediction.viennarna import (
    run_viennarna_fold,
    ViennaRNAFoldInput,
    ViennaRNAFoldConfig,
)

# Basic RNA folding
inputs = ViennaRNAFoldInput(
    sequences=["GGGAAACCC", "GGGUUUCCC"]
)
config = ViennaRNAFoldConfig(temperature=37.0)
result = run_viennarna_fold(inputs, config)

for fold in result.results:
    print(f"Sequence:  {fold.sequence}")
    print(f"Structure: {fold.structure}")
    print(f"MFE:       {fold.mfe:.2f} kcal/mol")
    print(f"Per-nt:    {fold.mfe / len(fold.sequence):.3f} kcal/mol/nt")
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`temperature`**:
  - Use 37C for physiological conditions
  - Lower temperatures (25C) stabilize structures
  - Higher temperatures (42C) destabilize; useful for thermosensor design
- **`use_dna_params`**:
  - Set True only for DNA sequences
  - RNA and DNA have different stacking parameters

**Common mistakes:**
1. **Ignoring sequence context:** The folding of a region depends on flanking sequences
2. **Over-interpreting MFE:** Similar MFE values (within 1-2 kcal/mol) represent similar stability
3. **Forgetting T to U conversion:** Tool auto-converts T to U for RNA parameters
4. **Expecting 3D information:** Dot-bracket is 2D only

**Tips for optimal results:**
- Include flanking sequences: Add ~50 nt context around regions of interest
- Compare suboptimal structures: MFE isn't always the biologically relevant structure
- Validate with experiments: Use SHAPE, DMS-MaPseq, or other probing methods
- Consider kinetic folding: For co-transcriptional folding, order of synthesis matters

**Edge cases to watch for:**
- **Very short sequences (<15 nt):** May have no stable structure (all '.')
- **Highly repetitive sequences:** May have many equivalent structures
- **Very long sequences (>10,000 nt):** Will be slow; consider windowed folding
- **GC-rich sequences:** Will be very stable; may trap into non-biological structures

## References

**Primary publications:**
- Lorenz et al. (2011). "ViennaRNA Package 2.0". *Algorithms for Molecular Biology*, 6:26. [DOI: 10.1186/1748-7188-6-26](https://doi.org/10.1186/1748-7188-6-26)
- Hofacker (2003). "Vienna RNA secondary structure server". *Nucleic Acids Research*, 31(13), 3429-3431.

**Energy parameters:**
- Turner & Mathews (2010). "NNDB: the nearest neighbor parameter database for predicting stability of nucleic acid secondary structure". *Nucleic Acids Research*, 38(suppl_1), D532-D538.
- Mathews et al. (2004). "Incorporating chemical modification constraints into a dynamic programming algorithm for prediction of RNA secondary structure". *PNAS*, 101(19), 7287-7292.

**Implementation:**
- GitHub: [https://github.com/ViennaRNA/ViennaRNA](https://github.com/ViennaRNA/ViennaRNA)
- Documentation: [https://viennarna.readthedocs.io](https://viennarna.readthedocs.io)
- PyPI: [https://pypi.org/project/ViennaRNA/](https://pypi.org/project/ViennaRNA/)

## Related Tools

**Tools often used together:**
- **`evo2`**: Generate RNA sequences with Evo, then predict secondary structures with ViennaRNA

**Alternative tools:**
- **`boltz2-prediction`**: For 3D RNA structure prediction when 2D folding is insufficient
- **`alphafold3-prediction`**: For RNA-protein complex 3D structure prediction
