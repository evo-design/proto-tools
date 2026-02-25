# TMalign Structure Alignment

## Overview

TMalign performs pairwise protein structure alignment using the TM-score metric (Zhang & Skolnick, 2005). It aligns two protein structures and returns TM-scores normalized by the length of each chain, providing a length-independent measure of structural similarity.

- **Tool key**: `tmalign-alignment`
- **Input**: Two PDB text blobs (query and reference structures)
- **Output**: Two TM-scores (normalized by Chain 1 and Chain 2 lengths)
- **Execution**: CPU only (compiled C++ binary)

## When to Use This Tool

**Use when you need to:**
- Compare two monomeric protein structures for topological similarity
- Quantify structural similarity with a length-independent metric (TM-score)
- Evaluate how well a predicted structure matches a reference/target

**Do NOT use when you need to:**
- Compare multimeric complexes -- use USalign (`usalign-alignment`) instead
- Compute RMSD -- use PyMOL's cealign via the `structure-rmsd` constraint
- Predict structures from sequence -- use ESMFold, AlphaFold3, etc.

## Biological Background

TM-score (Template Modeling score) measures the topological similarity of two protein structures. Unlike RMSD, TM-score is length-independent and falls in the range (0, 1]:
- **TM-score > 0.5**: Generally indicates proteins share the same fold
- **TM-score > 0.17**: Expected for randomly related proteins
- **TM-score = 1.0**: Perfect structural match

TMalign uses a heuristic alignment algorithm that optimizes the TM-score directly, making it more robust than RMSD-based methods for comparing structures of different lengths.

## Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `pdb_text_1` | `str` | PDB content of structure 1 (query) |
| `pdb_text_2` | `str` | PDB content of structure 2 (reference) |

## Configuration

TMalign uses default configuration inherited from `BaseConfig`. No tool-specific parameters.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `tm_score_chain_1` | `float` | TM-score normalized by length of Chain 1 (query) |
| `tm_score_chain_2` | `float` | TM-score normalized by length of Chain 2 (reference) |

Export format: `json`

## Quick Start

```python
from bio_programming_tools.tools.structure_alignment.tmalign import (
    TMalignInput, TMalignConfig, run_tmalign,
)

inputs = TMalignInput(pdb_text_1=query_pdb, pdb_text_2=reference_pdb)
result = run_tmalign(inputs, TMalignConfig())

print(f"TM-score (norm by query):     {result.tm_score_chain_1:.3f}")
print(f"TM-score (norm by reference): {result.tm_score_chain_2:.3f}")
```

## Interpreting Results

- The two TM-scores differ because they are normalized by different chain lengths
- For comparing a candidate to a fixed target, use the score normalized by the target length
- TM-score > 0.5 is a strong indicator of the same fold topology

## References

- Zhang, Y. & Skolnick, J. (2005). TM-align: a protein structure alignment algorithm based on the TM-score. *Nucleic Acids Research*, 33(7), 2302-2309. DOI: 10.1093/nar/gki524
- GitHub: https://github.com/pylelab/USalign

## Related Tools

- **USalign** (`usalign-alignment`) -- Universal alignment supporting multimers
- **Structure TM-score constraint** (`structure-tmscore`) -- Optimization constraint using TM-score with automatic tool selection
