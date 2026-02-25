# USalign Structure Alignment

## Overview

USalign (Universal Structure alignment) extends TMalign to support monomers, multimers, and nucleic acid structures (Zhang et al., 2022). It aligns two structures using `-mm 1 -ter 1` flags for multimer support and returns TM-scores normalized by each structure's length.

- **Tool key**: `usalign-alignment`
- **Input**: Two PDB text blobs (query and reference structures)
- **Output**: Two TM-scores (normalized by Structure 1 and Structure 2 lengths)
- **Execution**: CPU only (compiled C++ binary)

## When to Use This Tool

**Use when you need to:**
- Compare multimeric protein complexes (oligomers, heteromers)
- Align structures containing both protein and nucleic acid chains
- Compare any pair of macromolecular structures universally

**Do NOT use when you need to:**
- Compare two simple monomers -- TMalign (`tmalign-alignment`) is faster and equivalent
- Compute RMSD -- use PyMOL's cealign via the `structure-rmsd` constraint
- Predict structures from sequence -- use ESMFold, AlphaFold3, etc.

## Biological Background

USalign generalizes TM-score alignment to handle:
- **Monomers**: Standard pairwise structure alignment (equivalent to TMalign)
- **Multimers**: Aligns oligomeric complexes by considering all chains jointly
- **Nucleic acids**: Aligns RNA/DNA structures using nucleotide-aware scoring
- **Mixed complexes**: Handles protein-nucleic acid complexes

TM-score interpretation is the same as TMalign:
- **TM-score > 0.5**: Same fold / similar complex architecture
- **TM-score > 0.17**: Random similarity baseline
- **TM-score = 1.0**: Identical structures

## Input Parameters

| Field | Type | Description |
|-------|------|-------------|
| `pdb_text_1` | `str` | PDB content of structure 1 (query) |
| `pdb_text_2` | `str` | PDB content of structure 2 (reference) |

## Configuration

USalign uses default configuration inherited from `BaseConfig`. No tool-specific parameters.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `tm_score_structure_1` | `float` | TM-score normalized by length of Structure 1 (query) |
| `tm_score_structure_2` | `float` | TM-score normalized by length of Structure 2 (reference) |

Export format: `json`

## Quick Start

```python
from bio_programming_tools.tools.structure_alignment.usalign import (
    USalignInput, USalignConfig, run_usalign,
)

inputs = USalignInput(pdb_text_1=query_pdb, pdb_text_2=reference_pdb)
result = run_usalign(inputs, USalignConfig())

print(f"TM-score (norm by query):     {result.tm_score_structure_1:.3f}")
print(f"TM-score (norm by reference): {result.tm_score_structure_2:.3f}")
```

## Interpreting Results

- The two TM-scores differ because they are normalized by different structure lengths
- For multimers, the alignment considers all chains jointly
- For comparing a candidate to a fixed target, use the score normalized by the target length

## References

- Zhang, C., Shine, M., Pyle, A.M. & Zhang, Y. (2022). US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes. *Nature Methods*, 19(9), 1109-1115. DOI: 10.1038/s41592-022-01585-1
- GitHub: https://github.com/pylelab/USalign

## Related Tools

- **TMalign** (`tmalign-alignment`) -- Faster monomer-only alignment
- **Structure TM-score constraint** (`structure-tmscore`) -- Optimization constraint using TM-score with automatic tool selection
