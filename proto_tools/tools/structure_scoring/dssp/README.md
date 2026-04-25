<a href="https://bio-pro.mintlify.app/tools/structure-scoring/dssp"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# DSSP Secondary Structure

## Overview

DSSP assigns protein secondary structure from 3D coordinates using hydrogen-bond and geometry rules. It is an assignment program for structures with coordinates, not a sequence-based secondary-structure predictor. This module exposes `dssp-secondary-structure`, which reports helix, sheet, and loop percentages for a selected chain in each input structure.

- **Tool key**: `dssp-secondary-structure`
- **Input**: Protein `Structure` objects or structure paths with a `chain_id`
- **Output**: Per-chain `helix_pct`, `sheet_pct`, and `loop_pct`
- **Execution**: CPU only, local standalone environment via `ToolInstance`

## Background

**What does this tool assign?**
DSSP classifies each protein residue into secondary-structure states from atomic coordinates. This wrapper collapses those assignments into helix (`H`, `G`, `I`), sheet (`E`), and loop percentages for the selected chain.

**Why is this important?**
- Protein design: filter designs with undesired helix, sheet, or loop composition.
- Pipeline compatibility: use the DSSP executable through the standard Proto `ToolInstance` interface.
- Structural analysis: summarize secondary-structure content from experimentally solved or predicted structures.

**Scientific foundation:**
DSSP identifies recurring hydrogen-bonding and geometric patterns in protein backbones. Repeating turns are assigned as helices, repeating bridges form ladders/sheets, and other assigned states are treated here as loop/coil for coarse composition scoring.

**Primary use cases:**
- Match pipelines that require DSSP-backed secondary-structure assignment.
- Score designed binder chains for helix/sheet/loop composition.
- Compare one chain's assigned secondary-structure composition across a batch of structures.

## How It Works

**Method overview:**
The tool converts each input structure to PDB text, preserving a mapping from original chain labels to PDB-compatible single-character labels. The standalone runner writes each PDB to a temporary file, calls Biopython's `Bio.PDB.DSSP` wrapper against the DSSP executable, counts residue assignments for the selected chain, rounds percentages to two decimals, and removes the temporary file.

**Key assumptions:**
- Inputs contain protein backbone atoms for the selected chain.
- The selected chain exists in the input structure.
- PDB conversion can represent all chains with single-character labels.

**Limitations:**
- Reports coarse helix/sheet/loop percentages, not per-residue DSSP assignments.
- Counts `E` as sheet; DSSP `B`, `T`, `S`, unassigned states, and `P` from DSSP 4 are grouped as loop.
- Uses the first model parsed by Biopython for multi-model structures.

**Computational requirements:**
- **Hardware:** CPU only
- **Runtime:** Depends on structure size, file I/O, and external DSSP execution time
- **Scalability:** Batched inputs dispatch through one standalone DSSP worker

## Input Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `list[DSSPStructureInput]` | Structures and chains to analyze. A single structure input is auto-wrapped into a list. |
| `structure` | `Structure` | Protein structure object, structure path, or structure content accepted by `Structure`. |
| `chain_id` | `str` | Chain label to analyze. Defaults to `A`. |

## Configuration

No DSSP-specific configuration parameters are needed. Standard `BaseConfig` fields such as `timeout` and `verbose` are still available.

## Output Specification

| Field | Type | Units | Interpretation |
|-------|------|-------|----------------|
| `chain_id` | `str` | none | Original input chain label analyzed. |
| `helix_pct` | `float` | percent | Fraction of counted residues assigned DSSP `H`, `G`, or `I`. |
| `sheet_pct` | `float` | percent | Fraction of counted residues assigned DSSP `E`. |
| `loop_pct` | `float` | percent | Fraction of counted residues assigned any other DSSP state. |

Supported export formats: `csv`, `json`.

## Quick Start Examples

```python
from proto_tools.entities.structures import Structure
from proto_tools.tools.structure_scoring.dssp import (
    DSSPSecondaryStructureConfig,
    DSSPSecondaryStructureInput,
    DSSPStructureInput,
    run_dssp_secondary_structure,
)

result = run_dssp_secondary_structure(
    DSSPSecondaryStructureInput(
        inputs=[
            DSSPStructureInput(
                structure=Structure(structure="/path/to/complex.pdb"),
                chain_id="B",
            )
        ]
    ),
    DSSPSecondaryStructureConfig(),
)

metrics = result.results[0]
print(metrics.helix_pct, metrics.sheet_pct, metrics.loop_pct)
```

## Best Practices & Gotchas

- Use `chain_id` to score the designed or analyzed chain explicitly; the default is chain `A`.
- Provide a 3D coordinate structure. DSSP does not predict secondary structure from sequence alone.
- Use `structure-metrics` when you need lightweight in-process metrics without installing DSSP.
- Use PyRosetta structure-scoring tools for energy, SASA, SAP, relax, or interface metrics. DSSP only assigns secondary structure.
- Multi-character chain labels are mapped to PDB-compatible chain IDs before the standalone DSSP runner executes.

## References

- Hekkelman, M. L., Alvarez Salmoral, D., Perrakis, A. & Joosten, R. P. (2025). DSSP 4: FAIR annotation of protein secondary structure. *Protein Science*, 34(8), e70208. DOI: [10.1002/pro.70208](https://doi.org/10.1002/pro.70208)
- Kabsch, W. & Sander, C. (1983). Dictionary of protein secondary structure: Pattern recognition of hydrogen-bonded and geometrical features. *Biopolymers*, 22(12), 2577-2637. DOI: [10.1002/bip.360221211](https://doi.org/10.1002/bip.360221211)
- Biopython `Bio.PDB.DSSP` API documentation: [https://biopython.org/docs/dev/api/Bio.PDB.DSSP.html](https://biopython.org/docs/dev/api/Bio.PDB.DSSP.html)
- PDB-REDO `mkdssp` documentation: [https://github.com/PDB-REDO/dssp/blob/trunk/doc/mkdssp.md](https://github.com/PDB-REDO/dssp/blob/trunk/doc/mkdssp.md)
- PDB-REDO DSSP repository: [https://github.com/PDB-REDO/dssp](https://github.com/PDB-REDO/dssp)

## Related Tools

- `structure-metrics` - in-process secondary-structure and compactness metrics.
- `pyrosetta` - physics-based structure-scoring tools for energies, SASA, SAP, and relax workflows.
