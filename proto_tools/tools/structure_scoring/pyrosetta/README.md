<a href="https://bio-pro.mintlify.app/tools/structure-scoring/pyrosetta"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# PyRosetta Scoring Functions

> [!IMPORTANT]
> **License:** PyRosetta is distributed under the [Rosetta Software License](https://www.rosettacommons.org/software/license-and-download). Free for academic and non-commercial use. Commercial users must obtain a license from [UW CoMotion](https://els2.comotion.uw.edu/product/pyrosetta). By using this tool, you accept these terms.

> [!IMPORTANT]
> NOTE: This README is placeholder text! We need to review and update

## Overview

PyRosetta provides physics-based scoring of protein structures using the Rosetta molecular modeling suite. Three operations are available: Spatial Aggregation Propensity (SAP) scoring, Solvent Accessible Surface Area (SASA) computation, and full Rosetta energy scoring with optional FastRelax.

## Background

**Spatial Aggregation Propensity (SAP)** quantifies how much hydrophobic surface area is exposed on a protein. Proteins with large patches of exposed hydrophobicity are prone to [aggregation](https://en.wikipedia.org/wiki/Protein_aggregation) -- a major concern in therapeutic protein development. SAP computes a per-atom hydrophobicity score weighted by solvent exposure, then sums across the surface. Higher SAP scores indicate greater aggregation risk.

**Solvent Accessible Surface Area (SASA)** measures the total surface area of a protein that is accessible to solvent molecules (modeled as a spherical probe, typically 1.4 A radius for water). SASA is fundamental to understanding protein folding thermodynamics: buried residues contribute to the [hydrophobic core](https://en.wikipedia.org/wiki/Hydrophobic_core), while exposed residues interact with solvent and binding partners. Per-residue SASA values reveal which positions are buried vs. solvent-exposed.

**Rosetta Energy Scoring** evaluates protein structures using a physics-based [energy function](https://en.wikipedia.org/wiki/Force_field_(chemistry)) that combines van der Waals interactions, electrostatics, hydrogen bonding, solvation, and backbone torsion preferences. The score is reported in Rosetta Energy Units (REU). Lower (more negative) total energies indicate more favorable structures. [FastRelax](https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover) is optionally applied before scoring to resolve steric clashes and backbone strain.

## Tool Catalog

| Tool Key | Label | Description |
|----------|-------|-------------|
| `pyrosetta-sap` | PyRosetta SAP Score | Compute Spatial Aggregation Propensity scores |
| `pyrosetta-sasa` | PyRosetta SASA | Compute Solvent Accessible Surface Area (total and per-residue) |
| `pyrosetta-energy` | PyRosetta Energy Score | Compute Rosetta energy scores with optional FastRelax |

## Execution Modes

| Mode | Backend | Device |
|------|---------|--------|
| Standalone env | `ToolInstance("pyrosetta")` running `standalone/inference.py` | CPU only |

All three tools share a single standalone micromamba environment with PyRosetta installed. PyRosetta initialization adds a few seconds of overhead on the first call; subsequent calls within a persistent `ToolInstance` skip this.

## How It Works

**SAP (`pyrosetta-sap`)**: Loads the structure into a Rosetta Pose, then computes per-atom SAP scores using Rosetta's `core.pack.guidance_scoreterms.sap` module. The result is a single scalar SAP score per structure.

**SASA (`pyrosetta-sasa`)**: Loads the structure into a Rosetta Pose, then runs `SasaCalc` with a configurable probe radius. Returns both the total SASA and a per-residue breakdown with chain, residue index, residue name, and SASA value.

**Energy (`pyrosetta-energy`)**: Loads the structure, optionally applies FastRelax (constrained to starting coordinates by default), then scores with the specified score function (default `ref2015`). Returns the total energy, a breakdown by score term (fa_atr, fa_rep, fa_sol, etc.), and per-residue total energies.

## Input Parameters

All three tools accept the same input structure:

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `list[ScoringStructureInput]` | Protein structures to score, each with optional `chain_ids` for chain selection. Accepts PDB file paths, PDB content strings, Structure objects, or dicts with `structure` and `chain_ids` keys. A single input is automatically wrapped in a list. |

## Configuration

### `pyrosetta-sap`

No additional configuration parameters. Uses `PyRosettaSAPConfig` with defaults only.

### `pyrosetta-sasa`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `probe_radius` | `float` | `1.4` | > 0.0 | Solvent probe radius in Angstroms. Standard water probe is 1.4 A. |

### `pyrosetta-energy`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scorefxn` | `str` | `"ref2015"` | -- | Rosetta score function name (e.g., `ref2015`, `beta_nov16`, `ref2015_cart`) |
| `relax` | `bool` | `True` | -- | Run FastRelax before scoring to resolve clashes and strain |
| `relax_cycles` | `int` | `5` | 1-15 | Number of FastRelax cycles (more = better convergence, slower) |
| `constrain_to_start` | `bool` | `True` | -- | Constrain relaxation to starting coordinates to prevent large deviations |

## Output Specification

### `pyrosetta-sap`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[SAPResult]` | One per input structure |

Each `SAPResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `sap_score` | `float` | SAP score. Higher values indicate more aggregation-prone surface hydrophobicity. |

### `pyrosetta-sasa`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[SASAResult]` | One per input structure |

Each `SASAResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `total_sasa` | `float` | Total SASA in Angstroms squared |
| `per_residue` | `list[ResidueSASA]` | Per-residue SASA breakdown |

Each `ResidueSASA` contains:

| Field | Type | Description |
|-------|------|-------------|
| `chain_id` | `str` | Chain identifier |
| `residue_index` | `int` | 1-indexed residue position |
| `residue_name` | `str` | Three-letter residue code |
| `sasa` | `float` | SASA in Angstroms squared |

### `pyrosetta-energy`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[EnergyResult]` | One per input structure |

Each `EnergyResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `total_energy` | `float` | Total Rosetta energy in REU |
| `energy_terms` | `dict[str, float]` | Breakdown by score term (fa_atr, fa_rep, fa_sol, etc.) |
| `per_residue` | `list[ResidueEnergy]` | Per-residue energy breakdown |
| `relaxed` | `bool` | Whether FastRelax was applied before scoring |

Each `ResidueEnergy` contains:

| Field | Type | Description |
|-------|------|-------------|
| `chain_id` | `str` | Chain identifier |
| `residue_index` | `int` | 1-indexed residue position |
| `residue_name` | `str` | Three-letter residue code |
| `total_energy` | `float` | Total residue energy in REU |

Export formats: `csv`, `json` (all three tools).

## Interpreting Results

**SAP scores:**
- SAP is unitless and relative. Lower is better for therapeutic developability.
- Typical well-behaved antibodies have SAP scores below ~100. Values above ~150 suggest significant aggregation risk.
- Compare SAP across design variants rather than relying on absolute thresholds, since scores depend on protein size and topology.

**SASA values:**
- Total SASA scales with protein size. Normalize by residue count for cross-protein comparisons.
- Per-residue SASA near 0 indicates a fully buried residue. Values above ~100 A^2 indicate significant solvent exposure.
- Hydrophobic residues with high SASA are candidates for aggregation hotspots or redesign targets.

**Energy scores:**
- Rosetta energies are in REU (not kcal/mol). More negative = more favorable.
- Absolute energies are not meaningful across different proteins. Compare variants of the same protein.
- Key energy terms: `fa_atr` (attractive van der Waals), `fa_rep` (repulsive van der Waals, penalizes clashes), `fa_sol` (solvation), `hbond_*` (hydrogen bonds).
- High `fa_rep` values indicate steric clashes -- run with `relax=True` to resolve these before interpreting other terms.
- Per-residue energies identify problematic positions (e.g., residues with high positive energy are under strain).
- **Chain selection filters display, not computation.** When you set `chain_ids` on an energy input, `total_energy` and `energy_terms` are still computed on the full pose — the selection only filters which residues appear in `per_residue`. Each per-residue energy reflects that residue's contribution *in the context of the full complex* (including pair interactions with the un-selected chains), not the chain's energy in isolation. To score a chain as if it were isolated, extract it into its own Structure first and score it separately.

## Quick Start Examples

**Compute SAP scores:**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    run_pyrosetta_sap, PyRosettaSAPInput,
)

result = run_pyrosetta_sap(PyRosettaSAPInput(inputs=["/path/to/protein.pdb"]))

for r in result.results:
    print(f"SAP score: {r.sap_score:.2f}")
```

**Compute SASA with custom probe radius:**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    run_pyrosetta_sasa, PyRosettaSASAInput, PyRosettaSASAConfig,
)

result = run_pyrosetta_sasa(
    PyRosettaSASAInput(inputs=["/path/to/protein.pdb"]),
    PyRosettaSASAConfig(probe_radius=1.4),
)

for r in result.results:
    print(f"Total SASA: {r.total_sasa:.1f} A^2")
    for res in r.per_residue[:5]:
        print(f"  {res.chain_id}:{res.residue_name}{res.residue_index} = {res.sasa:.1f} A^2")
```

**Energy scoring with FastRelax:**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    run_pyrosetta_energy, PyRosettaEnergyInput, PyRosettaEnergyConfig,
)

result = run_pyrosetta_energy(
    PyRosettaEnergyInput(inputs=["/path/to/protein.pdb"]),
    PyRosettaEnergyConfig(scorefxn="ref2015", relax=True, relax_cycles=5),
)

for r in result.results:
    print(f"Total energy: {r.total_energy:.1f} REU (relaxed={r.relaxed})")
    print(f"  fa_atr: {r.energy_terms.get('fa_atr', 0):.1f}")
    print(f"  fa_rep: {r.energy_terms.get('fa_rep', 0):.1f}")
    print(f"  fa_sol: {r.energy_terms.get('fa_sol', 0):.1f}")
```

## Best Practices & Gotchas

- **`linux-aarch64` runs a stale 2023.11 build.** The `conda.rosettacommons.org/linux-aarch64` channel has not been updated since March 2023 — the newest available pyrosetta on aarch64 is `2023.11` (Python 3.10), while `linux-64`, `osx-arm64`, and `osx-x86_64` ship the current `2026.06` builds. Scoring functions, FastRelax, and SAP have evolved over the past ~3 years, so values computed on `linux-aarch64` may differ slightly from values computed on other platforms. The setup script prints a warning banner on aarch64; reproduce final/published numbers on x86_64 or macOS.
- **Always relax before energy scoring.** Raw PDB structures (from X-ray or prediction) typically have steric clashes that produce extremely high `fa_rep` values. The default `relax=True` resolves this. Only set `relax=False` if you have already relaxed the structure.
- **PyRosetta initialization overhead.** The first call in a session takes a few seconds to initialize PyRosetta. Use `ToolInstance.persist()` for batch workloads to amortize this cost.
- **Chain labels round-trip through the original format.** PDB stores chain IDs as a single character, while mmCIF permits arbitrary-length labels (e.g. `"Heavy"`, `"Light"`, or PDB bundle names like `"AA"`). You can pass either format directly, reference chains by their native labels in `chain_ids`, and read the same labels back in `per_residue[i].chain_id`. The tool internally shortens multi-character chain IDs to fit PDB format when dispatching to PyRosetta and restores the originals in the output, so the conversion is invisible. Structures with more than 62 unique chain labels cannot be represented in PDB and are rejected up front with a clear error.
- **SAP is size-dependent.** Larger proteins naturally have higher SAP scores. Compare SAP across variants of the same protein, not across proteins of different sizes.
- **Energy comparisons require the same score function.** Never compare REU values computed with different `scorefxn` settings (e.g., `ref2015` vs. `beta_nov16`).
- **Relax cycles trade off accuracy vs. speed.** The default 5 cycles is a good balance. Increase to 10-15 for publication-quality results. Reduce to 1-2 for rapid screening.
- **Constrain to start coordinates.** The default `constrain_to_start=True` prevents FastRelax from drastically altering the structure. Disable only if you want unconstrained relaxation (e.g., to find the nearest energy minimum).
- **Coordinates are 1-indexed.** Per-residue output uses 1-indexed residue positions consistent with PDB numbering.

## References

- Chaudhury, S., Lyskov, S., & Gray, J.J. (2010). PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta. *Bioinformatics*, 26(5), 689-691. DOI: [10.1093/bioinformatics/btq007](https://doi.org/10.1093/bioinformatics/btq007)
- Alford, R.F., Leaver-Fay, A., Jeliazkov, J.R., et al. (2017). The Rosetta all-atom energy function for macromolecular modeling and design. *J. Chem. Theory Comput.*, 13(6), 3031-3048. DOI: [10.1021/acs.jctc.7b00125](https://doi.org/10.1021/acs.jctc.7b00125)
- RosettaCommons: https://www.rosettacommons.org/
- PyRosetta documentation: https://www.pyrosetta.org/

## Related Tools

**Often used together:**
- **Structure prediction** (`esmfold-prediction`, `alphafold3-prediction`) -- Generate structures to score
- **Inverse folding** (`proteinmpnn-sample`, `ligandmpnn-sample`) -- Design sequences, then score the resulting structures
- **Structure metrics** (`structure-metrics`) -- Compute geometric quality metrics (longest alpha helix, radius of gyration)
- **TM-align** (`tmalign-align`) -- Structural alignment and TM-score comparison

**Alternatives:**
- **Structure metrics** -- For coarse geometric filters (helix length, gyration radius) without physics-based energy
- **ESM2 scoring** (`esm2-score`) -- Learned sequence-based fitness scoring (complementary to physics-based scoring)
