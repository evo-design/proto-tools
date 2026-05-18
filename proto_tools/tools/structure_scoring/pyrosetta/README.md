<a href="https://bio-pro.mintlify.app/tools/structure-scoring/pyrosetta"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# PyRosetta Scoring Functions

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

> [!IMPORTANT]
> **License:** PyRosetta is distributed under the [Rosetta Software License](https://www.rosettacommons.org/software/license-and-download). Free for academic and non-commercial use. Commercial users must obtain a license from [UW CoMotion](https://els2.comotion.uw.edu/product/pyrosetta). By using this tool, you accept these terms.

## Overview

PyRosetta provides physics-based scoring of protein structures using the Rosetta molecular modeling suite. Five operations are available: Spatial Aggregation Propensity (SAP) scoring, Solvent Accessible Surface Area (SASA) computation, Rosetta energy scoring, standalone FastRelax that returns the relaxed structure, and interface analysis of two-chain complexes via `InterfaceAnalyzerMover`. The three scoring tools and the interface analyzer all expose an opt-in `pre_relax_structures` preprocess that calls `pyrosetta-relax` before scoring.

## Background

**Spatial Aggregation Propensity (SAP)** quantifies how much hydrophobic surface area is exposed on a protein. Proteins with large patches of exposed hydrophobicity are prone to [aggregation](https://en.wikipedia.org/wiki/Protein_aggregation) -- a major concern in therapeutic protein development. SAP computes a per-atom hydrophobicity score weighted by solvent exposure, then sums across the surface. Higher SAP scores indicate greater aggregation risk.

**Solvent Accessible Surface Area (SASA)** measures the total surface area of a protein that is accessible to solvent molecules (modeled as a spherical probe, typically 1.4 A radius for water). SASA is fundamental to understanding protein folding thermodynamics: buried residues contribute to the [hydrophobic core](https://en.wikipedia.org/wiki/Hydrophobic_core), while exposed residues interact with solvent and binding partners. Per-residue SASA values reveal which positions are buried vs. solvent-exposed.

**Rosetta Energy Scoring** evaluates protein structures using a physics-based [energy function](https://en.wikipedia.org/wiki/Force_field_(chemistry)) that combines van der Waals interactions, electrostatics, hydrogen bonding, solvation, and backbone torsion preferences. The score is reported in Rosetta Energy Units (REU). Lower (more negative) total energies indicate more favorable structures. [FastRelax](https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover) is optionally applied before scoring to resolve steric clashes and backbone strain.

**Interface Analysis** quantifies the quality of protein-protein interfaces using Rosetta's `InterfaceAnalyzerMover` plus auxiliary residue-composition and surface-layer analyses. Seven metrics are emitted per complex: shape complementarity (`interface_sc`, 0-1), interface hydrogen-bond count (`interface_hbonds`), binding ΔG (`interface_dG`, in REU), buried SASA (`interface_dSASA`, Å²), interface packing statistic (`interface_packstat`, 0-1), interface-residue apolar fraction (`interface_hydrophobicity`, %), and binder surface apolar fraction (`surface_hydrophobicity`, 0-1).

## Tools

### PyRosetta Energy Score (`pyrosetta-energy`)

Compute Rosetta energy scores using PyRosetta.

Scores each protein structure using the specified Rosetta score function.
To resolve steric clashes and strain before scoring, set
`config.pre_relax_structures=True` — the framework's `Config.preprocess`
hook then dispatches `pyrosetta-relax` and substitutes the relaxed
structures before this function runs.

Chain selection only filters the per-residue breakdown; the whole pose
is always scored, so `total_energy`/`energy_terms` are always the
full-complex values and selected-residue energies are in-complex
contributions, not isolated-chain energies. See :class:`PyRosettaEnergyMetrics`
for details.

### PyRosetta Interface Analyzer (`pyrosetta-interface-analyzer`)

Compute interface-quality metrics for two-chain complexes using PyRosetta.

Runs Rosetta's `InterfaceAnalyzerMover` to compute shape complementarity,
interface H-bond count, binding ΔG, buried SASA, and packing statistic;
computes `interface_hydrophobicity` from interface-residue AA composition
(binder residues with any atom within 4.0 Å of any target atom); and
computes `surface_hydrophobicity` from `LayerSelector(pick_surface=True)`
applied to the binder sub-pose.

The interface for each complex is defined by the `target_chain` and
`binder_chain` fields on the corresponding :class:`InterfaceStructureInput`.
Chain-label validity is enforced at input construction. To analyze an
already-relaxed pose, pass it in directly; to relax-then-analyze in one
dispatch, set `config.pre_relax_structures=True`.

### PyRosetta FastRelax (`pyrosetta-relax`)

Run FastRelax on protein structures and return the relaxed coordinates.

Designed for cofolding filter pipelines (Germinal-style binder design)
where downstream geometric / energetic gates need to operate on a
relaxed pose. The relaxed Structure chains cleanly into
`run_pyrosetta_energy`, `run_pyrosetta_sap`, `run_pyrosetta_sasa`,
or Structure-aware non-PyRosetta tools.

### PyRosetta SAP Score (`pyrosetta-sap`)

Compute Spatial Aggregation Propensity (SAP) scores using PyRosetta.

SAP quantifies the aggregation propensity of a protein's surface by
measuring exposed hydrophobicity. Higher scores indicate greater
aggregation risk. Per-residue contributions identify which residues
drive aggregation propensity.

Chain selection controls which residues are scored, while the full
structure is always used for SASA and burial context.

### PyRosetta SASA (`pyrosetta-sasa`)

Compute Solvent Accessible Surface Area (SASA) using PyRosetta.

Calculates total and per-residue SASA using the SasaCalc module with
a configurable probe radius. SASA measures the surface area of a protein
accessible to solvent molecules.

## Tool Catalog

| Tool Key | Label | Description |
|----------|-------|-------------|
| `pyrosetta-sap` | PyRosetta SAP Score | Compute Spatial Aggregation Propensity scores |
| `pyrosetta-sasa` | PyRosetta SASA | Compute Solvent Accessible Surface Area (total and per-residue) |
| `pyrosetta-energy` | PyRosetta Energy Score | Compute Rosetta energy scores with optional FastRelax |
| `pyrosetta-relax` | PyRosetta FastRelax | Run FastRelax and return the relaxed `Structure` plus its total score, for chaining into downstream filters or scorers |
| `pyrosetta-interface-analyzer` | PyRosetta Interface Analyzer | Compute interface-quality metrics (shape complementarity, H-bonds, ΔG, dSASA, packing, hydrophobicity) for two-chain complexes with optional FastRelax |

## Execution Modes

| Mode | Backend | Device |
|------|---------|--------|
| Standalone env | `ToolInstance("pyrosetta")` running `standalone/inference.py` | CPU only |

All five tools share a single standalone micromamba environment with PyRosetta installed. PyRosetta initialization adds a few seconds of overhead on the first call; subsequent calls within a persistent `ToolInstance` skip this. The interface analyzer additionally depends on `scipy` (already installed in the env).

## How It Works

**SAP (`pyrosetta-sap`)**: Loads the structure into a Rosetta Pose, then computes per-atom SAP scores using Rosetta's `core.pack.guidance_scoreterms.sap` module. The result is a single scalar SAP score per structure.

**SASA (`pyrosetta-sasa`)**: Loads the structure into a Rosetta Pose, then runs `SasaCalc` with a configurable probe radius. Returns both the total SASA and a per-residue breakdown with chain, residue index, residue name, and SASA value.

**Energy (`pyrosetta-energy`)**: Loads the structure and scores it with the specified score function (default `ref2015`). Returns the total energy, a breakdown by score term (fa_atr, fa_rep, fa_sol, etc.), and per-residue total energies. By default no FastRelax is applied — the input structure is scored as-given. To resolve steric clashes and strain in raw predicted structures, set `config.pre_relax_structures=True` (see Relax preprocess below).

**Relax (`pyrosetta-relax`)**: Runs FastRelax and returns the relaxed coordinates as a `Structure` plus the total Rosetta energy. The returned `Structure` is a drop-in replacement for the input (chain labels and format preserved), so it composes cleanly into `pyrosetta-energy`, `pyrosetta-sap`, `pyrosetta-sasa`, or geometric `Structure` methods.

**Interface analyzer (`pyrosetta-interface-analyzer`)**: Loads the structure into a Rosetta Pose, runs `InterfaceAnalyzerMover` to extract shape complementarity, interface H-bond count, binding ΔG, buried SASA, and packing statistic; computes `interface_hydrophobicity` from the composition of interface residues (binder residues with any atom within 4.0 Å of any target atom, scored against the apolar set `"ACFILMPVWY"`); and computes `surface_hydrophobicity` by applying `LayerSelector(pick_surface=True)` to the binder sub-pose and counting apolar + aromatic surface residues. Each input carries its own `target_chain` and `binder_chain` (see Input Parameters); input-native mmCIF chain labels (including multi-character labels) are supported and transparently shortened for PyRosetta's PDB-based internals.

**Relax preprocess (on energy/sap/sasa)**: All three scoring tools and the interface analyzer accept `pre_relax_structures: bool` and `relax_config: PyRosettaRelaxConfig` on their config. When `pre_relax_structures=True`, the framework's `Config.preprocess` hook dispatches `pyrosetta-relax` on every input structure first and substitutes the relaxed structures before scoring. This composes the same `pyrosetta-relax` tool as a preprocess step — there is exactly one FastRelax implementation in the codebase. For batch workloads, run inside a persistent `ToolInstance` so the relax + scoring dispatches share one PyRosetta init.

## Input Parameters

### Scoring tools (`pyrosetta-sap`, `pyrosetta-sasa`, `pyrosetta-energy`, `pyrosetta-relax`)

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `list[ScoringStructureInput]` | Protein structures to score, each with optional `chain_ids` for chain selection. Accepts PDB file paths, PDB content strings, Structure objects, or dicts with `structure` and `chain_ids` keys. A single input is automatically wrapped in a list. |

### `pyrosetta-interface-analyzer`

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `list[InterfaceStructureInput]` | Two-chain complexes to analyze. Each entry carries the `structure` plus the `target_chain` and `binder_chain` labels that define the interface. Chain labels are validated against the structure at construction time — a label not present in the structure raises `ValidationError` immediately. Accepts PDB/CIF file paths, content strings, `Structure` objects, or dicts with `structure` / `target_chain` / `binder_chain` keys. Bare structures default to `target_chain="A"`, `binder_chain="B"`. |

Each `InterfaceStructureInput` has:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `structure` | `Structure` | — | The complex (file path, content string, or `Structure` object). |
| `target_chain` | `str` | `"A"` | Chain label for the target (e.g. receptor) side of the interface, in the structure's native namespace. |
| `binder_chain` | `str` | `"B"` | Chain label for the binder side of the interface, in the structure's native namespace. Must differ from `target_chain`. |

## Configuration

### Shared `pre_relax_structures` preprocess (energy/sap/sasa/interface-analyzer)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pre_relax_structures` | `bool` | `False` | If `True`, run `pyrosetta-relax` on each input structure before scoring. |
| `relax_config` | `PyRosettaRelaxConfig` | `PyRosettaRelaxConfig()` (factory) | Settings used when `pre_relax_structures=True`. UI visibility gated on the toggle via the proto-ui overlay layer. |

### `pyrosetta-sap`

No tool-specific configuration parameters beyond the shared relax preprocess above.

### `pyrosetta-sasa`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `probe_radius` | `float` | `1.4` | > 0.0 | Solvent probe radius in Angstroms. Standard water probe is 1.4 A. |

### `pyrosetta-energy`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scorefxn` | `str` | `"ref2015"` | -- | Rosetta score function name (e.g., `ref2015`, `beta_nov16`, `ref2015_cart`) |

### `pyrosetta-relax`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scorefxn` | `str` | `"ref2015"` | -- | Rosetta score function name |
| `relax_cycles` | `int` | `1` | 1-15 | Number of FastRelax repeats. Default `1` matches Germinal's `-relax:default_repeats 1` |
| `constrain_to_start` | `bool` | `True` | -- | Add a coordinate-constraint term to the relax scorefxn so atoms stay near input positions |

### `pyrosetta-interface-analyzer`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scorefxn` | `str` | `"ref2015"` | -- | Rosetta score function name |

> **Note:** The interface chains (`target_chain` / `binder_chain`) live on each `InterfaceStructureInput` in `inputs`, not on the config — see Input Parameters.

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

Each `ResidueEnergy` contains:

| Field | Type | Description |
|-------|------|-------------|
| `chain_id` | `str` | Chain identifier |
| `residue_index` | `int` | 1-indexed residue position |
| `residue_name` | `str` | Three-letter residue code |
| `total_energy` | `float` | Total residue energy in REU |

### `pyrosetta-relax`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[PyRosettaRelaxMetrics]` | One per input structure |

Each `PyRosettaRelaxMetrics` contains:

| Field | Type | Description |
|-------|------|-------------|
| `total_score` | `float` | Total Rosetta energy of the relaxed pose, in REU |
| `relaxed` | `bool` | Always `True` for outputs of this tool |
| `relax` | `RelaxResult` | Relaxed structure and run metadata |

Each `RelaxResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `relaxed_structure` | `Structure` | Relaxed coordinates. Drop-in replacement for the input: chain labels match and the source format (PDB/CIF) is preserved. |
| `relax_cycles` | `int` | Number of FastRelax repeats applied |

### `pyrosetta-interface-analyzer`

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[PyRosettaInterfaceAnalyzerMetrics]` | One per input structure |

Each `PyRosettaInterfaceAnalyzerMetrics` contains seven metrics:

| Field | Type | Range | Unit | Description |
|-------|------|-------|------|-------------|
| `interface_sc` | `float` | 0-1 | -- | Shape complementarity (1 = perfect fit) |
| `interface_hbonds` | `int` | ≥ 0 | count | Hydrogen bonds across the interface |
| `interface_dG` | `float` | -- | REU | Binding ΔG |
| `interface_dSASA` | `float` | ≥ 0 | Å² | Interface buried SASA |
| `interface_packstat` | `float` | 0-1 | -- | Interface packing statistic |
| `interface_hydrophobicity` | `float` | 0-100 | % | Apolar + aromatic fraction of interface residues × 100 |
| `surface_hydrophobicity` | `float` | 0-1 | -- | Apolar + aromatic fraction of binder surface residues |

Export formats: `csv`, `json` (energy/sap/sasa/interface-analyzer); the relax tool returns a Structure that the caller persists separately.

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
- High `fa_rep` values indicate steric clashes -- run with `pre_relax_structures=True` to resolve these before interpreting other terms.
- Per-residue energies identify problematic positions (e.g., residues with high positive energy are under strain).
- **Chain selection filters display, not computation.** When you set `chain_ids` on an energy input, `total_energy` and `energy_terms` are still computed on the full pose — the selection only filters which residues appear in `per_residue`. Each per-residue energy reflects that residue's contribution *in the context of the full complex* (including pair interactions with the un-selected chains), not the chain's energy in isolation. To score a chain as if it were isolated, extract it into its own Structure first and score it separately.

**Interface-analysis metrics:**
- Most metrics are in well-bounded ranges (`interface_sc`, `interface_packstat`, `surface_hydrophobicity` in [0, 1]; `interface_hydrophobicity` in [0, 100]). `interface_dG` is unbounded; more negative = more favorable binding.
- Typical binder-design filter gates: `interface_sc ≥ 0.6`, `interface_hbonds ≥ 3`, `interface_hydrophobicity ≥ 45`, `surface_hydrophobicity ≤ 0.4`.
- Raw predicted complexes often carry clashes that distort the energy-based metrics (`interface_dG`, `interface_packstat`). Use `pre_relax_structures=True` before reading these values.

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

**Energy scoring (no relax — the default):**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    run_pyrosetta_energy, PyRosettaEnergyInput, PyRosettaEnergyConfig,
)

result = run_pyrosetta_energy(
    PyRosettaEnergyInput(inputs=["/path/to/protein.pdb"]),
    PyRosettaEnergyConfig(scorefxn="ref2015"),
)

for r in result.results:
    print(f"Total energy: {r.total_energy:.1f} REU")
    print(f"  fa_atr: {r.energy_terms.get('fa_atr', 0):.1f}")
    print(f"  fa_rep: {r.energy_terms.get('fa_rep', 0):.1f}")
    print(f"  fa_sol: {r.energy_terms.get('fa_sol', 0):.1f}")
```

**Energy scoring with the `pre_relax_structures` preprocess:**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    run_pyrosetta_energy, PyRosettaEnergyInput, PyRosettaEnergyConfig,
    PyRosettaRelaxConfig,
)

# Set pre_relax_structures=True; the framework calls pyrosetta-relax
# on each input before energy scoring (one PyRosetta init in persistent mode).
result = run_pyrosetta_energy(
    PyRosettaEnergyInput(inputs=["/path/to/raw_prediction.pdb"]),
    PyRosettaEnergyConfig(
        pre_relax_structures=True,
        relax_config=PyRosettaRelaxConfig(relax_cycles=1),
    ),
)
print(f"Energy on relaxed pose: {result.results[0].total_energy:.1f} REU")
```

**Explicit relax → chain into another scorer:**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    run_pyrosetta_relax, PyRosettaRelaxInput, PyRosettaRelaxConfig,
    run_pyrosetta_energy, PyRosettaEnergyInput, PyRosettaEnergyConfig,
)

# 1. Relax the predicted structure once (e.g. cofold from Chai-1 or AF3).
relax_out = run_pyrosetta_relax(
    PyRosettaRelaxInput(inputs=["/path/to/cofold.pdb"]),
    PyRosettaRelaxConfig(relax_cycles=1),    # Germinal default
)
relaxed = relax_out.results[0].relax.relaxed_structure

# 2. Score the relaxed structure (default config = no further relax).
score_out = run_pyrosetta_energy(PyRosettaEnergyInput(inputs=[relaxed]))
print(f"Relaxed total energy: {score_out.results[0].total_energy:.1f} REU")
```

**Interface analysis of a two-chain complex:**
```python
from proto_tools.tools.structure_scoring.pyrosetta import (
    InterfaceStructureInput,
    PyRosettaInterfaceAnalyzerInput,
    run_pyrosetta_interface_analyzer,
)

# target_chain / binder_chain live on the input (defaults "A" / "B").
result = run_pyrosetta_interface_analyzer(
    PyRosettaInterfaceAnalyzerInput(
        inputs=[
            InterfaceStructureInput(
                structure="/path/to/binder_target_complex.pdb",
                target_chain="A",
                binder_chain="B",
            ),
        ],
    ),
)

m = result.results[0]
print(f"interface_sc:               {m.interface_sc:.3f}")
print(f"interface_hbonds:           {m.interface_hbonds}")
print(f"interface_dG:               {m.interface_dG:.1f} REU")
print(f"interface_dSASA:            {m.interface_dSASA:.1f} Å²")
print(f"interface_packstat:         {m.interface_packstat:.3f}")
print(f"interface_hydrophobicity:   {m.interface_hydrophobicity:.1f} %")
print(f"surface_hydrophobicity:     {m.surface_hydrophobicity:.3f}")
```

## Best Practices & Gotchas

- **`linux-aarch64` runs a stale 2023.11 build.** The `conda.rosettacommons.org/linux-aarch64` channel has not been updated since March 2023 — the newest available pyrosetta on aarch64 is `2023.11` (Python 3.10), while `linux-64`, `osx-arm64`, and `osx-x86_64` ship the current `2026.06` builds. Scoring functions, FastRelax, and SAP have evolved over the past ~3 years, so values computed on `linux-aarch64` may differ slightly from values computed on other platforms. The setup script prints a warning banner on aarch64; reproduce final/published numbers on x86_64 or macOS.
- **Relax before energy scoring on raw predictions.** Raw PDB structures from X-ray or structure prediction typically have steric clashes that produce extremely high `fa_rep` values. Set `pre_relax_structures=True` (or call `pyrosetta-relax` first) when scoring such inputs. The default is `pre_relax_structures=False` so that callers passing already-relaxed structures don't double-relax silently.
- **PyRosetta initialization overhead.** The first call in a session takes a few seconds to initialize PyRosetta. Use `ToolInstance.persist()` for batch workloads to amortize this cost — especially important when `pre_relax_structures=True`, since each scored structure triggers two dispatches (relax + score) that both hit the same persistent worker.
- **Seed-sensitive stochastic protocols.** `pyrosetta-relax` and `pyrosetta-interface-analyzer` depend on Rosetta's process-global RNG. When `config.seed is None`, repeated calls skip cache lookup/storage and iterable dedup so the RNG can advance naturally. When `config.seed` is set on a multi-input call, each input shares the seeded RNG state; if you need deterministic per-item seeds (rare — most callers pass distinct inputs, which diverge naturally), seed PyRosetta explicitly inside your own loop and call the tool per item.
- **Chain labels round-trip through the original format.** PDB stores chain IDs as a single character, while mmCIF permits arbitrary-length labels (e.g. `"Heavy"`, `"Light"`, or PDB bundle names like `"AA"`). You can pass either format directly, reference chains by their native labels in `chain_ids`, and read the same labels back in `per_residue[i].chain_id`. The tool internally shortens multi-character chain IDs to fit PDB format when dispatching to PyRosetta and restores the originals in the output, so the conversion is invisible. Structures with more than 62 unique chain labels cannot be represented in PDB and are rejected up front with a clear error.
- **SAP is size-dependent.** Larger proteins naturally have higher SAP scores. Compare SAP across variants of the same protein, not across proteins of different sizes.
- **Energy comparisons require the same score function.** Never compare REU values computed with different `scorefxn` settings (e.g., `ref2015` vs. `beta_nov16`).
- **Relax cycles trade off accuracy vs. speed.** The `pyrosetta-relax` default is 1 cycle (matches Germinal). Increase via `PyRosettaRelaxConfig(relax_cycles=N)` to 5–15 for higher-quality convergence at the cost of runtime.
- **Constrain to start coordinates.** The `PyRosettaRelaxConfig.constrain_to_start=True` default prevents FastRelax from drastically altering the structure. Disable only if you want unconstrained relaxation (e.g., to find the nearest energy minimum).
- **Coordinates are 1-indexed.** Per-residue output uses 1-indexed residue positions consistent with PDB numbering.
- **Interface chains are validated at input construction.** `InterfaceStructureInput` raises `ValidationError` immediately if `target_chain` or `binder_chain` is not present in the structure, or if the two labels are equal — you don't have to wait for a PyRosetta dispatch to discover a typo.

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
