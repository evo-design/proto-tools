<a href="https://bio-pro.mintlify.app/tools/structure-scoring/pyrosetta"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# PyRosetta

> [!NOTE]
> **License:** PyRosetta is licensed under Custom (PyRosetta Software License) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://www.pyrosetta.org/home/licensing-pyrosetta) for full terms.

## Overview

[PyRosetta](https://www.pyrosetta.org/) is the Python interface to the [Rosetta](https://github.com/RosettaCommons/rosetta) molecular modelling suite from the [RosettaCommons](https://rosettacommons.org/) consortium. This toolkit exposes five physics-based operations: Spatial Aggregation Propensity (SAP) scoring, Solvent Accessible Surface Area (SASA) computation, full Rosetta energy scoring, FastRelax structural minimisation, and interface analysis of two-chain complexes through `InterfaceAnalyzerMover`. The scoring tools and the interface analyzer can run FastRelax first via an opt-in `pre_relax_structures` preprocess.

## Background

The Rosetta molecular modelling suite ([Alford et al., 2017](https://doi.org/10.1021/acs.jctc.7b00125)) provides an all-atom energy function that combines van der Waals interactions, hydrogen bonding, electrostatics, and an implicit solvation model into a single score reported in Rosetta Energy Units (REU). The current community-standard energy function, REF2015, is parametrised against small-molecule and X-ray crystal structure data and is the default score function used by every tool in this toolkit. PyRosetta ([Chaudhury, Lyskov, and Gray, 2010](https://doi.org/10.1093/bioinformatics/btq007)) exposes the Rosetta sampling and scoring functions through a Python interface, which this toolkit invokes to compute per-residue and overall energies together with a breakdown by score term.

Spatial Aggregation Propensity (SAP) ([Chennamsetty, Voynov, Kayser, Helk, and Trout, 2009](https://doi.org/10.1073/pnas.0904191106)) quantifies how much hydrophobic surface area is exposed on a protein. SAP combines per-residue hydrophobicity with local solvent exposure within a sphere around each surface atom and aggregates the contributions across the protein, with higher values corresponding to greater aggregation risk. The published method was originally developed for therapeutic antibody engineering and has become a standard developability filter in protein design.

Solvent Accessible Surface Area (SASA) measures the surface area of a protein that is accessible to a spherical solvent probe (1.4 Å for water by default). Per-residue SASA values distinguish buried residues that contribute to the hydrophobic core from solvent-exposed residues that interact with the surroundings. Rosetta's FastRelax protocol performs many rounds of side-chain repacking and energy minimisation while gradually ramping the repulsive weight in the score function, which finds a low-energy conformation near the input structure and resolves the steric clashes that would otherwise dominate the energy. The `InterfaceAnalyzerMover` extracts a set of structural descriptors that characterise the binding interface of a two-chain complex, including binding-energy difference (`dG_separated`), interface buried SASA (`dSASA_int`), hydrogen bond count (`hbonds_int`), packing statistic (`packstat`), and shape complementarity (`sc_value`), and is widely used as the basis for filter cascades in binder-design pipelines.

### Learning Resources

- [PyRosetta documentation](https://www.pyrosetta.org/) (Gray Lab, Johns Hopkins University). Tutorials, API reference, and installation guidance for the underlying Python interface.
- [RosettaCommons documentation](https://www.rosettacommons.org/docs/latest/Home) (RosettaCommons). Reference manual for the Rosetta scoring functions, movers, and protocols invoked by this toolkit.
- [FastRelax mover reference](https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover) (RosettaCommons). Documentation of the FastRelax protocol exposed as `pyrosetta-relax`.

## Tools

### PyRosetta Energy Score (`pyrosetta-energy`)

Scores one or more protein structures with a Rosetta score function and returns the total energy, a per-term breakdown (fa_atr, fa_rep, fa_sol, hbond_*, etc.), and a per-residue energy contribution. The full pose is always scored regardless of any chain selection. By default the input structure is scored as given. Set `pre_relax_structures=True` to run FastRelax first.

#### Applications

This tool is appropriate for relative energy comparison across variants of the same protein. Representative applications include ranking sequence designs by predicted stability after a relax pass, identifying problematic residues through their per-residue energy contributions, and quantifying the energy cost of mutations or conformational changes.

#### Usage Tips

- **Compare REU values only across variants of the same protein with the same score function.** Rosetta energies are not absolute thermodynamic quantities and do not transfer across proteins of different sizes or across different score function settings. Switching `scorefxn` from `ref2015` to `beta_nov16` produces a different scale and the values are not comparable.
- **Run with `pre_relax_structures=True` when scoring raw predicted complexes.** Predicted structures from AlphaFold, Chai, Boltz, and similar tools commonly carry steric clashes that produce extremely high `fa_rep` values and dominate the total energy. Relaxing first resolves these clashes so the other energy terms become interpretable.
- **Chain selection filters the per-residue breakdown only.** When `chains_to_score` is set on a `ScoringStructureInput`, `total_energy` and `energy_terms` are still computed on the full pose. Each per-residue energy reflects that residue's contribution within the full complex, including pair interactions with the unselected chains. Score a chain in isolation by extracting it into its own `Structure` first.

### PyRosetta Interface Analyzer (`pyrosetta-interface-analyzer`)

Runs Rosetta's `InterfaceAnalyzerMover` on a two-chain complex and returns seven always-on interface descriptors together with an optional eighth (`delta_unsat_hbonds`, available when DAlphaBall is installed). The interface is defined by the `target_chain` and `binder_chain` fields on each `InterfaceStructureInput` and is validated at input construction.

#### Applications

This tool is appropriate for filtering and ranking designed protein binders against a target. Representative applications include gating candidate binders on shape complementarity and hydrogen bond count, ranking by predicted binding-energy difference, and identifying poses with excessive interface-buried hydrophobic surface area.

#### Usage Tips

- **The seven always-on metrics span well-defined ranges.** `interface_sc` is in 0 to 1 (higher is better fit), `interface_packstat` is in 0 to 1 (higher is better packing), `interface_hydrophobicity` is in 0 to 100 (percent apolar plus aromatic interface residues), `surface_hydrophobicity` is in 0 to 1 (apolar plus aromatic fraction of the binder surface), `interface_hbonds` is an integer count, `interface_dSASA` is in Å², and `interface_dG` is in REU (more negative indicates more favourable binding).
- **`delta_unsat_hbonds` requires DAlphaBall and is reported as `None` when the SASA dependency is unavailable.** The Rosetta `BuriedUnsatHbonds` filter uses DAlphaBall for accurate buried-surface SASA. The standalone environment installs DAlphaBall when possible. When the metric is `None`, the rest of the seven always-on metrics are still produced normally.
- **Relax raw predicted complexes before reading the energy-derived metrics.** `interface_dG` and `interface_packstat` are sensitive to steric clashes in unrelaxed structures. Set `pre_relax_structures=True` on the configuration to run FastRelax first, or call `pyrosetta-relax` explicitly and pass the relaxed structure back in.
- **Chain labels follow the input format.** PDB stores chain IDs as a single character, while mmCIF accepts multi-character labels. The tool transparently shortens multi-character labels to single characters when dispatching to PyRosetta and restores the originals in the output.

### PyRosetta FastRelax (`pyrosetta-relax`)

Runs Rosetta's FastRelax protocol on one or more input structures and returns the relaxed coordinates as a `Structure` together with the total Rosetta energy. The returned structure preserves the original chain labels and source format so that it composes directly into any of the other tools in this toolkit or into geometric `Structure` methods.

#### Applications

This tool is appropriate as a preprocessing step before downstream energy scoring, interface analysis, or geometric filtering of raw predicted structures. Representative applications include resolving steric clashes in cofolded complexes before binder-design filter cascades, generating a relaxed reference pose before screening sequence variants, and producing a stable starting point for further structural analyses.

#### Usage Tips

- **`relax_cycles` defaults to `1` and accepts integer values from 1 to 15.** A single FastRelax cycle matches the default used by the Germinal binder-design pipeline and is appropriate for most filter-cascade applications. Increase to 5 to 15 for higher-quality convergence at proportional runtime cost.
- **`constrain_to_start=True` (the default) prevents FastRelax from drastically altering the structure.** This adds a coordinate-constraint term to the relax score function so atoms stay near their input positions. Set to `False` for unconstrained minimisation when the goal is to find the nearest energy minimum.
- **Additional FastRelax controls are available on `PyRosettaRelaxConfig`.** Pass `disable_jumps=True` to lock the inter-chain rigid-body degrees of freedom during relaxation, `align_to_start=True` to superpose the relaxed pose back onto the starting pose after relaxation, or `copy_b_factors_from_start=True` to copy the per-residue B-factor values from the input.

### PyRosetta SAP Score (`pyrosetta-sap`)

Scores one or more protein structures with the Spatial Aggregation Propensity protocol from Rosetta's `core.pack.guidance_scoreterms.sap` module and returns the overall SAP score together with a per-residue SAP contribution breakdown. Higher values indicate greater predicted aggregation risk.

#### Applications

This tool is appropriate for developability assessment during therapeutic protein and antibody engineering, where surface aggregation propensity is a critical liability. Representative applications include ranking antibody variants by predicted aggregation risk, identifying surface mutations that reduce SAP without affecting binding, and screening computationally designed proteins for developability before experimental characterisation.

#### Usage Tips

- **SAP is size-dependent and only meaningfully compared across variants of the same protein.** Larger proteins naturally have higher absolute SAP values because more total surface area contributes. Comparisons across different proteins or different chain compositions are not informative.
- **`chains_to_score` controls which residues contribute to the score.** Setting `chains_to_score=["A"]` on a `ScoringStructureInput` restricts the SAP sum to residues of chain A. The full structure is still loaded so the surrounding context informs the burial calculation, but only the selected chain's atoms contribute to the score.

### PyRosetta SASA (`pyrosetta-sasa`)

Computes total and per-residue Solvent Accessible Surface Area using Rosetta's `SasaCalc` module with a configurable probe radius. Returns the total SASA in Å² together with a per-residue breakdown of chain, 1-indexed residue index, three-letter residue name, and SASA value.

#### Applications

This tool is appropriate for identifying buried and exposed residues, characterising hydrophobic surface patches, and computing a surface-area baseline for downstream developability or interaction analyses. Representative applications include flagging exposed hydrophobic residues as redesign candidates, summarising the surface-residue composition of a designed protein, and computing the buried surface area difference between bound and unbound states.

#### Usage Tips

- **`probe_radius` defaults to 1.4 Å.** This is the conventional water probe radius. Larger probe values are sometimes used to approximate the accessibility seen by larger solvent molecules or interaction partners.
- **Per-residue SASA values near 0 indicate fully buried residues.** Values above approximately 100 Å² indicate significant solvent exposure for a typical residue. Combine with residue identity to identify exposed hydrophobic residues as aggregation hotspots.
- **Total SASA scales with protein size.** Normalise by residue count or surface area when comparing across proteins of different sizes.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PyRosetta tool in this toolkit (`pyrosetta-energy`, `pyrosetta-interface-analyzer`, `pyrosetta-relax`, `pyrosetta-sap`, `pyrosetta-sasa`).

- **Every tool accepts a list of inputs in a single call.** The scoring, relaxation, and SASA tools take a list of `ScoringStructureInput` entries, and the interface analyzer takes a list of `InterfaceStructureInput` entries. Each entry independently accepts a `Structure` object, a file path, a PDB or mmCIF content string, or a dict shorthand. A single bare input is automatically wrapped in a list. Results are returned in the same order as the inputs.
- **The four scoring and interface-analyzer tools share an opt-in `pre_relax_structures` preprocess that runs `pyrosetta-relax` first.** Set `pre_relax_structures=True` and optionally pass a `PyRosettaRelaxConfig` to relax every input structure before scoring. The framework's preprocess hook dispatches `pyrosetta-relax` and substitutes the relaxed structures, so there is exactly one FastRelax implementation in the codebase.
- **Per-residue output uses 1-indexed positions consistent with PDB numbering.** Residue indices in the per-residue energy and per-residue SASA breakdowns correspond directly to the residue numbers in the input structure.
