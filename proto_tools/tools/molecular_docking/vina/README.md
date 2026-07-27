<a href="https://bio-pro.mintlify.app/tools/molecular-docking/vina"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# AutoDock Vina

> [!NOTE]
> **License:** AutoDock Vina's own code is licensed under Apache-2.0, and it federates over bundled data sources and components, each under its own license terms.
>
> Bundled dependencies, each under its own license:
>
> - [Meeko](https://github.com/forlilab/Meeko/blob/develop/LICENSE): Custom (LGPL-2.1-or-later)
> - [RCSB Protein Data Bank fixture data](https://www.rcsb.org/pages/policies): CC0-1.0
>
> Review each source's terms before commercial use or redistribution.

## Overview

[AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) predicts how a small molecule can bind within a specified region of a rigid receptor. This toolkit prepares receptor and ligand PDBQT models with Meeko, runs Vina or Vinardo scoring on CPU, and returns ranked poses with affinity scores, RMSD bounds, bond-correct SDF and PDBQT exports, and preparation provenance including residue-template assignments and ligand-minimization convergence.

## Background

AutoDock Vina ([Trott and Olson, 2010](https://doi.org/10.1002/jcc.21334)) combines a knowledge-inspired scoring function with iterated local search to explore ligand translations, orientations, and rotatable bonds inside a user-defined three-dimensional grid. Vina 1.2 ([Eberhardt et al., 2021](https://doi.org/10.1021/acs.jcim.1c00203)) added Python bindings, expanded atom typing, and support for the Vinardo scoring function ([Quiroga and Villarreal, 2016](https://doi.org/10.1371/journal.pone.0155183)).

Docking requires two distinct decisions: the chemical state of the molecules and the region to search. The receptor remains rigid, while Vina samples the ligand's movable torsions. Meeko ([Santos-Martins et al., 2025](https://doi.org/10.1021/acs.jcim.5c02271)) assigns AutoDock atom types and partial charges, writes PDBQT inputs, and reconstructs each predicted pose as an SDF record with the ligand's original bond orders.

The reported affinity is a docking score in kcal/mol. More negative scores rank more favorably within the same receptor, ligand, box, and scoring setup, but they are not calibrated experimental binding free energies. The RMSD lower and upper bounds measure each returned mode's distance from the best predicted mode; they are not RMSDs against a crystallographic reference pose. Docking is most useful for generating plausible binding hypotheses, enriching a virtual-screening shortlist, or comparing poses before higher-cost simulation or experimental validation.

The bundled receptor and reference-ligand fixtures are derived from chain A of RCSB PDB entry [1IEP](https://www.rcsb.org/structure/1IEP), the c-Abl/imatinib complex reported by [Nagar et al. (2002)](https://pubmed.ncbi.nlm.nih.gov/12154025/). RCSB PDB archive data is distributed under CC0-1.0.

### Learning Resources

- [AutoDock Vina documentation](https://autodock-vina.readthedocs.io/en/latest/) - official installation, docking, and scoring documentation.
- [AutoDock Vina repository](https://github.com/ccsb-scripps/AutoDock-Vina) - source code, releases, and issue tracker.
- [Meeko repository](https://github.com/forlilab/Meeko) - receptor and ligand preparation implementation used by this toolkit.

## Tools

### AutoDock Vina Docking (`vina-docking`)

Prepares one rigid receptor and one small-molecule ligand, searches either an explicit box or a box derived from a coordinate-bearing reference ligand, and returns the ranked poses retained by Vina. A bare SMILES string is accepted as ligand input; the tool generates a seeded three-dimensional conformer before docking.

#### Applications

Use this tool for redocking a known ligand into an experimental receptor, proposing binding modes for analogs, screening a focused set of compounds against a defined pocket, or producing initial protein-ligand poses for molecular dynamics and free-energy workflows. It is also useful for checking whether a designed pocket can accommodate a candidate ligand without severe steric conflicts.

#### Usage Tips

- **Define the search box from pocket evidence.** Use `VinaSearchBox` when a known pocket center is available. Use `VinaReferenceLigandBox` when a co-crystallized ligand is aligned to the receptor; `padding` is added on both sides of each ligand axis.
- **Keep the box focused but large enough for the ligand.** An oversized box makes the search less efficient, while a box that clips the ligand or pocket can exclude valid poses. A 20 to 25 angstrom box is a common starting point for drug-like ligands.
- **Stay within the grid-allocation limits.** Each box axis is limited to 100 angstroms, `grid_spacing` must be at least 0.1 angstrom, and the resolved map may contain at most 2,000,000 grid points. Increase spacing or reduce the box when validation reports a larger allocation.
- **Prepare the receptor's chemical state deliberately.** Resolve missing atoms, alternate locations, protonation states, cofactors, metals, and waters before docking. Meeko adds missing hydrogens from its residue templates but does not perform environment-aware pKa prediction; a hydrogen-free ambiguous histidine can therefore receive the default `HIE` template. Inspect `receptor_template_assignments` in output metadata. Unsupported residues fail by default; `allow_bad_residues=True` deletes residues Meeko cannot parameterize, reports their identifiers in `ignored_receptor_residues`, and emits a warning.
- **Encode ligand protonation and stereochemistry in the SMILES.** The tool preserves the input graph, generates a new seeded conformer, and minimizes it with MMFF94 or UFF when parameters are available. It does not enumerate tautomers, protonation states, or undefined stereocenters; an undefined center may produce one seed-dependent geometry. Specify stereochemistry explicitly and evaluate each intended chemical state as a separate input.
- **Increase `exhaustiveness` for production searches.** The default of 8 is suitable for an initial run. Larger or more flexible ligands often need repeated seeds and higher exhaustiveness to establish that the top-ranked pose is stable.
- **Use `seed` for exact reruns.** When omitted, the framework generates a positive signed 32-bit seed and returns it in the output. Reuse the returned seed with the same environment and configuration to reproduce the search.
- **Treat affinity as a ranking signal, not an absolute binding measurement.** Compare scores only across chemically and procedurally consistent runs, and inspect interactions and pose plausibility before drawing conclusions.
- **Use SDF for downstream chemistry workflows.** Each pose and the combined result are returned as SDF with reconstructed bond orders. PDBQT is also retained for AutoDock interoperability and auditability.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every AutoDock Vina tool in this toolkit (`vina-docking`).

- **The first local call creates an isolated environment.** It installs Vina 1.2.7 from conda-forge and pinned Meeko 0.7.1, after which the environment is reused. Conda-forge supplies Vina builds for supported Linux and macOS architectures. Persistent execution avoids repeated worker startup and chemistry-library imports across a docking batch.
- **Docking is CPU-only.** `cpu=0` lets Vina use all visible CPUs; set a positive value to bound each run's thread consumption. GPU device settings do not accelerate this toolkit.
- **The receptor is rigid.** Side-chain or backbone flexibility, covalent docking, explicit-solvent sampling, and induced-fit refinement are outside this tool's scope.
- **Outputs include complete provenance needed to repeat a run.** The concrete seed, resolved box, effective search controls, Vina, Meeko, and RDKit versions, requested and returned pose counts, receptor omissions and template assignments, and ligand-minimization convergence are returned alongside per-pose scores and SDF/PDBQT payloads.
