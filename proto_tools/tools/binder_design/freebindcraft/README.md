<img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)"><a href="https://bio-pro.mintlify.app/tools/binder-design/freebindcraft"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# FreeBindCraft

> [!NOTE]
> **License:** FreeBindCraft uses MIT for code and CC-BY-4.0 for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/cytokineking/FreeBindCraft/blob/master/LICENSE) and [model weights license](https://github.com/google-deepmind/alphafold#model-parameters-license) for full terms.

## Overview

[FreeBindCraft](https://github.com/cytokineking/FreeBindCraft) is a PyRosetta-free fork of [BindCraft](https://github.com/martinpacesa/BindCraft), the de novo protein binder design pipeline from the Correia Lab at EPFL. It hallucinates a binder against a frozen target with AlphaFold2, refines it with ProteinMPNN, re-validates the complex, and scores the interface — replacing every PyRosetta step with an open-source equivalent (OpenMM, FreeSASA, sc-rs). The result is a single registered tool that returns accepted binders with their relaxed complexes and per-design metrics, with no PyRosetta license restriction so it can be hosted commercially.

## Background

BindCraft ([Pacesa et al., 2025](https://doi.org/10.1038/s41586-025-09429-6)) addresses the problem of generating protein binders against a target without high-throughput experimental screening or curated structural templates, reporting experimental success rates of 10 to 100 percent across diverse targets. FreeBindCraft preserves the identical AlphaFold2-hallucination and ProteinMPNN design logic — the hallucination losses, MPNN sampling, and AF2 validation are unchanged — while swapping out the proprietary scoring backend.

The pipeline chains four stages per design trajectory. First, an AlphaFold2 hallucination step initialises a binder of randomly sampled length adjacent to the frozen target and optimises the binder logits by gradient descent against a weighted sum of structural losses (per-residue pLDDT, intra-binder and inter-chain PAE, intra-binder and interface contact counts, interface pTM, a helicity bias, and a radius-of-gyration term). Second, the hallucinated backbone is handed to ProteinMPNN ([Dauparas et al., 2022](https://doi.org/10.1126/science.add2187)), which samples foldable sequences while optionally holding interface residues fixed. Third, each refined complex is re-predicted from scratch with AlphaFold2 multimer ([Jumper et al., 2021](https://doi.org/10.1038/s41586-021-03819-2)) as an independent validation. Fourth, the validated complex is relaxed and scored — where BindCraft uses PyRosetta, FreeBindCraft uses an OpenMM-based relaxation protocol (PDBFixer cleanup, FASPR side-chain packing, implicit solvation) and computes interface metrics with FreeSASA, Biopython, and the [sc-rs](https://github.com/cytokineking/sc-rs) shape-complementarity binary.

### PyRosetta-free metrics

FreeBindCraft computes the AlphaFold2 confidence metrics (pLDDT, pTM, interface pTM, ipSAE, PAE) and the geometry-based interface metrics — shape complementarity (sc-rs), buried surface area and SASA fractions (FreeSASA), interface-residue counts, secondary-structure composition, hotspot/target/binder RMSDs, and pre- and post-relaxation clash counts (geometric, Biopython) — for real. Metrics that depend on Rosetta's energy function and lack an open-source equivalent (interface binding energy `dG`, `dG/dSASA`, `Binder_Energy_Score`, `PackStat`, and hydrogen-bond counts) are emitted upstream as placeholders only to satisfy default filters; this toolkit does **not** surface them, so every metric returned by `freebindcraft-design` is a real measurement.

### Learning Resources

- [cytokineking/FreeBindCraft](https://github.com/cytokineking/FreeBindCraft). The PyRosetta-free fork, its `--no-pyrosetta` install/runtime flag, and the technical overview of the open-source scoring replacements.
- [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft) (Correia Lab, EPFL). The upstream BindCraft repository, command-line interface, and reference filter configurations.

## Tools

### FreeBindCraft Binder Design (`freebindcraft-design`)

Designs one or more de novo protein binders against a user-supplied target. The tool takes a target structure together with the target chain identifiers, an optional hotspot residue list, and a binder length range, and runs the FreeBindCraft pipeline until either the requested number of accepted designs has been produced or the configured trajectory limit has been reached. The output carries each accepted binder as an amino-acid sequence, an OpenMM-relaxed target-binder complex `Structure` with per-residue pLDDT in the B-factor column, and the per-design PyRosetta-free metrics used by the filter check.

#### Applications

This tool is appropriate for de novo binder generation against a structurally characterised target where no curated antibody scaffold or pre-existing binder is available, and where a permissive license is required for hosting or commercial use. Representative applications include designing miniprotein binders against cell-surface receptors, generating binders that occlude a specific epitope or active site through hotspot targeting, and producing structurally diverse binder candidates for downstream therapeutic engineering.

#### Usage Tips

- **Provide a hotspot residue list when targeting a defined epitope.** Set `target_hotspot_residues` to a comma-separated list of 1-indexed residue positions on the target structure, with ranges supported (for example `"1-10,56,78"`). Without hotspots the binder may land anywhere on the target surface.
- **`binder_lengths` defaults to `(65, 150)` residues, matching the upstream default.** Binders below approximately 50 residues are effectively peptides and the AlphaFold2 multimer signal weakens; binders above approximately 200 residues introduce significant GPU memory and runtime costs.
- **`weights_helicity` controls the helix bias during hallucination.** The default of `-0.3` is a mild anti-helix bias because AlphaFold2 tends to over-produce alpha-helical bundles. Set a positive value to encourage helices, or set `random_helicity=True` to randomise the sign per trajectory.
- **`optimise_beta=True` (the default) adds extra hallucination iterations and AlphaFold2 recycles when a trajectory looks beta-heavy.** Keep this enabled for any target that may favour beta-strand interfaces, such as immunoglobulin folds.
- **`filter_overrides` lets you relax or tighten individual filter thresholds.** Pass a dict keyed by upstream metric name (such as `"Average_i_pTM"`) and valued as a filter dict (`{"threshold": 0.45, "higher": True}`). Lower the interface pTM or shape complementarity threshold first if zero designs are accepted on a hard target. Note that filters on PyRosetta-only metrics (`Average_dG`, H-bond counts, `Average_PackStat`) pass trivially against placeholder values, so design selection is driven by the AlphaFold2 confidence, shape-complementarity, SASA, and clash metrics.
- **Production runs use `number_of_final_designs=100` and `max_trajectories=False`.** For a smoke test, set both to `1` together with reduced iteration counts (for example `soft_iterations=10`, `temporary_iterations=5`, `hard_iterations=2`, `greedy_iterations=2`).
- **The output is iterable.** Iterating directly over the returned `FreeBindCraftOutput` yields each accepted `FreeBindCraftDesign` in turn, and `len(result)` returns the number of accepted designs.
- **Complementary tools cover adjacent design tasks.** Reach for `proteinmpnn-sample` when an existing target-bound binder backbone only needs sequence redesign, `rfdiffusion3-design` when only a backbone is required without an accompanying sequence, and `bindcraft-design` (the PyRosetta variant) when Rosetta interface energetics are required for academic, non-commercial use.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to the FreeBindCraft tool in this toolkit (`freebindcraft-design`).

- **The pipeline runs on a single GPU per trajectory and benefits from 32 to 80 GB of GPU memory.** AlphaFold2 multimer dominates the memory footprint and scales with the combined target plus binder length. To parallelise across multiple GPUs, run multiple instances of `freebindcraft-design` concurrently through a `ToolPool`.
- **OpenMM relaxation is GPU-accelerated and runs noticeably faster than PyRosetta's CPU-bound FastRelax**, so a FreeBindCraft trajectory typically completes faster than the equivalent BindCraft trajectory.
- **The first run downloads approximately 5.5 GB of AlphaFold2 weights together with the ColabDesign and FreeBindCraft repositories.** Subsequent runs reuse the cached weights, which are shared with the proto-tools `alphafold2` toolkit.
