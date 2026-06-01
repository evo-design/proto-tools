<a href="https://bio-pro.mintlify.app/tools/binder-design/bindcraft"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# BindCraft

![BindCraft pipeline](https://raw.githubusercontent.com/martinpacesa/BindCraft/7cd4ace1b7407adf66a50dfefa47de2270f5e4a9/pipeline.png)

> *Image source: [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft)*

> [!NOTE]
> **License:** BindCraft's own code is licensed under MIT, but it runs as a pipeline that depends on bundled components and model weights under separate license terms, including non-commercial or restricted-use terms. The bundled model weights are licensed under CC-BY-4.0. As a whole the pipeline has restrictions around commercial use and may require explicit attribution when utilized.
>
> Bundled dependencies, each under its own license:
>
> - [PyRosetta](https://bio-pro.mintlify.app/tools/structure-scoring/pyrosetta): Custom (PyRosetta Software License)
>
> Review the [code license](https://github.com/martinpacesa/BindCraft/blob/main/LICENSE) and the [model weights license](https://github.com/google-deepmind/alphafold#model-parameters-license) before any commercial use or redistribution.

## Overview

[BindCraft](https://github.com/martinpacesa/BindCraft) is a de novo protein binder design pipeline from the [Correia Lab](https://www.epfl.ch/labs/lpdi/) at EPFL. It hallucinates a binder against a frozen target by back-propagating a structural objective through AlphaFold2, refines the design with ProteinMPNN, re-validates the redesigned complex with AlphaFold2, and filters every candidate against a battery of physics-based interface metrics computed with PyRosetta. This toolkit exposes the full pipeline through a single registered tool that returns the accepted designs together with their relaxed complexes and per-design metrics.

## Background

BindCraft ([Pacesa et al., 2025](https://doi.org/10.1038/s41586-025-09429-6)) addresses the problem of generating protein binders against a target without the need for high-throughput experimental screening or curated structural templates. The published pipeline reports experimental success rates of 10 to 100 percent across diverse and challenging targets including cell-surface receptors, common allergens, de novo designed proteins, and multi-domain nucleases such as CRISPR-Cas9, and produces binders with nanomolar affinity. The authors demonstrate functional and therapeutic applications including reduction of IgE binding to birch allergen in patient-derived samples, modulation of Cas9 gene editing activity, and reduction of cytotoxicity from a foodborne bacterial enterotoxin.

The pipeline chains four stages per design trajectory. First, an AlphaFold2 hallucination step initialises a binder of randomly sampled length adjacent to the frozen target and optimises the binder logits by gradient descent against a weighted sum of structural losses that includes per-residue pLDDT, intra-binder and inter-chain PAE, intra-binder and interface contact counts, interface pTM, a helicity bias, and a radius-of-gyration term. Second, the hallucinated backbone is handed to ProteinMPNN ([Dauparas et al., 2022](https://doi.org/10.1126/science.add2187)) which samples a set of foldable sequences while optionally holding interface residues fixed. Third, each ProteinMPNN-refined complex is re-predicted from scratch with AlphaFold2 multimer ([Jumper et al., 2021](https://doi.org/10.1038/s41586-021-03819-2)) as an independent validation of the design. Fourth, the validated complex is relaxed with PyRosetta and scored against an extensive set of interface metrics including binding-energy difference, shape complementarity, buried surface area, hydrogen bond counts, packing statistic, secondary-structure composition, and hotspot RMSD. A trajectory is accepted only when every metric clears the corresponding upstream filter threshold.

### Learning Resources

- [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft) (Correia Lab, EPFL). Official BindCraft repository, command-line interface, and reference filter configurations.
- [BindCraft tutorial notebook](https://github.com/martinpacesa/BindCraft/blob/main/notebooks/BindCraft.ipynb) (Correia Lab). Walkthrough of the design pipeline with pre-set example targets and parameter explanations.

## Tools

### BindCraft Binder Design (`bindcraft-design`)

Designs one or more de novo protein binders against a user-supplied target. The tool takes a target structure together with the target chain identifiers, an optional hotspot residue list, and a binder length range, and runs the BindCraft pipeline until either the requested number of accepted designs has been produced or the configured trajectory limit has been reached. The output carries each accepted binder as an amino-acid sequence, a relaxed target-binder complex `Structure` with per-residue pLDDT in the B-factor column, and the per-design BindCraft metrics used by the filter check.

#### Applications

This tool is appropriate for de novo binder generation against a structurally characterised target where no curated antibody scaffold or pre-existing binder is available. Representative applications include designing miniprotein binders against cell-surface receptors, generating binders that occlude a specific epitope or active site through hotspot targeting, producing structurally diverse binder candidates for downstream therapeutic engineering, and benchmarking AlphaFold2-hallucination as a binder discovery method against alternative approaches.

#### Usage Tips

- **Provide a hotspot residue list when targeting a defined epitope.** Set `target_hotspot_residues` to a comma-separated list of residue positions on the target structure, with ranges supported (for example `"1-10,56,78"`). Residue numbering is 1-indexed to match standard biological residue numbering conventions. Without hotspots the binder may land anywhere on the target surface. With hotspots, BindCraft biases the hallucination loss to bring the binder into contact with the specified residues. Choose functional residues such as active sites, paratope contacts, or catalytic loops rather than arbitrary surface positions.
- **`binder_lengths` defaults to `(65, 150)` residues, matching the upstream default.** Binders below approximately 50 residues are effectively peptides and the AlphaFold2 multimer signal weakens. Binders above approximately 200 residues introduce significant GPU memory and per-trajectory runtime costs. Choose a tighter range to focus a campaign on a specific binder size class.
- **`weights_helicity` controls the helix bias during hallucination.** The default of `-0.3` is a mild anti-helix bias chosen by the upstream authors because AlphaFold2 tends to over-produce alpha-helical bundles. Set a positive value to encourage helices for helix-friendly targets, or set `random_helicity=True` to randomise the sign per trajectory and increase secondary-structure diversity across the campaign.
- **`optimise_beta=True` (the default) adds extra hallucination iterations and AlphaFold2 recycles when a trajectory looks beta-heavy.** Keep this enabled for any target that may favour beta-strand interfaces, such as immunoglobulin folds. The behaviour is gated on detected sheet content during the trajectory.
- **`filter_overrides` lets you relax or tighten individual filter thresholds.** Pass a dict keyed by upstream metric name (such as `"Average_i_pTM"`) and valued as a filter dict (`{"threshold": 0.45, "higher": True}`). Only the listed metrics are overridden; every other filter keeps its upstream default. Lower the interface pTM or shape complementarity threshold first if zero designs are accepted on a hard target.
- **Production runs use `number_of_final_designs=100` and `max_trajectories=False`.** This is the upstream default and produces enough accepted designs for downstream triage and experimental ordering. For a smoke test, set both to `1` together with reduced iteration counts (for example `soft_iterations=10`, `temporary_iterations=5`, `hard_iterations=2`, `greedy_iterations=2`) to verify the install and produce a single sample.
- **`enable_rejection_check=True` (the default) aborts a run early if the rolling acceptance rate falls below `acceptance_rate=0.01` after `start_monitoring=600` trajectories.** Disable this gate when working on stubborn targets where you are willing to grind through many failed trajectories before the first acceptance.
- **The output is iterable.** Iterating directly over the returned `BindCraftOutput` yields each accepted `BindCraftDesign` in turn, and `len(result)` returns the number of accepted designs.
- **Complementary tools cover adjacent design tasks.** Reach for `proteinmpnn-sample` when an existing target-bound binder backbone only needs sequence redesign, `alphafold2-gradient` (with the Germinal backend) or a dedicated antibody-design pipeline for CDR-only redesign on a fixed antibody framework, `rfdiffusion3-design` when only a backbone is required without an accompanying sequence, and chemistry-aware ligand generation, docking, and scoring tools when the target is a small-molecule ligand rather than a protein binder.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every BindCraft tool in this toolkit (`bindcraft-design`).

- **The pipeline runs on a single GPU per trajectory and benefits from 32 to 80 GB of GPU memory.** AlphaFold2 multimer dominates the memory footprint and scales with the combined target plus binder length. For targets larger than approximately 2000 residues, trim the target to its binder-accessible domain before running. To parallelise across multiple GPUs, run multiple instances of `bindcraft-design` concurrently through a `ToolPool`.
- **The first run downloads approximately 5.5 GB of AlphaFold2 weights together with the ColabDesign, ProteinMPNN, and BindCraft repositories.** Subsequent runs reuse the cached weights, which are shared with the proto-tools `alphafold2` toolkit.
