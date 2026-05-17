<a href="https://bio-pro.mintlify.app/tools/structure-prediction/boltz2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Boltz-2

![Boltz-2](https://github.com/jwohlwend/boltz/raw/main/docs/boltz1_pred_figure.png)

> *Image source: [jwohlwend/boltz](https://github.com/jwohlwend/boltz)*

> [!NOTE]
> **License:** Boltz-2 is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/jwohlwend/boltz/blob/main/LICENSE) for full terms.

## Overview

Boltz-2 is an openly licensed biomolecular structure prediction model from the [MIT Jameel Clinic](https://jclinic.mit.edu/) and [Recursion](https://www.recursion.com/), built in the AlphaFold3 family: a diffusion model that predicts the joint 3D structure of complexes mixing proteins, DNA, RNA, and small-molecule ligands. This toolkit runs Boltz-2 structure prediction on a local GPU, with optional ColabFold multiple-sequence alignments. (Boltz-2 the model also predicts binding affinity; this toolkit currently exposes its structure prediction.)

## Background

Boltz-2 ([Passaro et al., 2025](https://doi.org/10.1101/2025.06.14.659707)) predicts the joint 3D structure of a biomolecular assembly from the sequences and chemical components it contains. It builds on Boltz-1, one of the most widely used open-source alternatives to AlphaFold3, extending that co-folding model with a binding-affinity module, improved controllability, and additional training data. Like AlphaFold3, a single model folds complexes that mix proteins, DNA, RNA, and small-molecule ligands and predicts how those components are arranged relative to one another. Each protein chain can be paired with a multiple-sequence alignment (MSA) of evolutionarily related sequences, whose covariation patterns supply the evolutionary signal the model uses to place residues.

Architecturally, Boltz-2 reproduces AlphaFold3: it carries a single representation of the input tokens and a pairwise representation over token pairs, refines them through an AlphaFold3-style trunk, and generates all-atom coordinates with a diffusion module that starts from noise and iteratively denoises into a structure. Several structures can be sampled per complex and ranked by a confidence score, reported as a complex predicted local distance difference test (pLDDT) for local reliability, a predicted aligned error (PAE) for the relative placement of any two tokens, and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores that summarize overall and interface accuracy. Beyond structure, Boltz-2 adds a binding-affinity module that approaches the accuracy of physics-based free-energy perturbation while running more than 1000 times faster.

The reference implementation is open-sourced at [jwohlwend/boltz](https://github.com/jwohlwend/boltz) under the MIT license, covering the code, weights, and training pipeline for both academic and commercial use, with the released weights distributed as `boltz-community/boltz-2`. It was developed by the Boltz team at the [MIT Jameel Clinic](https://jclinic.mit.edu/) together with [Recursion](https://www.recursion.com/).

### Learning Resources

- [Boltz-2: democratizing biomolecular interaction modeling](https://boltz.bio/boltz2) (MIT Jameel Clinic and Recursion) - an accessible overview of Boltz-2, including how it extends on the work of Boltz-1 and its binding-affinity capability.

## Tools

### Boltz-2 Structure Prediction (`boltz2-prediction`)

Predicts the 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains; the assembly is folded by Boltz-2 and returned as a predicted `Structure` per complex with confidence metrics: a complex pLDDT, pTM, interface pTM, per-chain and pairwise-chain pTM/ipTM, and predicted aligned error.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-DNA and protein-RNA complexes or protein-ligand binding poses. Running it on a multi-chain complex also estimates how confidently the components are placed relative to each other through interface pTM and PAE, which is informative for assessing predicted interfaces.

#### Usage Tips

- **`use_msa` defaults to `True`.** A ColabFold search generates an MSA for each protein chain; set it `False` for single-sequence prediction, or attach precomputed MSAs to the input. Protein chains with no detectable homologs fall back to an empty MSA.
- **Structures come from a diffusion process.** `diffusion_samples` (default `1`) independent samples are drawn per complex and the best is kept by `confidence_score`; `sampling_steps` (default `200`) sets the number of denoising steps and `step_scale` (default `1.5`) trades accuracy for sample diversity, where lower values are more diverse.
- **`recycling_steps` (default `3`) trades accuracy for time.** More recycling iterations refine the prediction but increase runtime.
- **Confidence is reported as a complex pLDDT, pTM, ipTM, and PAE.** `confidence_score`, the primary metric, is `iptm` for multi-chain complexes and `ptm` for a single chain; `complex_plddt` is on a 0 to 1 scale and PAE is in angstroms (0 to about 32). Set `include_pae_matrix` to attach the full per-token PAE matrix.
- **Multi-modal inputs.** Protein, DNA, RNA, and ligand entities are supported; chain modifications are not.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Boltz-2 tool in this toolkit (`boltz2-prediction`).

- **Requires a GPU.** Boltz-2 runs through a PyTorch backend and needs an NVIDIA GPU; CPU execution is not practical.
- **MSA-based and AlphaFold3-style.** Unlike ESMFold's single-sequence folding, Boltz-2 uses optional MSAs and a diffusion process. Predictions are stochastic, so set `seed` for reproducibility; `subsample_msa` and unseeded runs are intentionally non-deterministic.
- **Structure prediction only.** This toolkit wraps Boltz-2's structure prediction; the model's separate binding-affinity prediction is not exposed here.
