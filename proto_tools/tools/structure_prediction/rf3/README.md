<a href="https://bio-pro.mintlify.app/tools/structure-prediction/rf3"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# RoseTTAFold3 (RF3)

![RoseTTAFold3 (RF3)](https://proto-bio.github.io/proto-assets/images/tool/rf3/hero.png)

> [!NOTE]
> **License:** RoseTTAFold3 (RF3) is open source and free for academic and commercial use under a BSD-3-Clause license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/RosettaCommons/foundry/blob/production/LICENSE.md) for full terms.

## Overview

RoseTTAFold3 (RF3) is an all-atom biomolecular structure prediction model from the [Institute for Protein Design](https://www.ipd.uw.edu/) at the University of Washington, distributed as part of the [RosettaCommons Foundry](https://github.com/RosettaCommons/foundry) framework. It predicts the joint 3D structure of complexes that combine proteins, DNA, RNA, and small-molecule ligands, with first-class support for chirality. This toolkit runs RF3 structure prediction from sequences, SMILES, and CCD codes, with optional ColabFold multiple-sequence alignments.

## Background

RoseTTAFold3 ([Corley et al., 2025](https://doi.org/10.1101/2025.08.14.670328)) predicts the joint 3D structure of a biomolecular assembly from the sequences and chemical components it contains. It is the latest entry in the RoseTTAFold lineage from the Baker and DiMaio labs at the Institute for Protein Design, and like AlphaFold3 and Boltz-2 it folds proteins, nucleic acids, and small-molecule ligands within a single model. The preprint reports that an improved treatment of chirality narrows the performance gap between RF3 and the closed-source AlphaFold3 on biomolecular benchmarks.

Architecturally RF3 builds on the new [AtomWorks](https://github.com/RosettaCommons/atomworks) data framework introduced alongside it in the preprint, and uses an AlphaFold3 style trunk together with a diffusion module that samples several candidate structures per complex from random noise. The best sample is selected by a composite ranking score that combines interface pTM, overall pTM, and a clash penalty. Alongside the predicted coordinates, RF3 reports per-residue and overall pLDDT, per-chain confidence, predicted aligned error (PAE) and predicted distance error (PDE), chain-pair PAE and PDE matrices for multi-chain inputs, and a boolean flag for steric clashes.

The reference implementation is open-sourced as part of the [RosettaCommons/foundry](https://github.com/RosettaCommons/foundry) monorepo under the BSD-3-Clause license, with model weights served openly from the IPD file server. It was developed at the [Institute for Protein Design](https://www.ipd.uw.edu/) (UW).

## Tools

### RoseTTAFold3 Prediction (`rf3-prediction`)

Predicts the 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains. The assembly is folded by RF3 and returned as a predicted `Structure` per complex with confidence metrics, including average pLDDT, pTM, interface pTM for multi-chain inputs, per-chain pTM, an overall and chain-pair PAE and PDE in angstroms, a composite ranking score, and a steric-clash flag.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-DNA and protein-RNA complexes or protein-ligand binding poses. Within this toolkit it is also the model whose architecture has explicit chirality representations built in, which is relevant when modelling chiral small molecules, D-amino-acid residues, or peptides where stereochemistry matters. For multi-chain inputs the reported chain-pair PAE and PDE matrices together with interface pTM estimate how confidently the components are placed relative to each other, useful for ranking or filtering predicted interfaces.

#### Usage Tips

- **`use_msa` defaults to `True`.** A ColabFold search generates an MSA for each protein chain. Set it `False` for single-sequence prediction, or attach precomputed MSAs to the input.
- **Diffusion samples are ranked by `ranking_score`.** `diffusion_batch_size` (default `5`) independent samples are drawn per complex. The best by `ranking_score = 0.8*iptm + 0.2*ptm - 100*has_clash` is returned, with `num_steps` (default `50`) controlling the denoising step count.
- **`n_recycles` (default `10`) trades accuracy for time.** More recycling iterations refine the prediction at higher runtime. Leave the upstream default of `10` unless you have a specific reason to lower it.
- **Cyclic chains.** Mark chains as cyclic (head-to-tail) with `cyclic_chains=["A", ...]`.
- **No template or conformer conditioning.** RF3 can condition on input coordinates (templates, holo ligand conformers), but this wrapper accepts only sequences, SMILES, and CCD codes — no coordinate input — so those upstream options are not exposed.
- **No per-token PAE matrix.** Unlike Boltz-2 and AlphaFold3, RF3 emits only chain-pair PAE aggregates (`avg_pae`, `chain_pair_pae`, `chain_pair_pae_min`) and a separate `pde` (predicted distance error). The inherited `include_pae_matrix` toggle is rejected by `RF3Config`.
- **Multi-modal inputs.** Protein, DNA, RNA, and ligand entities are supported.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every RF3 tool in this toolkit (`rf3-prediction`).

- **Requires a GPU.** RF3 runs through a PyTorch backend and needs an NVIDIA GPU. CPU execution is not practical.
- **Open weights.** The RF3 checkpoint is downloaded automatically from the IPD file server during environment setup and lands in the proto-tools weights cache. No request form or token is required.
- **Predictions are stochastic.** Structures come from a diffusion process, so repeated runs vary unless sampling is seeded. The wrapper advances the seed per complex within a batch so duplicate inputs in one call still diversify.
