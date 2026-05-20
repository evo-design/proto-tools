<a href="https://bio-pro.mintlify.app/tools/structure-prediction/chai1"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Chai-1

![Chai-1](https://framerusercontent.com/images/GHT9QAB4zu0jdGsoQxwESRTW8.png)

> *Image source: [Chai Discovery](https://www.chaidiscovery.com/)*

> [!NOTE]
> **License:** Chai-1 is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/chaidiscovery/chai-lab/blob/main/LICENSE) for full terms.

## Overview

First released in 2024, Chai-1 is [Chai Discovery](https://www.chaidiscovery.com/)'s multi-modal foundation model for molecular structure prediction, folding proteins together with small-molecule ligands and glycans. This toolkit runs Chai-1 to predict the joint 3D structure of protein, ligand, and glycan complexes from sequence, optionally conditioned on ESM embeddings and ColabFold multiple-sequence alignments.

## Background

Chai-1 ([Chai Discovery, 2024](https://doi.org/10.1101/2024.10.10.615955)) predicts the joint 3D structure of a biomolecular assembly from the sequences and chemical components it contains. It is a multi-modal foundation model that folds proteins together with small-molecule ligands, nucleic acids, glycans, and covalent modifications in a single model. Each protein chain can be conditioned on evolutionary signal, either through a multiple-sequence alignment (MSA) of related sequences or through embeddings from an ESM protein language model.

Internally, Chai-1 follows the all-atom co-folding approach popularized by AlphaFold3. It tokenizes the assembly the same way, with one token per amino-acid residue or nucleotide and one token per atom for ligands and modified residues. A trunk network then builds and refines token and pairwise representations, optionally conditioned on the MSA and ESM embeddings, and a diffusion module generates the all-atom coordinates by starting from noise and iteratively denoising into a structure. Several structures are sampled per input and ranked by an aggregate confidence score. Predicted confidence includes a per-atom predicted local distance difference test (pLDDT) for local reliability, a predicted aligned error (PAE) for the relative placement of any two tokens, and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores that summarize overall and interface accuracy.

The reference implementation is open-sourced by [Chai Discovery](https://www.chaidiscovery.com/) at [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab), with both the code and the model weights released under the Apache-2.0 license for academic and commercial use, including drug discovery. Chai Discovery also runs the model as a hosted web platform at [lab.chaidiscovery.com](https://lab.chaidiscovery.com).

### Learning Resources

- [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab) (Chai Discovery) - the official repository and inference code, linking the technical report and the hosted [Chai Discovery web platform](https://lab.chaidiscovery.com) for running predictions in the browser.

## Tools

### Chai-1 Structure Prediction (`chai1-prediction`)

Predicts the 3D structure of a biomolecular complex. Each input complex can combine protein, ligand, and glycan chains; the assembly is folded by Chai-1 and returned as a predicted `Structure` per complex with confidence metrics: average pLDDT, pTM, interface pTM, predicted aligned error, and an overall confidence score.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-ligand binding poses and glycosylated proteins, which makes it well suited to drug-discovery screening and modeling carbohydrate-decorated targets. For a multi-chain complex it also reports how confidently the chains are placed relative to one another: interface pTM (ipTM) gives a single 0-to-1 score for the overall inter-chain arrangement, and the cross-chain blocks of the PAE matrix show which inter-chain regions are positioned confidently versus uncertainly, so you can rank or filter predicted complexes before trusting a pose downstream.

#### Usage Tips

- **Total length is capped at 2,048 tokens per complex** (1 per amino-acid residue, 1 per heavy atom for ligands and glycans); longer inputs are rejected.
- **`use_esm_embeddings` defaults to `True`.** Chai-1 conditions on embeddings from an ESM protein language model; they are used with or without an MSA.
- **`use_msa` defaults to `True`.** A ColabFold search generates an MSA for each protein chain; set it `False` for single-sequence prediction, or attach precomputed MSAs to the input.
- **Sampling and refinement are configurable.** `num_diffn_samples` (default `5`) independent samples are drawn per complex and the best is kept by `confidence_score`; `num_diffn_timesteps` (default `200`) sets the denoising steps and `num_trunk_recycles` (default `3`) trades accuracy for runtime.
- **Confidence is reported as pLDDT, pTM, ipTM, PAE, and a confidence score.** `avg_plddt`, the primary metric, is on a 0 to 1 scale; ipTM is meaningful only for multi-chain complexes. Set `include_pae_matrix` to attach the full per-token PAE matrix.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Chai-1 tool in this toolkit (`chai1-prediction`).

- **Requires a GPU.** Chai-1 runs through a PyTorch backend and needs an NVIDIA GPU; CPU execution is not practical. `low_memory` (default `True`) streams features per sample to reduce peak GPU memory at some cost in speed.
- **Protein, ligand, and glycan only** The Chai-1 model additionally supports DNA, RNA, and covalent modifications; this toolkit currently wraps protein, ligand, and glycan prediction. Use AlphaFold3, Boltz-2, or Protenix for nucleic-acid complexes.
- **Predictions are stochastic.** Structures come from a diffusion process; set `seed` for reproducible sampling. `recycle_msa_subsample` and unseeded runs are intentionally non-deterministic.
