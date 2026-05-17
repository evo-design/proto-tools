<a href="https://bio-pro.mintlify.app/tools/structure-prediction/protenix"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Protenix

![Protenix](https://github.com/bytedance/Protenix/raw/main/assets/protenix_predictions.gif)

> *Image source: [bytedance/Protenix](https://github.com/bytedance/Protenix)*

> [!NOTE]
> **License:** Protenix is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/bytedance/Protenix/blob/main/LICENSE) for full terms.

## Overview

Protenix is [ByteDance](https://www.bytedance.com/)'s open-source reproduction of AlphaFold3: a trainable PyTorch model that predicts the joint 3D structure of complexes mixing proteins, DNA, RNA, and small-molecule ligands, including modified residues. This toolkit runs Protenix structure prediction, with optional ColabFold multiple-sequence alignments and a choice of the base, mini, and tiny model variants that trade accuracy for speed.

## Background

Protenix ([ByteDance Research, 2025](https://doi.org/10.1101/2025.01.08.631967)) predicts the joint 3D structure of a biomolecular assembly from the sequences and chemical components it contains. It is a trainable, openly licensed reproduction of the AlphaFold3 architecture: like AlphaFold3, one model folds complexes that mix proteins, DNA, RNA, and small-molecule ligands, and predicts how those components are arranged relative to one another. Each protein chain can be paired with a multiple-sequence alignment (MSA) of evolutionarily related sequences, whose covariation patterns supply the evolutionary signal the model uses to place residues.

Architecturally, Protenix follows AlphaFold3 rather than AlphaFold2: it carries a single representation of the input tokens and a pairwise representation over token pairs, refines them through a Pairformer trunk, and generates all-atom coordinates with a diffusion module that starts from noise and iteratively denoises into a structure, in place of AlphaFold2's structure module. Several structures are sampled per random seed and ranked by a confidence score. Protenix is distributed in several sizes: full-parameter base models for highest accuracy, and lighter mini and tiny variants for faster, lower-memory prediction; the `mini_esm` and `mini_ism` variants replace the MSA with learned embeddings — from the ESM-2 protein language model or the ISM inverse-structure model, respectively — so they can fold without an alignment. Predicted confidence includes a per-residue predicted local distance difference test (pLDDT) for local reliability, a predicted aligned error (PAE) for the relative placement of any two tokens, a global predicted distance error (gPDE), and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores that summarize overall and interface accuracy.

The reference implementation is open-sourced at [bytedance/Protenix](https://github.com/bytedance/Protenix), with both the code and the model parameters released under the Apache-2.0 license for academic and commercial use. It was developed by [ByteDance](https://www.bytedance.com/)'s AI4Science team as a comprehensive reproduction of AlphaFold3, trained on comparable data to reach competitive accuracy across protein, nucleic-acid, and protein-ligand benchmarks.

### Learning Resources

- [bytedance/Protenix](https://github.com/bytedance/Protenix) (ByteDance) - the official repository, with a model card for each variant, benchmark results across protein, nucleic-acid, and ligand tasks, and a link to the hosted [Protenix web server](https://protenix-server.com).

## Tools

### Protenix Structure Prediction (`protenix-prediction`)

Predicts the 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains, with optional post-translational and nucleotide modifications; the assembly is folded by Protenix and returned as a predicted `Structure` per complex with confidence metrics: average pLDDT, pTM, interface pTM, per-chain and pairwise-chain scores, a global predicted distance error, and predicted aligned error.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-DNA and protein-RNA complexes, protein-ligand binding poses, and chains carrying modified residues. For a multi-chain complex it also reports how confidently the chains are placed relative to one another: interface pTM (ipTM) gives a single 0-to-1 score for the overall inter-chain arrangement, per-chain-pair ipTM scores each individual interface, and the cross-chain blocks of the PAE matrix show which specific inter-chain regions are positioned confidently versus uncertainly. These let you rank or filter predicted complexes and judge whether a docking pose or binding interface is reliable before trusting it downstream.

#### Usage Tips

- **`model_name` selects the accuracy/speed trade-off.** The default `protenix_base_default_v1.0.0` is the most accurate (10 Pairformer cycles, 200 diffusion steps); the `mini` and `tiny` variants are far faster with fewer cycles and steps, and `protenix_mini_esm_v0.5.0` / `protenix_mini_ism_v0.5.0` use protein language-model embeddings for MSA-free prediction.
- **`use_msa` defaults to `True`.** A ColabFold search generates an MSA for each protein chain; set it `False`, attach precomputed MSAs, or use an ESM/ISM mini variant to skip alignments entirely.
- **Diffusion sampling is controlled by `seeds` and `num_diffusion_samples`.** Protenix draws `num_diffusion_samples` (default `5`) structures per seed and keeps the best by ranking score; the total number of candidates is `len(seeds)` times `num_diffusion_samples`. Setting `seed` overrides `seeds` with a single value for reproducibility.
- **`num_pairformer_cycles` (default `10`) and `num_diffusion_steps` (default `200`) trade accuracy for time.** More cycles and steps refine the prediction but increase runtime. These defaults are applied for every `model_name`; the mini and tiny variants run their native low-step schedules best, so lower these yourself for faster runs with them.
- **Confidence is reported as pLDDT, pTM, ipTM, gPDE, and PAE.** `confidence_score`, the ranking score and primary metric, selects the best sample; `avg_plddt` is on a 0 to 1 scale and PAE and gPDE are in angstroms. `has_clash` flags steric clashes. Set `include_pae_matrix` to attach the full per-token PAE matrix.
- **Modified residues are supported.** Protein PTMs and DNA/RNA modifications are passed through as CCD codes, as in AlphaFold3.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Protenix tool in this toolkit (`protenix-prediction`).

- **Requires a GPU.** Protenix runs through a PyTorch backend and needs an NVIDIA GPU; base models are memory-intensive and slower, while mini and tiny variants run on more modest hardware. CPU execution is not practical.
- **Open AlphaFold3 reproduction.** Unlike AlphaFold3, whose weights are gated and non-commercial, Protenix releases both code and weights under Apache-2.0 for academic and commercial use. Like Boltz-2 it follows the AlphaFold3 diffusion architecture, and additionally accepts modified residues.
- **Predictions are stochastic.** Structures come from a diffusion process; set `seed` (or `seeds`) for reproducible sampling.
