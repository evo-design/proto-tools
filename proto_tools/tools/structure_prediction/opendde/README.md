<a href="https://bio-pro.mintlify.app/tools/structure-prediction/opendde"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# OpenDDE

![OpenDDE](https://proto-bio.github.io/proto-assets/images/tool/opendde/hero.png)

> [!NOTE]
> **License:** OpenDDE is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/aurekaresearch/OpenDDE/blob/main/LICENSE) for full terms.

## Overview

OpenDDE is [Aureka AI Research](https://github.com/aurekaresearch)'s open-source, all-atom biomolecular co-folding foundation model in the AlphaFold3 family: a single model that predicts the joint 3D structure of complexes mixing proteins, DNA, RNA, small-molecule ligands, and ions. This toolkit runs OpenDDE structure prediction on a local GPU, with optional multiple-sequence alignments, and returns per-complex confidence metrics.

## Background

OpenDDE ([Aureka AI Research, 2026](https://arxiv.org/abs/2607.03787)) predicts the joint 3D structure of a biomolecular assembly from the sequences and chemical components it contains. It is an openly licensed, all-atom co-folding model where one model folds complexes that mix proteins, DNA, RNA, and small-molecule ligands and predicts how those components are arranged relative to one another. Each protein chain can be paired with a multiple-sequence alignment (MSA) of evolutionarily related sequences, whose covariation patterns supply the evolutionary signal the model uses to place residues.

Architecturally, OpenDDE follows AlphaFold3: it carries a single representation of the input tokens and a pairwise representation over token pairs, refines them through a Pairformer-style trunk, and generates all-atom coordinates with a diffusion module that starts from noise and iteratively denoises into a structure. Several structures can be sampled per complex and ranked by a confidence score. Predicted confidence includes a per-residue predicted local distance difference test (pLDDT) for local reliability, a global predicted distance error (gPDE) for the relative placement of tokens, and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores that summarize overall and interface accuracy, together with an overall ranking score used to select the best sample.

The reference implementation is open-sourced at [aurekaresearch/OpenDDE](https://github.com/aurekaresearch/OpenDDE), with both the code and the model parameters released under the Apache-2.0 license for academic and commercial use. It builds on ideas and components from [Protenix](https://github.com/bytedance/Protenix), [OpenFold](https://github.com/aqlaboratory/openfold), and [ColabFold](https://github.com/sokrypton/ColabFold). Two checkpoints are released: a general-purpose model and an antibody-antigen-tuned variant. It was developed by Aureka AI Research as an open drug-discovery engine spanning structure prediction, design, and optimization.

### Learning Resources

- [OpenDDE Technical Report](https://huggingface.co/aurekaresearch/OpenDDE/blob/main/docs/OpenDDE_Technical_reports.pdf) (Aureka AI Research) - the technical report describing OpenDDE's architecture, training data, and benchmark results.

## Tools

### OpenDDE Structure Prediction (`opendde-prediction`)

Predicts the 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains; the assembly is folded by OpenDDE and returned as a predicted `Structure` per complex with confidence metrics: average pLDDT, pTM, interface pTM for multi-chain complexes, a global predicted distance error, and a ranking score.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-DNA and protein-RNA complexes or protein-ligand binding poses, and the `opendde_abag` checkpoint targets antibody-antigen complexes specifically. Running it on a multi-chain complex also estimates how confidently the components are placed relative to each other through interface pTM and the global predicted distance error, which is informative for ranking predicted interfaces before trusting them downstream.

#### Usage Tips

- **`model_checkpoint` selects the weights.** Pass a bundled model name — `opendde_v1` (default, general-purpose) or `opendde_abag` (antibody-antigen tuned) — both auto-downloaded into `PROTO_MODEL_CACHE` on first inference; or pass a path to a custom `.pt` checkpoint to fold with your own weights.
- **`use_msa` defaults to `True`.** An MMseqs2 homology search generates an MSA for each protein chain; set it `False` to fold single-sequence, or attach precomputed MSAs to the input, which always take precedence. OpenDDE's own internal MSA search runs only when `use_msa=True` and no MSAs are supplied.
- **`num_samples` draws independent diffusion samples.** OpenDDE draws `num_samples` (default `1`) structures per complex and keeps the best by ranking score, so raising it explores more candidate conformations at proportional cost. Predictions are stochastic; set `seed` for reproducibility.
- **`num_steps` and `num_cycles` trade accuracy for time.** `num_steps` (default `200`) sets the number of diffusion denoising steps and `num_cycles` (default `10`) sets the recycling iterations; higher values refine the prediction but increase runtime.
- **`use_template` and `use_rna_msa` enable OpenDDE's extra pipelines.** Both default to `False`; enable `use_template` for OpenDDE's template search and `use_rna_msa` for its RNA MSA pipeline when folding RNA-containing complexes.
- **Confidence is reported as pLDDT, pTM, ipTM, gPDE, and a ranking score.** `avg_plddt` (0 to 100) is the primary per-structure quality metric; `iptm` is 0.0 for single-chain inputs, and `gpde` is in angstroms. OpenDDE writes only scalar summary confidences, so no per-token PAE matrix is available and the inherited `include_pae_matrix` is ignored.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every OpenDDE tool in this toolkit (`opendde-prediction`).

- **Requires a GPU.** OpenDDE runs through a PyTorch backend and needs an NVIDIA GPU; CPU execution is not practical.
- **Open AlphaFold3-style co-folding model.** OpenDDE releases both code and weights under Apache-2.0 for academic and commercial use, and follows the AlphaFold3 diffusion architecture like Boltz-2 and Protenix. Its `opendde_abag` checkpoint additionally specializes in antibody-antigen complexes.
- **Predictions are stochastic.** Structures come from a diffusion process, so repeated runs vary unless sampling is seeded.
- **Early preview upstream.** Checkpoints and interfaces may change, so pin the version you validate against.
