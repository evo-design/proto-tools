<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESMFold2

![ESMFold2](https://proto-bio.github.io/proto-assets/images/tool/esmfold2/hero.png)

> [!NOTE]
> **License:** ESMFold2 is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/Biohub/esm/blob/main/LICENSE.md) for full terms.

## Overview

ESMFold2 is [Biohub](https://biohub.ai)'s all-atom biomolecular structure predictor and successor to ESMFold. It folds complexes containing proteins, DNA, RNA, and small-molecule ligands, from sequence alone or with an optional multiple-sequence alignment (MSA). This toolkit runs ESMFold2 structure prediction on a local GPU and provides access to both the MSA-capable `esmfold2` checkpoint and the inference-optimized single-sequence `esmfold2-fast` checkpoint through a single tool.

## Background

ESMFold2 ([Candido et al., 2026](https://biohub.ai/papers/esm_protein.pdf)) extends the ESM family from protein-only single-sequence folding to all-atom prediction of biomolecular complexes. Where the original ESMFold ([Lin et al., 2023](https://doi.org/10.1126/science.ade2574)) used the ESM-2 protein language model as a learned substitute for an MSA and folded a single protein chain into a backbone, ESMFold2 supports proteins, DNA, RNA, small-molecule ligands, modified residues, and covalent bonds in a single joint prediction, comparable in scope to AlphaFold3 and Boltz-2. The model can be run in single-sequence mode, or, when an MSA is available for a protein chain, conditioned on the alignment to recover the evolutionary signal that aids prediction of difficult or sparsely-engineered targets.

Architecturally, ESMFold2 conditions on representations from the frozen ESMC 6B language model, pools them into a two-dimensional pair representation refined through a stack of folding layers with a stabilized recurrent update, and concludes with a diffusion transformer that denoises directly into all-atom coordinates. Two inference-time parameters, the number of refinement loops through the folding stack and the number of diffusion sampling steps, trade computation time for accuracy and can materially improve predictions on difficult targets, especially antibody-antigen complexes. Alongside the structure, ESMFold2 reports calibrated confidence: a per-residue predicted local distance difference test (pLDDT), a predicted aligned error (PAE) for the relative placement of any two tokens, and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores that summarize overall and interface accuracy.

Two checkpoints are available. `esmfold2` is the larger, MSA-capable model recommended for difficult or long targets where alignment signal aids prediction; `esmfold2-fast` is an inference-optimized single-sequence variant intended for high-throughput applications. Both are distributed under the MIT license at [Biohub/esm](https://github.com/Biohub/esm), the consolidated package that also distributes ESM3 and ESM C.

### Learning Resources

- [ESMFold2 model card](https://huggingface.co/biohub/ESMFold2) (Biohub) - architecture details, training data, benchmark results, and intended-use guidance for the MSA-capable checkpoint.

## Tools

### ESMFold2 Structure Prediction (`esmfold2-prediction`)

Predicts the all-atom 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains (with optional chain-level modifications); the assembly is folded by ESMFold2 and returned as a predicted `Structure` per complex with confidence metrics: pLDDT, pTM, interface pTM (for multi-chain complexes), and predicted aligned error.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-protein, protein-DNA, protein-RNA, and protein-ligand complexes, including antibody-antigen interfaces where ESMFold2 is reported to be competitive with AlphaFold3. Running it on a multi-chain complex also estimates how confidently the components are placed relative to each other through interface pTM and PAE, which is informative for assessing predicted interfaces.

#### Usage Tips

- **`model_checkpoint` selects the variant.** `esmfold2-fast` (default) is the inference-optimized single-sequence model and is appropriate for most high-throughput applications; select `esmfold2` (with `use_msa=True`, or by attaching precomputed `msas` on the input) for the larger MSA-capable model on difficult or long targets. Setting `use_msa=True` with `esmfold2-fast` raises a validation error, and `msas` supplied with `esmfold2-fast` are ignored with a logged warning.
- **`num_loops` (default `3`) and `num_sampling_steps` (default `50`) trade computation for accuracy.** Both parameters materially affect prediction quality, with the largest gains on difficult targets such as antibody-antigen complexes. Increasing either improves accuracy but extends runtime; decreasing them accelerates high-throughput screens at some accuracy cost.
- **Multi-modal inputs.** Protein, DNA, RNA, and small-molecule ligand chains are supported; ligands can be specified by CCD code or SMILES, and chain-level modifications are accepted. SMILES-based ligand input is supported but currently has known accuracy issues; CCD codes are recommended.
- **Confidence is reported as pLDDT, pTM, ipTM, and PAE.** Mean pLDDT (0 to 1) is the primary per-structure quality metric; `iptm` is emitted only for multi-chain complexes, and `avg_pae` is in angstroms (0 to about 32). Set `include_pae_matrix=True` to attach the full per-token PAE matrix.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESMFold2 tool in this toolkit (`esmfold2-prediction`).

- **Requires a GPU.** ESMFold2 runs through a PyTorch backend and needs an NVIDIA GPU; CPU execution is not practical.
- **Shared `biohub_esm` environment.** ESMFold2 is part of the consolidated [Biohub/esm](https://github.com/Biohub/esm) package and shares its standalone environment with the ESM3 and ESM C toolkits, so installing any one of them provisions the others.
- **AlphaFold3-style diffusion with optional MSAs.** Predictions are stochastic, so set `seed` for reproducibility across runs. MSAs are only consumed by the `esmfold2` checkpoint; the `esmfold2-fast` checkpoint is single-sequence by construction.
- **Structure prediction only.** This toolkit provides ESMFold2's structure prediction capability; the broader ESM family's language-model, generation, and embedding capabilities are provided by the sibling ESM3 and ESM C toolkits.
