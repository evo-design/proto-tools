<a href="https://bio-pro.mintlify.app/tools/structure-prediction/alphafold3"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# AlphaFold3

![AlphaFold3](https://raw.githubusercontent.com/CHTC/chtc-website-source/refs/heads/master/images/AlphaFold3.jpg)

> *Image source: [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)*

> [!NOTE]
> **License:** AlphaFold3 uses CC-BY-NC-SA-4.0 for code and Custom (AlphaFold 3 Model Parameters Terms of Use) for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Model weights are not publicly distributed and must be requested from the provider. Please refer to the [code license](https://github.com/google-deepmind/alphafold3/blob/main/LICENSE) and [model weights license](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) for full terms.

## Overview

AlphaFold3 (AF3) is Google DeepMind and Isomorphic Labs' third-generation AlphaFold, extending structure prediction beyond single proteins to the joint 3D structure of proteins together with DNA, RNA, and small-molecule ligands in one model. This toolkit runs AlphaFold3 so you can fold these mixed-molecule complexes from sequence through the proto framework, provided you have access to the gated model weights on your machine.

## Background

AlphaFold3 ([Abramson et al., 2024](https://doi.org/10.1038/s41586-024-07487-w)) predicts the joint 3D structure of a biomolecular assembly from the sequences and chemical components it contains. It extends AlphaFold2 beyond single proteins: one model folds complexes that mix proteins, DNA, RNA, and small-molecule ligands, and predicts how those parts are arranged relative to one another. As in AlphaFold2, each protein chain is paired with a multiple-sequence alignment (MSA) of related sequences, whose covariation patterns give the model an evolutionary signal for placing residues.

Internally, AlphaFold3 represents the assembly as a set of tokens: one per amino-acid residue or nucleotide, and one per atom for ligands and modified residues. It then learns a representation of every token and of every token pair. Where AlphaFold2 leaned on the large MSA-centric Evoformer, AlphaFold3 de-emphasizes the MSA, handling it in a separate preliminary module rather than iterating it through the deep trunk, and does most of its work in the 'Pairformer', which iteratively refines the token and pair representations through geometry-inspired "triangle attention" updates. The final representations are then fed into a diffusion module that iteratively denoises all-atom coordinates starting from random noise. Run from several random seeds, it produces multiple candidate structures, and the highest-confidence candidate is returned as the final prediction. In addition, AlphaFold3 reports calibrated confidence metrics such as the per-atom predicted local distance difference test (pLDDT) for local reliability, a predicted aligned error (PAE) for how well any two tokens are placed relative to each other, and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores for overall and interface accuracy.

### Learning Resources

- [The Illustrated AlphaFold](https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/) (by [Elana Simon](https://elanapearl.github.io/) and [Jake Silberg](https://jsilbergds.github.io/)) - a visual, diagram-driven walkthrough of the AlphaFold3 architecture, from input preparation through representation learning to structure prediction.
- [AlphaFold 3 predicts the structure and interactions of all of life's molecules](https://blog.google/technology/ai/google-deepmind-isomorphic-alphafold-3-ai-model/) (Google DeepMind and Isomorphic Labs) - the official announcement, with an accessible overview of what AlphaFold3 predicts and how it extends earlier models.

## Tools

### AlphaFold3 Structure Prediction (`alphafold3-prediction`)

Predicts the 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains; the assembly is folded by AlphaFold3 and returned as a predicted `Structure` per complex with confidence metrics: per-residue pLDDT, pTM, interface pTM for multi-chain complexes, and predicted aligned error.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-DNA and protein-RNA complexes or protein-ligand binding poses. Running it on a multi-chain complex also estimates how confidently the components are placed relative to each other through interface pTM and PAE, which is informative for assessing predicted interfaces.

#### Usage Tips

- **`use_msa` defaults to `True`.** An MSA is then generated by a ColabFold search for protein chains; set it `False` to skip the search, or attach precomputed MSAs to the input.
- **Diffusion sampling is controlled by `seeds` and `num_diffusion_samples`.** AlphaFold3 draws `num_diffusion_samples` (default `5`) structures per seed and keeps the best by ranking score, so a single seed is often enough; the total number of candidates is `len(seeds)` times `num_diffusion_samples`.
- **`num_recycles` (default `10`) trades accuracy for time.** More recycling iterations refine the prediction but increase runtime.
- **Confidence is reported as pLDDT, pTM, ipTM, and PAE.** Average pLDDT (0 to 1) is the primary per-structure quality metric; ipTM is populated only for multi-chain complexes.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every AlphaFold3 tool in this toolkit (`alphafold3-prediction`).

- **Requires a GPU.** AlphaFold3 needs an NVIDIA GPU; CPU execution is not practical.
- **Model weights are gated.** AlphaFold3 weights are not publicly distributed; access is restricted to non-commercial research and must be requested from Google DeepMind through their form, then made available to the tool before it can run.
