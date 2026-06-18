<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESMFold

![ESMFold](https://proto-bio.github.io/proto-assets/images/tool/esmfold/hero.png)

> [!NOTE]
> **License:** ESMFold is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/facebookresearch/esm/blob/main/LICENSE) for full terms.

## Overview

First released in 2022, ESMFold is Meta AI's protein language model based structure predictor. It folds a protein from its sequence using the embeddings of ESM-2 in place of a multiple-sequence alignment, making it roughly an order of magnitude faster than alignment-based methods like AlphaFold2 at competitive but generally somewhat lower accuracy. This toolkit provides two tools: structure prediction for proteins and complexes from sequence, and a differentiable confidence pass that returns the gradient of an ESMFold confidence loss for sequence design.

## Background

ESMFold ([Lin et al., 2023](https://doi.org/10.1126/science.ade2574)) predicts a protein's 3D structure directly from its amino-acid sequence, without the multiple-sequence alignment (MSA) that AlphaFold2 depends on. AlphaFold2 infers which residues are in contact by reading coevolution across an alignment of homologous sequences. ESMFold instead relies on the ESM-2 protein language model, which has already internalized those evolutionary patterns by pre-training on hundreds of millions of natural sequences, so it works from the lone sequence with no alignment built at inference time. Skipping the alignment search makes ESMFold roughly an order of magnitude faster than AlphaFold2, at some cost in accuracy on targets where a deep, diverse MSA would otherwise help.

The sequence first runs through a frozen ESM-2 transformer (the released model uses the 3-billion-parameter ESM-2), which produces a per-residue representation. A folding trunk, a simplified stand-in for AlphaFold2's Evoformer, refines that representation, and a structure module reused essentially unchanged from AlphaFold2 then places each residue as a rigid backbone frame to produce all-atom coordinates. The whole prediction is recycled through these stages several times. Alongside the coordinates, ESMFold reports calibrated confidence: a per-residue predicted local distance difference test (pLDDT) for local reliability, a predicted aligned error (PAE) for the expected error in one residue's position when the structure is aligned on another, and a predicted template-modeling score (pTM) for overall fold confidence.

Meta AI open-sources the reference implementation at [facebookresearch/esm](https://github.com/facebookresearch/esm) under the MIT license; the released model is the `esmfold_v1` checkpoint, whose structure module is taken from the OpenFold reimplementation of AlphaFold2. Because the language model carries the structural signal, ESM-2's perplexity on a sequence correlates with how accurate the predicted structure will be, and accuracy continues to improve as the ESM-2 backbone is scaled up. ESMFold's speed made its headline application possible: Meta AI folded over 600 million metagenomic protein sequences and released them as the [ESM Metagenomic Atlas](https://esmatlas.com).

### Learning Resources

- [ESM Metagenomic Atlas Blog Post](https://ai.meta.com/blog/protein-folding-esmfold-metagenomics/) (Meta AI) - an overview blog post of the ESM Metagenomic Atlas, which contains structure predictions for nearly the entire [MGnify](https://www.ebi.ac.uk/metagenomics/) database of metagenomic sequences.

## Tools

### ESMFold Structure Prediction (`esmfold-prediction`)

Predicts the 3D structure of one or more protein chains from their sequences. Each input complex (a single chain, or several chains folded together) is run through ESMFold and returned as a predicted `Structure` per complex with confidence metrics: per-residue pLDDT, a predicted TM-score (pTM), and predicted aligned error.

#### Applications

This tool folds a protein sequence into a 3D model for structural analysis or as input to downstream structure based tools. Because ESMFold does not use an MSA, it is well suited to de novo or heavily engineered sequences that have no natural homologs for an alignment to capture.

#### Usage Tips

- **No MSA or template search is used.** ESMFold does not incorporate MSAs into the prediction. There is no `use_msa` option (unlike Boltz-2 or Protenix), and passing one raises an error; the inherited `msas` input, by contrast, is ignored if supplied, emitting a single logged warning.
- **Multi-chain complexes are approximated with an internal glycine linker.** `chain_linker` (default 25 glycines) joins chains before folding and is stripped from the output; this works best for homomeric assemblies and is unreliable for true hetero-complexes. Use AlphaFold3, Boltz2, Chai-1, or Protenix for those.
- **Protein sequences only, with a hard cap of 2,400 residues per complex.** DNA, RNA, and ligands are not supported; `X` is allowed for unknown residues. The cap is enforced against the linked length actually folded: the sum of all chain residues plus the inter-chain `chain_linker` inserted between them (`len(chain_linker) * (chains - 1)`, 25 residues per junction by default). A multi-chain complex whose bare residues sum to just under 2,400 can still exceed the cap.
- **Confidence is reported as pLDDT, pTM, and PAE.** Average pLDDT (0 to 1) is the primary per-structure quality metric; set `include_pae_matrix` to attach the full per-residue PAE matrix.

### ESMFold Gradient (`esmfold-gradient`)

Runs a single differentiable ESMFold confidence pass: one forward-and-backward gradient evaluation, not an iterative design loop. For one or more designated chains, a relaxed `(L, 20)` amino-acid distribution replaces the discrete sequence, and ESMFold folds the complex under that soft input. The resulting pLDDT, pTM, and PAE terms are combined into one weighted scalar loss, and a single backward pass returns its gradient with respect to the input logits, along with the loss value, the per-term metrics, and the predicted `Structure`.

#### Applications

This tool supplies the loss and gradient signal that gradient-based or MCMC sequence-design loops optimize for foldability: minimizing the confidence loss pushes a relaxed sequence toward one ESMFold predicts will fold well. With `compute_gradient=False` it instead provides forward-only confidence scoring (loss, metrics, and predicted structure) of a candidate sequence for ranking or filtering.

#### Usage Tips

- **One pass per call; this tool is not an optimization loop.** It evaluates a single relaxed sequence. Drive it from a sequence-design optimizer, or call it repeatedly, to actually design a sequence.
- **`compute_gradient` defaults to `True`.** It runs a forward and backward pass and returns the gradient with respect to the input logits; set it `False` for forward-only scoring (`gradient=None`). The loss, metrics, and predicted structure are identical in both modes.
- **`loss_weights` selects and weights the confidence terms.** Non-negative weights over `plddt`, `ptm`, and `pae` (default `{"plddt": 1.0}`); terms with weight `0.0` are skipped, and all-zero weights short-circuit to a zero gradient with `loss=0.0`.
- **`logits` and the returned `gradient` share canonical amino-acid order `ACDEFGHIKLMNPQRSTVWY`.** Every chain listed in `target_chain_indices` must have length `len(logits)`; non-target chains fold normally with their fixed sequences.
- **`soft` and `hard` trade smoothness for discreteness.** The default (`soft=1.0`, `hard=0.0`) uses pure soft probabilities for smooth optimization; set `hard=1.0` for a straight-through estimator (the forward pass sees argmax tokens while gradients still flow through the soft probabilities).

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESMFold tool in this toolkit (`esmfold-prediction`, `esmfold-gradient`).

- **Requires a GPU.** Both tools run ESMFold through a PyTorch backend and need an NVIDIA GPU (roughly 16 GB of VRAM or more for longer sequences); CPU execution is not practical.
- **`max_batch_residues` is a starting cap, not a hard ceiling.** On CUDA OOM the wrapper halves the cap (floor = longest single complex) and re-splits the offending sub-batch, so the default `1200` is usually fine to leave in place.
- **MSA-free and single-sequence.** ESMFold folds from one sequence with no alignment or template search. Accuracy is generally lower than MSA-based methods on targets where a deep, diverse MSA would help.
- **`num_recycles` (default `4`) applies to both tools.** Each recycling iteration refines the structure; raising it improves accuracy at higher runtime.
