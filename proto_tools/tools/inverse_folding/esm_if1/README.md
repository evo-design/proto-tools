<a href="https://bio-pro.mintlify.app/tools/inverse-folding/esm-if1"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESM-IF1

![ESM-IF1](https://proto-bio.github.io/proto-assets/images/tool/esm_if1/hero.png)

> [!NOTE]
> **License:** ESM-IF1 is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/facebookresearch/esm/blob/main/LICENSE) for full terms.

## Overview

Released in 2022, ESM-IF1 is an inverse-folding model that predicts which amino-acid sequences fold into a given protein backbone. It was the first inverse-folding model trained at scale on millions of AlphaFold2-predicted structures, and it generalizes to complexes and binding interfaces. This toolkit also includes ProteinDPO, a fine-tuned variant of ESM-IF1 aligned to experimental stability measurements to favor more stable designs. Both design sequences for a backbone and score how well a sequence fits a structure.

## Background

ESM-IF1 ([Hsu et al., 2022](https://proceedings.mlr.press/v162/hsu22a.html)) solves the inverse-folding problem: given a protein backbone, it predicts an amino-acid sequence that will fold into that structure. This is the inverse of structure prediction and a core step in protein design, where a backbone is proposed first and a sequence that encodes it is designed afterwards.

Internally, ESM-IF1 is a sequence-to-sequence transformer with geometric input layers. A Geometric Vector Perceptron graph network encodes the backbone atom coordinates (N, C-alpha, C) into rotation-invariant per-residue features, and an autoregressive decoder then generates the sequence one residue at a time. Because experimentally determined structures are limited, the model was trained on roughly 12 million [UniRef50](https://www.uniprot.org/help/uniref) sequences whose structures were predicted with AlphaFold2, alongside experimental structures from [CATH](https://www.cathdb.info/). This raised native-sequence recovery to about 51% on structurally held-out backbones, and about 72% for buried residues. ESM-IF1 also handles complexes, partially masked structures, and binding interfaces. The reference implementation is maintained by [Meta AI](https://ai.meta.com/) in [facebookresearch/esm](https://github.com/facebookresearch/esm) and distributed in the `fair-esm` package.

ProteinDPO ([Widatalla et al., 2024](https://doi.org/10.1101/2024.05.20.595026)) is a variant of ESM-IF1 fine-tuned with Direct Preference Optimization (DPO) on a mega-scale experimental protein-stability dataset. It keeps the ESM-IF1 architecture but is trained to prefer stabilizing over destabilizing sequences for a given backbone, which improves both its designs and its stability scoring. ProteinDPO's implementation is available at [evo-design/protein-dpo](https://github.com/evo-design/protein-dpo).

### Learning Resources

- [ESM inverse folding examples](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) (Meta AI) - the official notebooks and scripts for running ESM-IF1 sequence design and scoring, including multi-chain complexes.

## Tools

### ESM-IF1 Sampling (`esm-if1-sample`)

Designs new sequences for a given backbone. Each input structure is encoded once and decoded into one or more candidate sequences, each returned with its average log-likelihood under the model.

#### Applications

Use this to redesign a natural protein or to generate sequences for a de novo backbone, including multi-chain complexes and binding interfaces where the surrounding chains are kept as context. With the default ProteinDPO weights the designs are biased toward higher experimental stability, which suits stabilization campaigns.

#### Usage Tips

- **`temperature` defaults to `1.0`, rather than the `0.1` used by the other inverse-folding tools.** ESM-IF1's reference inference samples at `1.0`, and this toolkit retains that default, so its designs are more diverse than those produced by the backbone-MPNN models. Lower it toward `0.1` for conservative, near-greedy designs, and raise it for greater variation.
- **`batch_size` controls how many sequences are produced per worker dispatch.** It defaults to `num_sequences_per_structure`, so the whole request is handled in one dispatch; the model still decodes the sequences one at a time. Lower it to bound peak GPU memory when a large request or long backbone exhausts memory.
- **Non-redesigned chains still shape the design.** Chains you do not select stay as fixed structural context rather than being ignored, so designing one chain of a complex accounts for its partners. `fixed_positions` is counted from 1, not 0 to follow biological conventions for residue selection.
- **Output is structured per design.** `output.design_sets[i].complexes[j]` is an `ESMIF1Design`; the designed target sequence is `design.designed_chains[0].sequence` and the log-likelihood is `design.metrics["log_likelihood"]`. `ESMIF1Design` is a `Complex` subclass and can be passed directly to structure predictors.

### ESM-IF1 Scoring (`esm-if1-score`)

Evaluates how well existing sequences fit a structure. Each sequence is scored against its paired structure using the full multi-chain context, returning the average log-likelihood and perplexity.

#### Applications

Use this to rank candidate sequences or assess point mutations by structural compatibility without generating new ones. With the default ProteinDPO weights the score also better reflects predicted stability, which is useful for prioritizing stabilizing variants.

#### Usage Tips
- **Lower perplexity is better, and it tracks the log-likelihood directly.** Perplexity is `exp(-avg_log_likelihood)`, so the two metrics rank candidates identically. Treat the score as compatibility under the model, not a guarantee the sequence folds, and confirm shortlisted candidates with a structure predictor.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESM-IF1 tool in this toolkit (`esm-if1-sample`, `esm-if1-score`).

- **Requires a GPU.** The geometric encoder and autoregressive decoder are not practical on CPU. Model weights download automatically on first use through the `fair-esm` package.
- **ProteinDPO is the default for both tools.** `esm-if1-sample` and `esm-if1-score` both default `weights_variant` to `protein_dpo`, the stability-aligned variant, so by default designs are biased toward stability and scores reflect predicted stability rather than the original ESM-IF1 likelihood. Set `weights_variant` to `esmif` for the original ESM-IF1 model in either tool.
