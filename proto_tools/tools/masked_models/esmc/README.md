<a href="https://bio-pro.mintlify.app/tools/masked-models/esmc"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# ESM C (Cambrian)

![ESM C (Cambrian)](https://cdn.prod.website-files.com/6606dc3fd5f6645318003df4/663e392ca11e77f1a562c2c6_og.png)

> *Image source: [EvolutionaryScale](https://www.evolutionaryscale.ai)*

> [!NOTE]
> **License:** ESM C (Cambrian) is licensed under Custom (Cambrian Open License Agreement) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement) for full terms.

## Overview

ESM C ("Cambrian") is EvolutionaryScale's embedding-focused protein language model. This toolkit wraps the openly licensed `esmc_300m` and `esmc_600m` models to produce per-sequence embeddings and optional per-position scores (logits) from supplied protein sequences. It provides only an embedding interface; it does not support sequence sampling or scoring.

## Background

ESM C ([EvolutionaryScale, 2024](https://www.evolutionaryscale.ai/blog/esm-cambrian)) is a protein language model trained with the masked language modeling objective: during training, residues are hidden at random and the model learns to predict the original amino acid from the surrounding residues on both sides. For each residue it produces a contextual numerical representation (an embedding), along with per-position scores (logits) over the 20 standard amino acids.

ESM C is distributed in the same `esm` software package as ESM3, but does not include ESM3's structure track or sequence-generation capability; it provides only embeddings and per-position scores. Two openly licensed model sizes are wrapped here: `esmc_300m` (embedding size 960, Cambrian Open License, commercial use permitted) and `esmc_600m` (embedding size 1152, Cambrian Non-Commercial License, research and internal use only). A larger 6B-parameter ESM C model is available only through EvolutionaryScale's hosted Forge service and is not exposed by this wrapper.

## Tools

### ESM C Embeddings (`esmc-embedding`)

Runs each input sequence through ESM C once and averages the per-residue representations, excluding the start and end tokens and any padding, into a single fixed-length vector per sequence. Per-position scores (logits) over the 20 standard amino acids are also returned when requested.

#### Applications

The averaged embedding is a learned numerical representation of a protein, suitable for machine-learning tasks such as clustering, classification, and property prediction, and for similarity search by comparing these vectors (for example with cosine similarity). The optional per-position scores give the model's predicted amino-acid preference at each site, useful for conservation analysis and for examining the model's expectations at specific positions. ESM C is embedding-focused, so it is the lighter-weight choice when you need embeddings or per-position scores but not sequence generation or scoring.

#### Usage Tips

- **`model_checkpoint` selects the model size and its license.** `esmc_300m` (the default, embedding size 960) is under the Cambrian Open License with commercial use permitted; `esmc_600m` (embedding size 1152) is Cambrian Non-Commercial only. Use `esmc_300m` for any commercial use.
- **`repr_layer` selects which internal model layer the embedding is taken from.** The default `-1` uses the final layer; other values select earlier layers.
- **Per-position scores are large.** Enabling `return_logits` adds an array of size (sequence length by 20) per sequence, which dominates runtime and memory for long inputs. Leave it set to `False` unless you need the per-position scores.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESM C tool in this toolkit (`esmc-embedding`).

- **ESM C shares the EvolutionaryScale `esm` environment with ESM3.** Both are distributed in the same `esm` package and use a single shared on-disk environment (`evolutionaryscale_esm`); installing either tool installs the environment for both.
- **The license depends on the model size.** `esmc_300m` (the default) is under the Cambrian Open License, with commercial use permitted subject to the naming and attribution requirement; `esmc_600m` is under the Cambrian Non-Commercial License and must not be used commercially. The 6B model is available only through EvolutionaryScale's hosted Forge service and is not wrapped here.
- **`batch_size` controls memory usage.** Lower it if you run out of GPU memory; raise it to process short sequences faster. For repeated single-batch calls, use `ToolInstance.persist_tool("esmc")` to keep the model loaded in memory between calls; for multi-GPU or large-batch runs, prefer `ToolPool`.
