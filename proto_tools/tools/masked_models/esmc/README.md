<a href="https://bio-pro.mintlify.app/tools/masked-models/esmc"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESM C (Cambrian)

![ESM C (Cambrian)](https://proto-bio.github.io/proto-assets/images/tool/esmc/hero.png)

> [!NOTE]
> **License:** ESM C (Cambrian) is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/Biohub/esm/blob/main/LICENSE.md) for full terms.

## Overview

ESM C ("Cambrian") is [Biohub](https://biohub.ai)'s embedding-focused protein language model. This toolkit wraps the `esmc_300m`, `esmc_600m`, and `esmc_6b` models to produce per-sequence embeddings and optional per-position scores (logits) from supplied protein sequences. It provides only an embedding interface; it does not support sequence sampling or scoring.

## Background

ESM C ([Biohub](https://biohub.ai/papers/esm_protein.pdf)) is a protein language model trained with the masked language modeling objective: during training, residues are hidden at random and the model learns to predict the original amino acid from the surrounding residues on both sides. For each residue it produces a contextual numerical representation (an embedding), along with per-position scores (logits) over the 20 standard amino acids.

ESM C is distributed in the same `esm` software package as ESM3, but does not include ESM3's structure track or sequence-generation capability; it provides only embeddings and per-position scores. Three model sizes are wrapped here, all MIT-licensed: `esmc_300m` (embedding size 960, 30 layers), `esmc_600m` (embedding size 1152, 36 layers), and `esmc_6b` (embedding size 2560, 80 layers). The 6B model is the largest ESM C variant and underpins both the ESM Atlas and the ESMFold2 structure predictor, which is trained on top of a frozen ESM C 6B.

## Tools

### ESM C Embeddings (`esmc-embedding`)

Runs each input sequence through ESM C once and averages the per-residue representations, excluding the start and end tokens and any padding, into a single fixed-length vector per sequence. Per-position scores (logits) over the 20 standard amino acids are also returned when requested.

#### Applications

The averaged embedding is a learned numerical representation of a protein, suitable for machine-learning tasks such as clustering, classification, and property prediction, and for similarity search by comparing these vectors (for example with cosine similarity). The optional per-position scores give the model's predicted amino-acid preference at each site, useful for conservation analysis and for examining the model's expectations at specific positions. ESM C is embedding-focused, so it is the lighter-weight choice when you need embeddings or per-position scores but not sequence generation or scoring.

#### Usage Tips

- **`model_checkpoint` selects the model size.** `esmc_300m` (the default) has embedding size 960, `esmc_600m` has 1152, and `esmc_6b` has 2560. Larger checkpoints give richer representations but cost more GPU memory and time — `esmc_6b` loads about 13 GB of bf16 weights, before activations that grow with sequence length and `batch_size`.
- **`repr_layer` selects which internal model layer the embedding is taken from.** The default `-1` uses the final layer; other values select earlier layers.
- **Per-position scores are large.** Enabling `return_logits` adds an array of size (sequence length by 20) per sequence, which dominates runtime and memory for long inputs. Leave it set to `False` unless you need the per-position scores.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESM C tool in this toolkit (`esmc-embedding`).

- **ESM C shares the Biohub `esm` environment with ESM3.** Both are distributed in the same `esm` package and use a single shared on-disk environment (`biohub_esm`); installing either tool installs the environment for both.
- **All checkpoints are MIT-licensed and ungated.** `esmc_300m`, `esmc_600m`, and `esmc_6b` are all free for academic and commercial use, and none require a HuggingFace token. Weights download automatically on first use; `esmc_6b` downloads roughly 25 GB.
- **`batch_size` controls memory usage.** Lower it if you run out of GPU memory; raise it to process short sequences faster. For repeated single-batch calls, use `ToolInstance.persist_tool("esmc")` to keep the model loaded in memory between calls; for multi-GPU or large-batch runs, prefer `ToolPool`.
