<a href="https://bio-pro.mintlify.app/tools/masked-models/esmc"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESM C (Cambrian)

![ESM C (Cambrian)](https://proto-bio.github.io/proto-assets/images/tool/esmc/hero.png)

> [!NOTE]
> **License:** ESM C (Cambrian) is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/Biohub/esm/blob/main/LICENSE.md) for full terms.

## Overview

ESM C ("Cambrian") is [Biohub](https://biohub.ai)'s embedding-focused protein language model. This toolkit wraps the `esmc_300m`, `esmc_600m`, and `esmc_6b` models to produce per-sequence embeddings and optional per-position scores (logits) from supplied protein sequences, and to decompose those activations with sparse autoencoders (SAEs) into a large, sparsely-active feature space that is easier to interpret than raw embeddings. It does not support sequence sampling or scoring.

## Background

ESM C ([Biohub](https://biohub.ai/papers/esm_protein.pdf)) is a protein language model trained with the masked language modeling objective: during training, residues are hidden at random and the model learns to predict the original amino acid from the surrounding residues on both sides. For each residue it produces a contextual numerical representation (an embedding), along with per-position scores (logits) over the 20 standard amino acids.

ESM C is distributed in the same `esm` software package as ESM3, but does not include ESM3's structure track or sequence-generation capability; it provides only embeddings and per-position scores. Three model sizes are wrapped here, all MIT-licensed: `esmc_300m` (embedding size 960, 30 layers), `esmc_600m` (embedding size 1152, 36 layers), and `esmc_6b` (embedding size 2560, 80 layers). The 6B model is the largest ESM C variant and underpins both the ESM Atlas and the ESMFold2 structure predictor, which is trained on top of a frozen ESM C 6B.

An SAE is trained to reconstruct a language model's internal activations through a bottleneck that permits only `k` active features per position out of a much larger codebook. The sparsity pressure pushes individual features toward single interpretable concepts, so a feature may correspond to a specific structural or functional property such as a zinc-binding site, a beta barrel, or a transmembrane helix. Biohub trained SAEs on ESM C using the TopK approach and used them to organize the ESM Atlas, a map of 6.8 billion proteins ([Biohub](https://www.biorxiv.org/content/10.64898/2026.06.03.729735)).

The SAEs were trained with two structural hyperparameters that set granularity. `k` fixes how many features are allowed to activate per residue, with lower values not able to reconstruct the activation as accurately — but higher values are harder to interpret, since a residue explained by 512 features is barely more legible than the dense embedding the SAE replaced. `k=64` is the balance Biohub trained across every layer. Additionally, the `codebook_size` of each SAE fixes how many features exist in total. Small codebooks group related concepts into one feature, for example a single metal-binding feature; large codebooks split that into dedicated zinc-finger, iron-sulfur, and calcium-binding features. Every combination of layer, `k`, and `codebook_size` is a separately trained SAE rather than a runtime setting, and not all combinations were published.

## Tools

### ESM C Embeddings (`esmc-embedding`)

Runs each input sequence through ESM C once and averages the per-residue representations, excluding the start and end tokens and any padding, into a single fixed-length vector per sequence. Per-position scores (logits) over the 20 standard amino acids are also returned when requested.

#### Applications

The averaged embedding is a learned numerical representation of a protein, suitable for machine-learning tasks such as clustering, classification, and property prediction, and for similarity search by comparing these vectors (for example with cosine similarity). The optional per-position scores give the model's predicted amino-acid preference at each site, useful for conservation analysis and for examining the model's expectations at specific positions. ESM C is embedding-focused, so it is the lighter-weight choice when you need embeddings or per-position scores but not sequence generation or scoring.

#### Usage Tips

- **`model_checkpoint` selects the model size.** `esmc_300m` (the default) has embedding size 960, `esmc_600m` has 1152, and `esmc_6b` has 2560. Larger checkpoints give richer representations but cost more GPU memory and time — `esmc_6b` loads about 13 GB of bf16 weights, before activations that grow with sequence length and `batch_size`.
- **`repr_layer` selects which internal model layer the embedding is taken from.** The default `-1` uses the final layer; other values select earlier layers.
- **Per-position scores are large.** Enabling `return_logits` adds an array of size (sequence length by 20) per sequence, which dominates runtime and memory for long inputs. Leave it set to `False` unless you need the per-position scores.

### ESM C SAE Features (`esmc-sae-features`)

Runs each sequence through the ESM C backbone once with SAEs attached to the requested layers, and returns the active codebook features at each residue, ordered by descending magnitude. Start and end tokens are stripped so positions align with the input sequence: `feature_indices[0]` holds the features for residue 1, and the `position` column of an exported CSV is 1-indexed, matching the rest of proto-tools.

#### Applications

Feature activations show which concepts the model recognizes at each residue, which supports interpreting what drives an embedding, locating functional sites without supervision, and comparing how proteins are represented internally. Because features are sparse and indexed, activations are directly comparable across proteins: within one SAE, a feature index always denotes the same learned concept. Indices are not comparable between different SAEs, including different layers of the same backbone, since each is trained separately and orders its codebook arbitrarily. The `ESMC-6B-sae-layer60-k64-codebook16384` SAE additionally has agent-generated natural-language descriptions for its codebook, available through the ESM Atlas.

#### Usage Tips

- **`layers` selects which activations are decomposed.** The default is the ~75%-depth layer Biohub sweeps (300M: 23, 600M: 27, 6B: 60), where representations transfer best to downstream tasks. Each extra layer adds a download and GPU memory.
- **`sae_target` picks what the SAE reads.** `hidden_states` (the default) decomposes the accumulated residual stream after a block, so features reflect everything the model has built up to that depth; it is what the ESM Atlas and the published feature descriptions use. `mlp_outputs` decomposes only that block's own MLP contribution before the residual add, which attributes a feature to one layer's computation. MLP-output SAEs are published only at `codebook_size=131072`.
- **`k` and `codebook_size` are only free at the sweep layer.** All-layer SAEs exist at `k=64` and one codebook size, so varying either requires `layers` to be exactly the sweep layer. The config rejects unpublished combinations and names the valid alternatives.
- **Only requested layers are downloaded, and layer size tracks `codebook_size`.** Each layer file holds an encoder and decoder of `d_model x codebook_size` weights, so a hidden-state layer is 0.13 GB on 300M and 0.34 GB on 6B, while an MLP-output layer (131072 codebook) is 1.0 GB and 2.7 GB respectively. Requesting every layer of the 6B MLP collection would pull roughly 217 GB; the tool logs a warning past 10 GB rather than refusing, since a deliberate multi-layer sweep is legitimate.
- **Rank features by normalized activation, not raw magnitude.** The largest raw activations belong to features that fire on nearly every protein and say little. Biohub's published statistics correct for this: `(activation / uniref90_max_activation) * uniref90_idf` scales a feature to [0, 1] and upweights rare ones. `describe_sae_features` in `helpers.py` returns both statistics alongside each feature's label, for the one SAE with published descriptions (`ESMC-6B-sae-layer60-k64-codebook16384`, which the 6B defaults resolve to).
- **Output size scales with `k` times sequence length.** Each residue carries `k` indices and `k` magnitudes, so a 300-residue protein at `k=64` yields 19,200 pairs per layer.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESM C tool in this toolkit (`esmc-embedding`, `esmc-sae-features`).

- **ESM C shares the Biohub `esm` environment with ESM3.** Both are distributed in the same `esm` package and use a single shared on-disk environment (`biohub_esm`); installing either tool installs the environment for both.
- **All checkpoints are MIT-licensed and ungated.** `esmc_300m`, `esmc_600m`, and `esmc_6b` are all free for academic and commercial use, and none require a HuggingFace token. Weights download automatically on first use; `esmc_6b` downloads roughly 25 GB.
- **The two tools load the backbone differently.** `esmc-embedding` reads the `esm`-package weights; `esmc-sae-features` loads the Transformers-format backbone (`biohub/ESMC-300M` and siblings), because the SAE API is defined on the Transformers model. Both repos hold the same parameters in different serializations, so using both tools downloads the backbone twice: 1.3 GB for 300M, 2.3 GB for 600M, 25.4 GB for 6B. This is deliberate — the SAEs are published and documented against the Transformers model, and reading the `esm`-package activations instead agrees on only about 99% of active features, which is the wrong trade for an interpretability tool. Each backbone is loaded only when its tool is first called.
- **`batch_size` controls memory usage.** Lower it if you run out of GPU memory; raise it to process short sequences faster. For repeated single-batch calls, use `ToolInstance.persist_tool("esmc")` to keep the model loaded in memory between calls; for multi-GPU or large-batch runs, prefer `ToolPool`.
