<a href="https://bio-pro.mintlify.app/tools/masked-models/codonfm"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# CodonFM (Encodon)

![CodonFM (Encodon)](https://proto-bio.github.io/proto-assets/images/tool/codonfm/hero.png)

> [!NOTE]
> **License:** CodonFM (Encodon) uses Apache-2.0 for code and Custom (NVIDIA Open Model License) for model weights. Please refer to the [code license](https://github.com/NVIDIA-BioNeMo/CodonFM/blob/main/LICENSE) and [model weights license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) for full terms.

## Overview

CodonFM (model name Encodon) is a codon-level masked language model of protein-coding sequences from NVIDIA. Unlike nucleotide- or amino-acid-level models, Encodon tokenizes a coding sequence in-frame at codon resolution (one token per 3 nt), so it directly captures synonymous-codon usage and codon-context signals that are invisible to an amino-acid model. This toolkit exposes four Encodon checkpoints (80M, 600M, 1B, and a codon-frequency-aware-masking 1B variant) for coding-sequence fitness scoring, mutation-effect prediction, embeddings, and differentiable sequence design.

## Background

Encodon is a BERT-style Transformer encoder trained with a masked-language-modeling objective over codon tokens ([Darabi et al., 2025](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf)). Each coding sequence is split into in-frame codons, a fraction of codon tokens are masked, and the model predicts the original codon from its full bidirectional context. Because the vocabulary is codons rather than amino acids, the model learns the *language of codon translation* — which synonymous codon is expected in a given context — and not only the encoded protein. The released family spans four checkpoints (80M, 600M, 1B parameters, plus a codon-frequency-aware-masking 1B variant), all reading sequences up to 2048 tokens, i.e. coding sequences up to `(2048 - 2) × 3 = 6138 nt` after the CLS/SEP tokens.

## Tools

### CodonFM Fitness (`codonfm-fitness`)

Runs the upstream Encodon fitness routine: one unmasked forward pass, followed by the mean log-probability assigned to each visible non-padding input token. The mean includes the CLS and SEP special tokens as well as the codons. Higher is more model-typical, but this score is not masked pseudo-log-likelihood.

#### Applications

Sequence fitness is a zero-shot naturalness score for ranking or filtering coding sequences: comparing codon-optimized designs against wild type, screening synthetic constructs, or scoring candidate CDS variants without any task-specific training.

#### Usage Tips

- **Fitness is a relative score against Encodon's training distribution, not an absolute quantity.** It is most meaningful when comparing related sequences of similar length; the per-token mean already normalizes for length, but very short sequences are noisy.
- **`batch_size` trades memory for throughput.** Lower it if you OOM on long CDS, raise it for short sequences.

### CodonFM Score (`codonfm-score`)

Scores individual **codon substitutions** by the reference-vs-alternate log-likelihood ratio at the mutated position. The reference codon is masked and the model's log-probability of the reference and alternate codons is compared; a positive `llr` (`ref − alt`) means the model favors the reference and the substitution is model-disfavored.

#### Applications

The masked log-likelihood ratio is a canonical zero-shot variant-effect estimator. Because Encodon is codon-level, it discriminates *synonymous* substitutions (same amino acid, different codon) that an amino-acid model cannot see — useful for studying codon-usage effects on expression and mRNA stability.

#### Usage Tips

- **Each mutation carries its own reference sequence, codon position (1-based), and ref/alt codons.** The reference codon is validated against the sequence at that position, so an off-by-one frame error is caught before dispatch rather than silently mis-scored.
- **Scores are position-conditional.** The same substitution scored in different sequence contexts will differ; that context-dependence is the point.

### CodonFM Embeddings (`codonfm-embedding`)

Returns the final-layer **CLS-token embedding** for each coding sequence — a fixed-length learned representation whose dimensionality follows the checkpoint.

#### Applications

The CLS embedding is a codon-aware sequence descriptor for downstream supervised tasks (classification, regression, clustering) and similarity search over coding sequences.

#### Usage Tips

- **Different checkpoints produce different embedding sizes.** Representations from one checkpoint do not transfer to another without re-fitting downstream models; pick one and keep it fixed for an analysis.

### CodonFM Gradient (`codonfm-gradient`)

Computes the gradient of the mean masked negative log-likelihood with respect to a relaxed `(L, 64)` distribution over all 64 DNA codons, including the three standard stop codons (lexicographic order `AAA, AAC, AAG, AAT, …`). Each row's current argmax codon is the masked-language-model target. The Encodon weights are frozen; the relaxed distribution passes through Encodon's embedding layer and normalization, each codon position is masked in turn, and a per-chunk backward pass accumulates the gradient. An optional Straight-Through Estimator runs the forward on hard one-hot codons while routing gradients through the soft probabilities.

#### Applications

This exposes Encodon as a differentiable, codon-level naturalness prior for continuous sequence design — usable inside gradient descent, MCMC, or any optimization loop over relaxed coding sequences (e.g. codon optimization with a learned constraint).

#### Usage Tips

- **`temperature` converts the raw input into a per-position distribution.** The default `1.0` applies `softmax(logits / T)`; set it to `None` only when every row is already a non-negative probability distribution summing to 1.
- **`use_ste` enables the Straight-Through Estimator** for stronger guidance toward discrete codons while keeping a usable gradient.
- **`compute_gradient=False` runs forward-only.** The `gradient` field is `None` but `loss` and `metrics` are still populated. This is a masked pseudo-log-likelihood objective; it is distinct from the visible-token objective returned by `codonfm-fitness`.

### CodonFM Sampling (`codonfm-sample`)

Resamples a subset of codons in a coding sequence. A number of codon positions (`num_mutations`, or `mask_fraction` of the codons) are chosen at random, masked, and refilled from Encodon's distribution over the 61 sense codons in a single forward pass. The sequence length is preserved and sampling cannot introduce a new stop codon. It can replace an existing stop if that position is selected, so keep a required terminal stop outside the editable region or restore it afterward.

#### Applications

Masked-codon resampling is the local-edit primitive behind coding-sequence design: it proposes model-plausible synonymous or missense codon changes for directed-evolution / MCMC / genetic-algorithm loops (e.g. as the mutation generator in a Proto Language optimizer, paired with the `codonfm-fitness` constraint).

#### Usage Tips

- **`num_mutations` overrides `mask_fraction`.** It sets an exact number of positions to resample, not a guaranteed Hamming distance: the model can draw the original codon again.
- **`temperature` controls diversity.** Below 1.0 sharpens toward the model's favorite codon; above 1.0 broadens exploration. Because sampling is stochastic, pass a `seed` for reproducible proposals.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every CodonFM tool in this toolkit (`codonfm-fitness`, `codonfm-score`, `codonfm-embedding`, `codonfm-gradient`, `codonfm-sample`).

- **Checkpoints download on demand.** The `nvidia/NV-CodonFM-Encodon-*-v1` repos are public; the standalone worker fetches the `.safetensors` weights and their `config.json` on first use and caches them. No HuggingFace token is required (one is used automatically if present).
- **A CUDA GPU is required.** The pinned upstream Encodon attention implementation uses xFormers kernels that do not provide a CPU execution path.
- **Inputs are codon-aligned nucleotide sequences.** Each length must be a multiple of 3; RNA `U` is mapped to `T`, while ambiguous bases such as `N` are rejected because upstream tokenization would shift codon positions. Inputs are not checked for a start codon, terminal stop, internal stops, or coding-strand orientation.
- **Max sequence length is 6138 nt (2046 codons).** Encodon's positional cap is 2048 tokens; longer inputs raise `ValueError` rather than truncating.
- **The default checkpoint is `encodon_80m`.** It is the fastest; the 600M/1B checkpoints trade throughput for fidelity. Pick a larger one for final scoring.
