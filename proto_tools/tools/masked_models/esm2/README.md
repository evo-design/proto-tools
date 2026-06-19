<a href="https://bio-pro.mintlify.app/tools/masked-models/esm2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESM2

![ESM2](https://proto-bio.github.io/proto-assets/images/tool/esm2/hero.png)

> [!NOTE]
> **License:** ESM2 is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/facebookresearch/esm/blob/main/LICENSE) for full terms.

## Overview

Published in 2023, ESM-2 is Meta AI's second-generation of protein masked language models. Spanning six checkpoints ranging in scale from 8M to 15B parameters, the ESM-2 model family has become a widely used tool for protein embedding generation, and zero-shot variant-effect prediction via masked log-probabilities.

## Background

In 2023, [Lin et al.](https://doi.org/10.1126/science.ade2574) introduced ESM-2, a family of Transformer encoders trained with a BERT-style masked-language-modeling objective. Training used [UniRef50](https://www.uniprot.org/help/uniref), a clustered subset of UniProt covering roughly 65 million unique protein sequences. A central focus of the paper was the impact of scale, which was treated as the experimental variable across six model checkpoints spanning more than three orders of magnitude (8M, 35M, 150M, 650M, 3B, and 15B parameters). ESM-2 models were trained using a simple masked language modeling (MLM) objective adapted from BERT. Unlike autoregressive language models, which predict each token from preceding context only, MLM lets every residue attend to its full sequence context in both directions. At each training step a randomly generated mask covers 15% of input residues and replaces those tokens with a `<mask>` symbol. The model is then trained to predict the original amino acid from the surrounding bidirectional context. No structural, functional, or alignment supervision is used.

ESM-2 has since become a de facto sequence representation model for protein engineering. Its direct successor, ESM3 ([Hayes et al., 2025](https://doi.org/10.1126/science.ads0018)), extends the recipe at [EvolutionaryScale](https://www.evolutionaryscale.ai) into a multimodal generative model that jointly handles sequence, structure, and function tracks via discrete diffusion. ESM-2 still remains the lightest and most widely deployed protein language model. Within this toolkit, the 650M checkpoint (`esm2_t33_650M_UR50D`) is a standard quality/speed tradeoff and is the default for every tool.

## Tools

### ESM2 Embeddings (`esm2-embedding`)

Runs a single forward pass over ESM-2 to extract contextualized per-residue hidden states. The hidden states are mean-pooled across valid positions to produce a fixed-length sequence descriptor. Per-position 20-way amino-acid logits over the canonical order `ACDEFGHIKLMNPQRSTVWY` are also returned on request.

#### Applications

The mean-pooled embedding is a standard learned protein representation for downstream supervised tasks like clustering, classification, and regression on protein properties. The same embeddings also power similarity search through cosine similarity on the mean vector. The per-position logits support variant-effect screening by comparing wild-type and mutant log-probabilities at each position. The underlying attention maps are themselves rich enough to recover residue-residue contacts without explicit supervision.

#### Usage Tips

- **The last transformer layer carries the richest bidirectional context.** `repr_layer` chooses which layer to read for the mean-pooled embedding; the default `-1` selects the last layer and is the standard pick for downstream classification, regression, and variant-effect work. Earlier layers can outperform the top on certain probes (contact prediction is the canonical example).
- **Per-position logits are large and slow to materialize.** Enabling `return_logits` adds a `seq_len × 20` float tensor per sequence to the output, dominating wall time on long inputs. Leave it `False` unless you actually need the per-position distribution.

### ESM2 Sampling (`esm2-sample`)

Selects positions to mutate via a specifiable masking strategy, replaces them with `<mask>`, and resamples from ESM-2's predicted distribution. Two decoding modes are available. The `single_pass` mode fills every masked position in one forward pass with independent draws. The `iterative_refinement` mode instead runs a [MaskGIT](https://arxiv.org/abs/2202.04200)-style multi-round commit loop. Each round of that loop uses a cosine or linear unmask schedule with optional temperature annealing. To target specific positions directly, pre-mask them yourself with `_` in the input string. The tool will then fill exactly those positions and skip the masking strategy entirely.

#### Applications

This tool drives guided point mutation, variant generation, and infilling at designable sites for protein engineering work. Resampling masked positions from a protein language model is the core operation behind directed-evolution proposals and antibody affinity maturation, which was demonstrated at experimental scale in [Hie et al., 2024](https://www.nature.com/articles/s41587-023-01763-2). It is also the inner loop of MaskGIT-style iterative refinement schemes adapted from image generation ([Chang et al., 2022](https://arxiv.org/abs/2202.04200)) for biological sequences.

#### Usage Tips

- **`iterative_refinement` produces more coherent joint samples than `single_pass`.** It is a multi-round MaskGIT-style commit loop (each round uses a cosine or linear unmask schedule) and is roughly `num_steps×` slower than the one-shot `single_pass` mode. Default to it whenever you mask more than a handful of sites.
- **`masking_strategy` controls which positions get masked before sampling.** See the [masking strategy README](https://github.com/evo-design/proto-tools/blob/main/proto_tools/transforms/masking/README.md) for the available selection methods and tuning parameters. As an alternative to passing a strategy, pre-mask exact positions yourself with `_` directly in the input string and the masking strategy is skipped entirely.
- **`temperature` scales the per-position logits before sampling.** Values of 0.5 to 0.7 yield conservative mutations close to the input; values above 1.0 broaden exploration of the model's distribution.
- **Long-range coherence is weak.** ESM-2 has no global coherence beyond its local context window, so very long-range dependencies between distant residues are not well captured even in iterative mode.
- **ESM-2 was trained as a masked language model, not with a generative objective.** Resampling masked positions works for local edits, but the model was optimized for representation rather than de novo generation. For generative workloads (large-scale infilling, sequence design), [ESM3](https://bio-pro.mintlify.app/tools/masked-models/esm3) adds an explicit generative training objective and is the better fit.

### ESM2 Scoring (`esm2-score`)

Computes the masked-language-model pseudo-perplexity for each input sequence. Each position is masked individually, and the model's log-probability of the true amino acid under bidirectional context is recorded. The per-position scores are then aggregated into per-sequence log-likelihood, average log-likelihood, and perplexity metrics.

#### Applications

ESM2 pseudo-perplexity is a standard fitness proxy when ranking variants, filtering generated sequences for naturalness, or comparing engineered constructs against wild type. The same masked log-likelihood difference between wild-type and mutant residues is a canonical zero-shot baseline for variant-effect prediction.

#### Usage Tips

- **Pseudo-perplexity is a relative score, not an absolute fitness.** It is measured against ESM-2's training distribution, which is UniRef50 (the natural proteins it saw during pretraining), which can bias it to proteins that are more heavily represented. The metric is also sensitive to length, so it is most useful for comparing closely related sequences of similar length.
- **Ambiguous residues are excluded.** Perplexity is computed only over the 20 canonical amino acids; `X`, `B`, `Z`, and similar are dropped from both the log-likelihood sum and the position count.

### ESM2 Gradient (`esm2-gradient`)

Computes the gradient of the mean masked negative log-likelihood with respect to a relaxed `(L, 20)` input distribution over the canonical amino-acid order `ACDEFGHIKLMNPQRSTVWY`. The ESM-2 weights are kept frozen throughout. The relaxed distribution is mixed against ESM-2's per-residue token embeddings to form a soft input. Each amino-acid position is then masked in turn, and a per-chunk backward pass accumulates the gradient. An optional Straight-Through Estimator runs the forward on hard one-hot tokens while still routing gradients through the soft probabilities.

#### Applications

This tool exposes ESM-2 as a differentiable, structure-free protein-language-model loss for use inside MCMC, gradient descent, or any other optimization loop over relaxed protein sequences. It is most often used as a naturalness prior in continuous design pipelines, including latent Bayesian optimization frameworks and discrete walk-jump sampling approaches for de novo protein design.

#### Usage Tips

- **`temperature` controls how the raw input is converted into a distribution.** With a value set, the tool applies `softmax(logits / T)` before the forward pass; leave it `None` (the default) if the input already sums to 1 per position.
- **`use_ste` enables the Straight-Through Estimator.** The forward then runs on hard one-hot tokens while gradients still route through the soft probabilities, giving stronger guidance toward discrete sequences. Leave it off for smooth optimization over the relaxed simplex.
- **`compute_gradient` toggles whether the backward pass runs.** When set to `False`, the `gradient` field is `None`, but `loss` and `metrics` (log-likelihood, perplexity, and so on) are still populated. Useful for ranking MCMC proposals without paying the backward cost.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESM-2 tool in this toolkit (`esm2-embedding`, `esm2-sample`, `esm2-score`, `esm2-gradient`).

- **Different ESM-2 checkpoints produce different embedding sizes.** Downstream tasks built on one checkpoint will not transfer to another without re-fitting; pick one and stick with it for an analysis.
- **Smaller checkpoints run faster.** The 150M and 35M variants are significantly faster than the 650M default, with drops in representation quality.
- **Max sequence length is 1022 residues.** ESM-2's positional encoding caps inputs at 1022 residues, and will raise `ValueError` on longer inputs rather than truncating.
- **`batch_size` controls memory usage across the toolkit.** Lower it if you OOM; raise it for short-sequence throughput. One nuance: for `esm2-score`, `batch_size` counts masked variants pooled across all input sequences rather than sequences themselves (each input contributes `L` masked variants).