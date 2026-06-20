<a href="https://bio-pro.mintlify.app/tools/masked-models/esm3"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# ESM3

![ESM3](https://proto-bio.github.io/proto-assets/images/tool/esm3/hero.png)

> [!NOTE]
> **License:** ESM3 uses Custom (Cambrian Open License Agreement) for code and Custom (Cambrian Non-Commercial License Agreement) for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Model weights are gated and require accepting the provider's terms and authenticating with a HuggingFace token. Please refer to the [code license](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement) and [model weights license](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement) for full terms.

## Overview

ESM3 is EvolutionaryScale's generative protein language model, trained jointly over sequence, structure, and function. This toolkit wraps the open `esm3_sm_open_v1` checkpoint to embed, sample masked positions in, and score supplied protein sequences.

## Background

In 2025, [Hayes et al.](https://doi.org/10.1126/science.ads0018) introduced ESM3, a generative model from [EvolutionaryScale](https://www.evolutionaryscale.ai) that departs from the encoder-only design of the ESM-1/ESM-2 line. ESM3 is a masked generative transformer that represents a protein across three simultaneous tracks (amino-acid sequence, discrete structure tokens, and function annotation). Training masks spans across all three tracks, so a single model can be prompted with any combination of partial sequence, structure, and function and asked to complete the rest. The flagship 98B-parameter model (`esm3-large-2024-03`) is available through the [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai) API under closed-beta access (also offered via AWS SageMaker); the publicly released open checkpoint, `esm3_sm_open_v1`, is the small 1.4B-parameter variant.

ESM3 is the multimodal successor to ESM-2 ([Lin et al., 2023](https://doi.org/10.1126/science.ade2574)). Where ESM-2 is a sequence-only masked language model, ESM3 adds structure and function tracks and a generative objective. For pure sequence-embedding workloads ESM-2 remains lighter and faster; ESM3 is the choice when masked generative editing matters. This toolkit exposes only the sequence-track operations (embeddings, masked sampling, scoring) over supplied sequences.

## Tools

### ESM3 Embeddings (`esm3-embedding`)

Runs a single forward pass over ESM3 and mean-pools the per-residue hidden states into a fixed-length sequence descriptor. Per-position amino-acid logits are returned on request.

#### Applications

The mean-pooled embedding is a learned protein representation for downstream supervised tasks such as clustering, classification, and property regression, and powers similarity search through cosine similarity on the mean vector.

#### Usage Tips

- **`repr_layer` selects which transformer layer is mean-pooled.** The default `-1` returns the post-norm output of the last block (matching ESM-2/ESMC `-1` semantics); other indices select pre-norm per-block hidden states, captured via a forward hook because `ESM3.forward` discards them.
- **Per-position logits are large.** Enabling `return_logits` adds a per-position vocabulary-sized float tensor per sequence, dominating wall time and memory on long inputs. Leave it `False` unless the per-position distribution is needed.

### ESM3 Sampling (`esm3-sample`)

Selects positions via a configurable masking strategy, masks them, and resamples from ESM3's predicted distribution. `single_pass` fills every masked position in one forward pass; `iterative_refinement` dispatches to ESM3's native `batch_generate` for multi-round commitment. Positions can also be pre-masked directly with `_` in the input string, or a masking strategy can be used.

#### Applications

This tool drives guided point mutation, variant generation, and infilling at designable sites. Resampling masked positions from a protein language model is the core operation behind directed-evolution proposals and antibody affinity maturation. Which positions are resampled is set by the [masking strategy](https://github.com/evo-design/proto-tools/blob/main/proto_tools/transforms/masking/README.md); see its README for the available selection methods and tuning parameters.

#### Usage Tips

- **`iterative_refinement` produces more coherent joint samples than `single_pass`.** It runs ESM3's `batch_generate` over `num_steps` rounds (cosine or linear unmask schedule) instead of filling every mask independently in one pass; it is roughly `num_steps×` slower. Default to it when masking more than a handful of sites.
- **`masking_strategy` controls which positions get masked before sampling.** See the [masking strategy README](https://github.com/evo-design/proto-tools/blob/main/proto_tools/transforms/masking/README.md) for the available selection methods and tuning parameters. As an alternative to passing a strategy, pre-mask exact positions with `_` directly in the input string and the masking strategy is skipped entirely.
- **`temperature` scales the per-position logits before sampling.** Values of 0.5 to 0.7 yield conservative mutations close to the input; values above 1.0 broaden exploration of the model's distribution.

### ESM3 Scoring (`esm3-score`)

Computes masked-language-model pseudo-perplexity for each input sequence. Each position is masked individually and the model's log-probability of the true amino acid under bidirectional context is recorded, then aggregated into per-sequence log-likelihood, average log-likelihood, and perplexity.

#### Applications

ESM3 pseudo-perplexity is a fitness proxy for ranking variants, filtering generated sequences for naturalness, or comparing engineered constructs against wild type. The masked log-likelihood difference between wild-type and mutant residues is a zero-shot baseline for variant-effect prediction.

#### Usage Tips

- **Pseudo-perplexity is a relative score, not an absolute fitness.** It is measured against the model's training distribution and is sensitive to length, so it is most useful for comparing closely related sequences of similar length.
- **Ambiguous residues are excluded.** Perplexity is computed only over the 20 canonical amino acids; `X`, `B`, `Z`, and similar are dropped from both the log-likelihood sum and the position count.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESM3 tool in this toolkit (`esm3-embedding`, `esm3-sample`, `esm3-score`).

- **ESM3 is a gated model and requires a HuggingFace token.** The open checkpoint lives behind a gated HuggingFace repo ([EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)). Set the `HF_TOKEN` environment variable with an account that has accepted the model license, or every tool raises before loading.
- **One open checkpoint is available.** `esm3_sm_open_v1` is the only public open-weights checkpoint; larger ESM3 models are EvolutionaryScale API-only and not wrapped here.
- **ESM3 is larger than many ESM-2 variants.** For sequence-embedding-only workloads, smaller ESM-2 variants are faster; consider reaching for ESM3 when you want masked generative editing. This toolkit takes only amino-acid sequences as input and does not expose the structure or function tracks.
- **`batch_size` controls memory usage across the toolkit.** Lower it if you OOM; raise it for short-sequence throughput. For `esm3-score`, `batch_size` counts masked variants pooled across all input sequences rather than sequences themselves (each input contributes one masked variant per position).
