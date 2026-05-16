<a href="https://bio-pro.mintlify.app/tools/causal-models/progen3"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# ProGen3

![ProGen3](https://cdn.prod.website-files.com/6769c2fceb550649b2f37b59/6769c2fceb550649b2f37d5e_ProGen.avif)

> *Image source: [Profluent](https://www.profluent.bio/showcase/progen3)*

> [!NOTE]
> **License:** ProGen3 uses Apache-2.0 for code and CC-BY-NC-SA-4.0 for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/Profluent-AI/progen3/blob/main/LICENSE-CODE) and [model weights license](https://github.com/Profluent-AI/progen3#license) for full terms.

## Overview

First released in 2025, ProGen3 is a family of autoregressive protein language models from Profluent that use a sparse mixture-of-experts architecture. It is trained on a large curated corpus of natural protein sequences.

## Background

ProGen3 ([Roney et al., 2025](https://doi.org/10.1101/2025.05.16.654471)) is a family of generative protein language models from Profluent. ProGen3 models employ a sparse mixture-of-experts (MoE) architecture, which routes model activations in the transformer feed-forward layers to smaller specialized MLPs to make each forward pass more computationally tractable. The published family spans 112 million to 46 billion parameters; this toolkit exposes the `progen3-112m` through `progen3-3b` checkpoints. Pre-training used roughly 1.5 trillion amino-acid tokens sampled from the Profluent Protein Atlas, a curated collection of full-length natural proteins.

Unlike a strictly left-to-right model, ProGen3 is trained autoregressively in **both directions**: forward predicts each residue from the N-terminus toward the C-terminus, and reverse predicts from the C-terminus toward the N-terminus. Generation runs in a chosen direction, and scoring combines both directions into a single per-residue likelihood. Two capabilities follow from this objective. Sampling from the predicted next-residue distributions produces new candidate protein sequences, and the likelihood the model assigns to an existing sequence provides a zero-shot proxy-fitness score with no additional task-specific training.

### Learning Resources

- [ProGen3 showcase](https://www.profluent.bio/showcase/progen3) (Profluent) - an accessible overview of ProGen3, the Profluent Protein Atlas training data, and downstream applications such as antibody design and compact gene editors.

## Tools

### ProGen3 Sampling (`progen3-sample`)

Generates protein sequences by autoregressive sampling. Given one or more prompt sequences, the model extends each prompt one amino acid at a time, drawing each residue from the model's predicted distribution under the configured `temperature` and `top_p` settings, in the chosen `direction`, until `max_new_tokens` residues have been generated (at least `min_new_tokens`).

#### Applications

This tool performs de novo protein design, generating novel sequences that resemble natural proteins, optionally conditioned on a prompt. Because generation can run in reverse (C-terminus toward N-terminus), a C-terminal fragment can be used as the prompt and the rest of the sequence grown toward the N-terminus, which a strictly left-to-right model cannot do.

#### Usage Tips

- **`direction` chooses which terminus is generated.** `"forward"` (the default) continues a prompt from the N-terminus toward the C-terminus; `"reverse"` treats the prompt as a C-terminal fragment and generates toward the N-terminus. Note that the reverse generation will append to the prompt to grow the sequence on the left. All starting sequences should still be provided in the left to right direction from N->C.
- **Sampling defaults are conservative.** `temperature` defaults to `0.2` and `top_p` to `0.95`, which keep generations close to natural-looking sequences; raise `temperature` for more diverse but riskier designs. This tool exposes only nucleus (`top_p`) sampling for ProGen3; there is no top-k cutoff.
- **`max_new_tokens` and `min_new_tokens` bound the generated length.** They count only newly generated residues (default `256` and `1`), separate from the prompt length.
- **Output includes the prompt by default.** `prepend_prompt=True` (the toolkit default) returns the prompt joined to its continuation; set it `False` to receive only the newly generated residues.
- **Generated sequences are candidates.** Validate them with downstream tools (for example structure prediction, function annotation, or homology search) before drawing biological conclusions.

### ProGen3 Scoring (`progen3-score`)

Scores existing protein sequences under ProGen3 using bidirectional likelihood. For each sequence it runs both a forward (N→C) and a reverse (C→N) pass, averages the per-position log-likelihoods into a single bidirectional value, and aggregates these into a log-likelihood, an average log-likelihood per residue, and a perplexity. It also exposes the forward, reverse, and bidirectional per-position values, and optionally the per-position logits.

#### Applications

This tool gives a zero-shot measure of how consistent a protein sequence is with ProGen3's training distribution, usable as a fitness or plausibility signal without additional task-specific training. Because it uses both directions, every residue is scored with full surrounding context rather than left context only. Use it to rank or filter candidate sequences (including the output of `progen3-sample`), to compare variants of a sequence, or to flag sequences far from the model's training distribution.

#### Usage Tips

- **Scores are bidirectional, not a single-direction log-likelihood.** The reported `log_likelihood`, `avg_log_likelihood`, and `perplexity` are derived from the averaged forward and reverse per-position values, so they are not directly comparable to a one-directional model's scores.
- **Compare length-normalized scores within one checkpoint.** Total `log_likelihood` scales with sequence length, so use `perplexity` or `avg_log_likelihood` when comparing sequences of different lengths. Different checkpoints learn different distributions that are not calibrated to a common scale, so scores from different `model_checkpoint` values are hard to compare directly; a lower perplexity means the sequence is more consistent with that checkpoint's training distribution.
- **`return_logits` defaults to `False`.** Leave it off unless you need the per-position distributions, since the logits tensor is large (sequence length by the token vocabulary).

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ProGen3 tool in this toolkit (`progen3-sample`, `progen3-score`).

- **Requires a GPU; memory scales with checkpoint size.** This toolkit exposes the `progen3-112m` through `progen3-3b` checkpoints; larger checkpoints are more capable but need substantially more GPU memory. CPU execution is not practical.
- **`batch_size` trades memory for throughput across both tools.** It sets how many same-length prompts (`progen3-sample`) or sequences (`progen3-score`) are processed per GPU forward pass. Raise it for higher throughput on many short sequences; lower it (default `1`) if generation or scoring runs out of GPU memory.
- **`model_checkpoint` selects the model size.** The default is `progen3-762m`; smaller checkpoints (`progen3-112m`, `progen3-219m`, `progen3-339m`) are faster and lighter, while `progen3-1b` and `progen3-3b` are more capable at higher memory cost.
