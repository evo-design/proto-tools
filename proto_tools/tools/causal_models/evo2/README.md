<a href="https://bio-pro.mintlify.app/tools/causal-models/evo2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Evo2

![Evo2](https://github.com/ArcInstitute/evo2/raw/main/evo2.jpg)

> *Image source: [ArcInstitute/evo2](https://github.com/arcinstitute/evo2)*

> [!NOTE]
> **License:** Evo2 is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/arcinstitute/evo2/blob/main/LICENSE) for full terms.

## Overview

Evo2 is an autoregressive DNA language model from Arc Institute and Stanford, trained at single-nucleotide resolution across all domains of life. This toolkit wraps it to generate new DNA sequences from a prompt and to score how likely supplied DNA sequences are under the model.

## Background

Evo2 ([Brixi et al., 2026](https://doi.org/10.1038/s41586-026-10176-5)) is a DNA language model trained with an autoregressive objective: during training the model learns to predict the next nucleotide given all preceding nucleotides. Training used the [OpenGenome2](https://huggingface.co/datasets/arcinstitute/opengenome2) dataset, which spans bacterial, archaeal, eukaryotic, and phage genomes across all domains of life, so the model is not restricted to any single clade. It is available at several scales, the largest being 40 billion parameters, and uses the StripedHyena 2 architecture, a sequence model that combines convolutional state-space layers with a smaller number of attention layers. This design lets the model process very long stretches of DNA, up to roughly one million nucleotides for the long-context checkpoints, without the memory cost a pure attention model would incur at that length. Several checkpoints are also offered with shorter context windows for lower memory use, and one variant is trained specifically on Microviridae phage genomes.

The autoregressive objective yields two capabilities directly. Sampling from the predicted next-nucleotide distributions produces new candidate sequences, and reading off the probabilities the model assigns to an existing sequence gives a likelihood score that reflects how closely the sequence matches the patterns seen during training. Evo2 is the second model in the Evo family; the earlier [Evo1](https://bio-pro.mintlify.app/tools/causal-models/evo1) was trained only on prokaryotic and phage genomes, whereas Evo2 extends to eukaryotic genomes and longer context.

### Learning Resources

- [The Illustrated Evo 2](https://research.nvidia.com/labs/dbr/blog/illustrated-evo2/) (NVIDIA Research) - a visual walkthrough of the Evo 2 architecture and how the model processes and generates DNA.
- [Evo 2 Mechanistic Interpretability](https://arcinstitute.org/tools/evo/evo-mech-interp) (Arc Institute) - an interactive look at the internal features Evo 2 learns, built with sparse autoencoders to surface interpretable genomic patterns.

## Tools

### Evo2 Sampling (`evo2-sample`)

Generates DNA sequences by autoregressive sampling. Given one or more prompt sequences in Evo2's prompt format, the model extends each prompt nucleotide by nucleotide, drawing each new nucleotide from the model's predicted distribution under the configured `temperature`, `top_k`, and `top_p` settings, until `max_new_tokens` new nucleotides have been produced or an end-of-sequence token is sampled. A key-value cache makes long generations efficient and can be carried forward to continue a generation.

#### Applications

This tool produces candidate DNA sequences for downstream design and screening, including genes, regulatory regions, and longer multi-gene segments. Because Evo2 is trained across all domains of life, it can be prompted with eukaryotic as well as prokaryotic and phage context, unlike the prokaryote-and-phage-only Evo1. The prompt sets the biological context for what follows.

#### Usage Tips

- **Match the checkpoint to the task.** `evo2_7b` (the default), `evo2_20b`, and `evo2_40b` are the 1M-context models in increasing size and capability. The `evo2_7b_base`, `evo2_40b_base`, and `evo2_1b_base` checkpoints are 8K-context counterparts (`evo2_1b_base` is the smallest); `evo2_7b_262k` is a 262K-context variant; `evo2_7b_microviridae` is a 7B model adapted on Microviridae genomes for generating that bacteriophage family.
- **Prompts use Evo2's prompt format.** Prompt strings follow Evo2's special tokenization (for example a leading `+~` before DNA); see the upstream [Evo2 documentation](https://github.com/arcinstitute/evo2) for the conventions.
- **`top_k` defaults to 4, the size of the DNA alphabet.** It exists mainly to keep generation on the four bases rather than other byte tokens, so it is not the diversity knob; control diversity with `temperature` (lower stays near the training distribution, higher explores it) and leave `top_p` at its default unless you specifically want nucleus sampling.
- **Output includes the prompt by default.** `prepend_prompt=True` (the default for this toolkit) returns the prompt joined to its continuation; set it `False` to receive only the newly generated nucleotides.
- **Prompt length plus `max_new_tokens` (default 32) must fit the checkpoint's context window.** The model cannot attend beyond that window, so a long prompt directly reduces how much can be generated; pick a longer-context checkpoint when the combined length is large.
- **`stop_at_eos` ends generation early** when the model emits an end-of-sequence token; set it to `False` to always produce the full `max_new_tokens`.
- **Generated sequences are candidates.** Validate them with downstream tools (for example ORF detection, structure prediction, or homology search) before drawing biological conclusions.

### Evo2 Scoring (`evo2-score`)

Scores existing DNA sequences under the Evo2 model. For each sequence, it computes the model's predicted probability of every nucleotide given the preceding nucleotides and aggregates these into a log-likelihood, an average log-likelihood per nucleotide, and a perplexity. Optionally returns the per-position logits and the token vocabulary.

#### Applications

This tool measures how well a DNA sequence matches the patterns the model learned during training across all domains of life. Lower perplexity means the sequence is more consistent with that distribution. Use it to rank or filter candidate sequences (including the output of `evo2-sample`), to compare variants of a sequence, or to assess sequences from organisms outside the prokaryotic and phage range that Evo1 covers.

#### Usage Tips

- **Compare length-normalized scores within one checkpoint.** Total `log_likelihood` scales with sequence length, so use `perplexity` or `avg_log_likelihood` when comparing sequences of different lengths. Different checkpoints learn different distributions that are not calibrated to a common scale, so scores from different `model_checkpoint` values are hard to compare directly; a lower perplexity means the sequence is more consistent with that checkpoint's training distribution.
- **`return_logits` defaults to `False`.** Leave it off unless you need the per-position distributions, since the logits tensor is large (sequence length by a 512-token vocabulary).
- **`prepend_bos` adds a beginning-of-sequence token** before scoring; leave it `False` unless matching a specific upstream convention.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Evo2 tool in this toolkit (`evo2-sample`, `evo2-score`).

- **Requires a high-memory GPU; memory scales with model size and context length.** The 7B checkpoint needs a high-memory NVIDIA GPU; the 20B and 40B models and the 1M-context checkpoints need substantially more. CPU execution is not practical.
- **`batch_size` trades memory for throughput across both tools.** It sets how many prompts (`evo2-sample`) or sequences (`evo2-score`) are processed per GPU forward pass. Raise it for higher throughput on many short sequences; lower it (default `1`) if generation or scoring runs out of GPU memory.
- **Trained across all domains of life.** Evo2 covers prokaryotic, eukaryotic, archaeal, and phage genomes. For prokaryote-and-phage-only generation with a smaller model, [Evo1](https://bio-pro.mintlify.app/tools/causal-models/evo1) is an alternative.
