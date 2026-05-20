<a href="https://bio-pro.mintlify.app/tools/causal-models/evo1"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Evo1

![Evo](https://github.com/evo-design/evo/raw/main/evo.jpg)

> *Image source: [evo-design/evo](https://github.com/evo-design/evo)*

> [!NOTE]
> **License:** Evo1 is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/evo-design/evo/blob/main/LICENSE) for full terms.

## Overview

Evo1 is an autoregressive DNA language model from Arc Institute and Stanford, trained at single-nucleotide resolution on prokaryotic and phage genomes. This toolkit wraps it as two tools that generate new DNA sequences from a prompt (`evo1-sample`) and score how likely existing DNA sequences are under the model (`evo1-score`).

## Background

Evo1 ([Nguyen et al., 2024](https://doi.org/10.1126/science.ado9336)) is a 7-billion-parameter DNA language model trained with an autoregressive objective: during training the model learns to predict the next nucleotide given all preceding nucleotides. Training used the [OpenGenome](https://huggingface.co/datasets/LongSafari/open-genome) dataset, roughly 2.7 million prokaryotic and phage genomes, so the model's predictions are most reliable for bacterial, archaeal, and phage sequences and are not expected to transfer well to eukaryotic genomes. It uses the StripedHyena architecture, a sequence model that combines convolutional state-space layers with a smaller number of attention layers. This design lets it process long stretches of DNA, up to 131,072 nucleotides for the long-context checkpoint, without the memory cost a pure attention model would incur at that length.

The autoregressive objective yields two capabilities directly. Sampling from the predicted next-nucleotide distributions produces new candidate sequences, and reading off the probabilities the model assigns to an existing sequence gives a likelihood score that reflects how closely the sequence matches the patterns seen during training. Alongside the base checkpoints, the authors released specialized variants trained on CRISPR loci and on transposable elements for those sequence types. Evo1 is the first model in the Evo family; [Evo2](https://bio-pro.mintlify.app/tools/causal-models/evo2) extends the approach to eukaryotic genomes and longer context.

### Learning Resources

- [Learning from DNA: a grand challenge in biology](https://hazyresearch.stanford.edu/blog/2024-03-14-evo) (Hazy Research, Stanford) - an accessible introduction to Evo from the authors, covering the motivation for genomic language modeling and how the model is trained and used.
- [Evo: DNA foundation modeling from molecular to genome scale](https://arcinstitute.org/news/evo) (Arc Institute) - an overview of Evo's capabilities, including genome-scale generation and the StripedHyena architecture.

## Tools

### Evo1 Sampling (`evo1-sample`)

Generates DNA sequences by autoregressive sampling. Given one or more prompt sequences, the model extends each prompt nucleotide by nucleotide, drawing each new nucleotide from the model's predicted distribution under the configured `temperature`, `top_k`, and `top_p` settings, until `max_new_tokens` new nucleotides have been produced. Optionally returns a per-sequence likelihood score (log-likelihood, average log-likelihood, and perplexity) for the generated sequences.

#### Applications

This tool produces candidate DNA sequences for downstream design and screening, including synthetic genes, regulatory regions, CRISPR systems (using the `evo-1-8k-crispr` checkpoint), and transposable elements (using the `evo-1-8k-transposon` checkpoint). The prompt sets the biological context for what follows, for example a start codon or promoter region.

#### Usage Tips

- **Match the checkpoint to the task.** `evo-1-8k-base` (the default) is the general prokaryotic and phage DNA model and `evo-1-131k-base` is its genome-scale, long-context counterpart. `evo-1-8k-crispr` and `evo-1-8k-transposon` are task-specific variants of `evo-1-8k-base` for generating CRISPR-Cas systems and IS200/IS605 transposons; use them when generating those systems and a base checkpoint otherwise.
- **`top_k` defaults to 4, the size of the DNA alphabet.** It exists mainly to keep generation on the four bases rather than other byte tokens, so it is not the diversity knob; control diversity with `temperature` (lower stays near the training distribution, higher explores it) and leave `top_p` at its default unless you specifically want nucleus sampling.
- **Output excludes the prompt by default.** `prepend_prompt=False` returns only the newly generated nucleotides, not the prompt joined to its continuation; set it `True` if you need the full sequence back.
- **Prompt length plus `max_new_tokens` must fit the checkpoint's context window** (8,192 nucleotides for the 8k checkpoints; `evo-1-131k-base` for longer). The model cannot attend beyond that window, so a long prompt directly reduces how much can be generated.
- **Generated sequences are candidates.** Validate them with downstream tools (for example ORF detection, structure prediction, or homology search) before drawing biological conclusions.

### Evo1 Scoring (`evo1-score`)

Scores existing DNA sequences under the Evo1 model. For each sequence, it computes the model's predicted probability of every nucleotide given the preceding nucleotides and aggregates these into a log-likelihood, an average log-likelihood per nucleotide, and a perplexity. Optionally returns the per-position logits and the token vocabulary.

#### Applications

This tool measures how well a DNA sequence matches the patterns the model learned from natural prokaryotic and phage genomes. Lower perplexity means the sequence is more consistent with that training distribution. Use it to rank or filter candidate sequences (including the output of `evo1-sample`), to compare variants of a sequence, or to flag sequences that fall far outside the model's training domain.

#### Usage Tips

- **Compare length-normalized scores within one checkpoint.** Total `log_likelihood` scales with sequence length, so use `perplexity` or `avg_log_likelihood` when comparing sequences of different lengths. Different checkpoints learn different distributions that are not calibrated to a common scale, so scores from different `model_name` values are hard to compare directly; a lower perplexity means the sequence is more consistent with that checkpoint's training distribution.
- **`return_logits` defaults to `False`.** Leave it off unless you need the per-position distributions, since the logits tensor is large (sequence length by a 512-token vocabulary).

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Evo1 tool in this toolkit (`evo1-sample`, `evo1-score`).

- **Requires a GPU.** An NVIDIA GPU with at least 24 GB of memory is recommended; CPU execution is possible but very slow and not practical for typical use.
- **`batch_size` trades memory for throughput across both tools.** It sets how many prompts (`evo1-sample`) or sequences (`evo1-score`) are processed per GPU forward pass. Raise it for higher throughput on many short sequences; lower it (default `1`) if generation or scoring runs out of GPU memory.
- **Trained on prokaryotic and phage genomes.** Predictions are most reliable within that domain. For eukaryotic genomes or longer context, use [Evo2](https://bio-pro.mintlify.app/tools/causal-models/evo2).
