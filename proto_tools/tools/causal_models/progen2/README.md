<a href="https://bio-pro.mintlify.app/tools/causal-models/progen2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ProGen2

> [!NOTE]
> **License:** ProGen2 has a BSD-3-Clause license. Please refer to [the license](https://github.com/enijkamp/progen2/blob/main/LICENSE.txt) for full terms.

## Overview

ProGen2 is an autoregressive protein language model from Salesforce Research, first released in 2022 and published in 2023, trained on natural protein sequences from genomic, metagenomic, and immune-repertoire databases.

## Background

ProGen2 ([Nijkamp et al., 2023](https://doi.org/10.1016/j.cels.2023.10.002)) is a family of autoregressive protein language models trained with a next-token prediction objective: during training the model learns to predict the next residue given all preceding residues. The family spans `progen2-small` (151 million parameters) up to `progen2-xlarge` (6.4 billion parameters). The checkpoints were trained on different protein collections as a result of the paper's finding that the training-data distribution has a large and sometimes counterintuitive effect on downstream performance. Most checkpoints are trained on natural proteins drawn from [UniRef90](https://www.uniprot.org/help/uniref) and the [BFD](https://bfd.mmseqs.com/) metagenomic set; `progen2-BFD90` uses the BFD90 collection, and `progen2-oas` is trained on antibody sequences from the [Observed Antibody Space](https://opig.stats.ox.ac.uk/webapps/oas/) database.

The autoregressive training objective instills two primary capabilities. First, new candidate protein sequences can be sampled from a starting prompt via the predicted next-residue distributions. Second, the model can be used to score existing protein sequences, as the likelihood the model assigns to a sequence is shown in the paper to provide a proxy zero-shot fitness score or measure of plausibility with no additional task-specific training.

## Tools

### ProGen2 Sampling (`progen2-sample`)

Generates protein sequences by autoregressive sampling. Given one or more prompt sequences, the model extends each prompt one amino acid at a time, drawing each residue from the model's predicted distribution under the configured `temperature`, `top_p`, and `top_k` settings, until a stop token is produced or `max_length` (prompt plus generated, default 256) is reached.

#### Applications

This tool performs de novo protein design, generating novel sequences that resemble natural proteins conditioned on a prompt such as a starting motif or partial domain. The antibody-trained `progen2-oas` checkpoint targets antibody and immune-repertoire generation specifically.

#### Usage Tips

- **Generated output is trimmed by default.** Generated sequences are cut at the first stop token with the start/stop sentinels removed (`truncate_at_stop` and `strip_special_tokens`, both `True`); set them `False` to keep the raw model output.
- **Sampling defaults are conservative.** `temperature` defaults to `0.2` and `top_p` to `0.95`, which keep generations close to natural-looking sequences; raise `temperature` for more diverse but riskier designs. `top_k` defaults to `0`, which disables top-k truncation so only nucleus (`top_p`) sampling is applied.
- **`max_length` counts the prompt.** It caps prompt plus generated length (default `256`), so a long prompt directly reduces how much can be generated.
- **Output includes the prompt by default.** `prepend_prompt=True` (the toolkit default) returns the prompt joined to its continuation; set it `False` to receive only the newly generated residues.
- **Generated sequences are candidates.** Validate them with downstream tools (for example structure prediction, function annotation, or homology search) before drawing biological conclusions.

### ProGen2 Scoring (`progen2-score`)

Scores existing protein sequences using ProGen2. For each sequence it computes the model's predicted probability of every residue given the preceding residues and aggregates these into a log-likelihood, an average log-likelihood per residue, and a perplexity (perplexity is fully determined by the average log-likelihood, computed as `exp(-avg_log_likelihood)`, but is the conventionally reported metric). Optionally returns the per-position logits and the token vocabulary.

#### Applications

This tool gives a zero-shot measure of how consistent a protein sequence is with ProGen2's training distribution, which is used in the paper as a proxy-fitness predictor without additional task-specific training. It can be used to rank or filter candidate sequences (including the output of `progen2-sample`), to compare variants of a sequence, or to flag sequences far from the model's training distribution.

#### Usage Tips

- **Compare length-normalized scores within one checkpoint.** Total `log_likelihood` scales with sequence length, so use `perplexity` or `avg_log_likelihood` when comparing sequences of different lengths. Different checkpoints learn different distributions that are not calibrated to a common scale, so scores from different `model_checkpoint` values are hard to compare directly. A lower perplexity means the sequence is more consistent with that checkpoint's training distribution.
- **`return_logits` defaults to `False`.** Leave it off unless you need the per-position distributions, since the logits tensor is large (sequence length by the token vocabulary).
- **A domain-matched checkpoint is not automatically better for scoring.** The ProGen2 paper found the antibody-specific `progen2-oas` checkpoint underperformed the universal checkpoints on antibody fitness prediction, so a universal checkpoint (such as the default `progen2-large`) is often the safer choice for scoring.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ProGen2 tool in this toolkit (`progen2-sample`, `progen2-score`).

- **Requires a GPU; memory scales with checkpoint size.** The larger checkpoints, up to `progen2-xlarge` at 6.4 billion parameters, need substantially more GPU memory than `progen2-small`. CPU execution is not practical.
- **`batch_size` trades memory for throughput across both tools.** It sets how many prompts (`progen2-sample`) or sequences (`progen2-score`) are processed per GPU forward pass. Raise it for higher throughput on many short sequences; lower it (default `1`) if generation or scoring runs out of GPU memory.
- **`model_checkpoint` selects the training distribution.** The default `progen2-large` and the `small`, `medium`, `base`, and `xlarge` checkpoints are trained on broad natural-protein collections (UniRef90 and BFD); `progen2-BFD90` is trained on the BFD90 set and `progen2-oas` on antibody sequences from the Observed Antibody Space. The choice of model has performance implications for both sampling and scoring.
