<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/parade"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# PARADE

![PARADE](https://proto-bio.github.io/proto-assets/images/tool/parade/hero.png)

> [!NOTE]
> **License:** PARADE is open source and free for academic and commercial use under an MIT license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/autosome-ru/parade/blob/master/LICENSE) for full terms.

## Overview

PARADE (Prediction And RAtional DEsign of mRNA UTRs) is a LegNet convolutional model for predicting cell-type-specific untranslated-region (UTR) activity and 3' UTR mRNA stability. This toolkit scores 5'/3' UTR sequences across the PARADE cell-line panel and predicts RNA/gDNA stability, using the checkpoints published with the paper.

## Background

PARADE ([Khoroshkin et al., 2024](https://doi.org/10.1101/2024.12.31.630783)) is a generative framework for designing UTRs with tailored cell-type-specific activity. Its predictive core adapts the DREAM-challenge LegNet architecture: an EfficientNet-style convolutional network with squeeze-and-excite blocks that reads a one-hot UTR sequence plus a reading-frame positional channel and, for activity, broadcast cell-condition channels.

The activity models are trained per construct type — one for 5' UTRs and one for 3' UTRs — and condition on a panel of named cell lines (`c1` MDA-MB-231, `c2` HepG2, `c4` Jurkat, `c6` SW480, `c17` NALM6, and, for 3' UTRs only, `c13` PA-1), returning a predicted activity mass-center for each. A separate 3' UTR model predicts mRNA stability as an RNA/gDNA log-ratio. The featurization matches the upstream reference pipeline exactly, so predictions reproduce the published values.

## Tools

### PARADE UTR Activity (`parade-activity`)

Predicts cell-type-specific activity for one or more 5' or 3' UTR sequences, returning one value per requested cell code.

#### Applications

Use this tool to rank UTR designs by predicted activity, screen candidate UTRs for a target cell line, or quantify the activity differential between cell types for cell-type-specific mRNA design.

#### Usage Tips

- **Pick the construct type.** Set `construct_type` to `utr5` or `utr3`; it selects the matching checkpoint and cell-code panel.
- **Cell codes are panel-specific.** `c13` exists only for `utr3`. Select a cell line by either its code (`c2`) or its cell-line name (`HepG2`, case-insensitive). Leave `cell_types` empty to return the full panel for the construct type.
- **Match the training length.** Upstream trained the 5' UTR model on ~50-nt inserts and the 3' UTR model on ~240-nt (roughly 200–300 nt) inserts; the model accepts any length (adaptive pooling) but predictions are only meaningful near the training regime.
- **Mixed lengths batch together.** Different-length sequences in one call are batched per length group; RNA input (`U`) is accepted and mapped to `T`.

### PARADE mRNA Stability (`parade-stability`)

Predicts 3' UTR mRNA stability as an RNA/gDNA log-ratio for one or more sequences; higher is more stable.

#### Applications

Use this tool to rank 3' UTR designs by predicted mRNA stability or to pair stability with cell-type-specific activity when selecting UTRs for downstream validation.

#### Usage Tips

- **Stability has no cell conditioning.** The model returns a single log-ratio per sequence.
- **Use the training length.** Upstream trained this stability model on 186-nt sequences (its `seqsize`); score near that length. Mixed lengths in one call are batched per length group.
- **Higher is more stable.** The `log_ratio` output is directly comparable across candidates.

### PARADE UTR Activity Gradient (`parade-gradient`)

Computes a weighted differentiable UTR-activity objective and, by default, returns the gradient with respect to batched relaxed UTR logits.

#### Applications

Use this tool inside gradient-based UTR design loops (e.g. Fast SeqProp) to maximize activity in an on-target cell line while minimizing it in off-target cell lines. It is designed for optimizer calls rather than final biological validation.

#### Usage Tips

- **Logits are batched.** Pass logits with shape `B x L x 4` in `A,C,G,T` order; use `B=1` for a single candidate. Every batch item must share one sequence length.
- **`temperature`** controls how the input logits are converted into nucleotide probabilities using `softmax(logits / temperature)`. Lower temperatures make the distribution more concentrated around the most likely nucleotide at each position, while higher temperatures produce more uniform distributions. The default is `1.0`, and the returned gradient is calculated with respect to the input logits. `soft` and `hard` then control how that relaxed state reaches the model.
- **Advanced option:** `temperature=None` skips softmax and treats the input as probabilities rather than logits. Most users should leave the default in place. In this case, each row must contain nonnegative values that sum to 1, and the returned gradient is calculated with respect to those probabilities. This is generally less convenient for gradient-based optimization because updating probabilities directly can cause them to become negative or no longer sum to 1. Optimizing logits avoids this issue, since softmax always produces a valid probability distribution. However, working directly with probabilities can be useful when using an optimizer designed for probability distributions or when exact zero probabilities are needed to exclude particular tokens.
- **Terms target cell codes.** Each loss term names a `cell_type` — its code (`c2`) or cell-line name (`HepG2`, case-insensitive) — a `direction` (`max`/`min`), and a `weight`; all must be in the `construct_type` panel.
- **`soft` and `hard` control the relaxation.** The default (`soft=1.0`, `hard=0.0`) feeds pure softmax probabilities. Set `hard=1.0` for a straight-through estimator — the forward pass sees a hard one-hot while gradients still flow through the probabilities. Lowering `soft` blends the raw input matrix in, which is differentiable but outside the model's training distribution.

### PARADE Stability Gradient (`parade-stability-gradient`)

Scores PARADE's predicted 3' UTR stability — the RNA/gDNA log-ratio, where higher means more stable — and returns its derivative with respect to a relaxed distribution over the UTR bases.

#### Applications

This is the differentiable form of `parade-stability`, intended as an objective inside a 3' UTR design loop: an optimizer holds a relaxed distribution over the UTR, asks for the gradient, and steps it toward sequences the model predicts will be more (or less) stable. Pair it with `parade-gradient` when a design needs both an activity profile and a stability target, treating the two as separate objectives.

#### Usage Tips

- **The objective is a single scalar, so there are no loss terms.** Where `parade-gradient` takes per-cell terms, stability takes just a `direction`: `"max"` minimizes `-log_ratio` and drives the sequence toward higher predicted stability, `"min"` does the reverse. The predicted `log_ratio` itself comes back in each candidate's metrics.
- **It runs the stability checkpoint**, which is a different model from the activity checkpoints — there is no `construct_type` here, since stability is not conditioned on a UTR construct.
- **One batch, one length.** The backend evaluates the batch in a single vectorized pass, so every candidate in a call must share a sequence length; the gradient comes back stacked and is split into one result per candidate.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PARADE tool in this toolkit (`parade-activity`, `parade-stability`, `parade-gradient`).

- **Runs on GPU or CPU.** The tools load a small PyTorch LegNet checkpoint; a GPU speeds up large batches but is not required.
- **Weights are provisioned automatically.** By default, the standalone worker downloads the published checkpoint from the pinned `autosome-ru/parade` commit into the managed model cache and verifies its MD5 checksum.
- **Predictions are faithful to the reference.** The vendored PARADE model/data modules are the verbatim upstream bodies (with only a provenance/Ruff header added per file), so the published checkpoints load and score exactly as they do upstream.

## References

- Khoroshkin, M. et al. A generative framework for enhanced cell-type specificity in rationally designed mRNAs. *bioRxiv* (2024). DOI: [10.1101/2024.12.31.630783](https://doi.org/10.1101/2024.12.31.630783)
- PARADE repository: [autosome-ru/parade](https://github.com/autosome-ru/parade)
- LegNet architecture: Penzar, D. et al. LegNet: a best-in-class deep learning model for short DNA regulatory regions. *Bioinformatics* 39 (2023). DOI: [10.1093/bioinformatics/btad457](https://doi.org/10.1093/bioinformatics/btad457)
