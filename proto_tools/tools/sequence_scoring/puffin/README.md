<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/puffin"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Puffin

![Puffin](https://proto-bio.github.io/proto-assets/images/tool/puffin/hero.png)

> [!NOTE]
> **License:** Puffin is licensed under Custom (UTSW Academic Software License) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://github.com/jzhoulab/puffin/blob/main/LICENSE) for full terms.

## Overview

Puffin is a sequence-based deep learning model for transcription initiation that predicts per-base initiation signal from DNA and decomposes the prediction into a small set of learned promoter motifs. This toolkit exposes a fast prediction path (`puffin-prediction`), a gradient-based motif-decomposition path (`puffin-interpretation`), and a differentiable track-signal objective (`puffin-gradient`) over the same checkpoint.

## Background

In 2024, [Dudnyk et al.](https://doi.org/10.1126/science.adj0116) introduced Puffin, a deep learning model that explains transcription initiation in the human genome at basepair resolution from sequence alone. The model is trained against five [transcription initiation](https://en.wikipedia.org/wiki/Transcription_(biology)) assays (FANTOM CAGE, ENCODE CAGE, ENCODE RAMPAGE, GRO-cap, PRO-cap), each predicted on both strands. The output is a per-base 10-channel signal that can be interpreted as `ln(count_scale_signal + 1)`.

Puffin is structurally constrained: its first convolutional layer plays the role of a learned [motif](https://en.wikipedia.org/wiki/Sequence_motif) filter bank, and the model exposes per-base activation and contribution scores for nine promoter motifs (CREB, ETS, NFY, NRF1, SP, TATA, U1_snRNP, YY1, ZNF143) on each strand. A tenth `Long Inr` filter is used internally by the model to construct the per-base initiator-effect track but is not exposed per-motif. The minimum input is 651 bp because the model uses 325 bp of padding on each side of the predicted output span.

The wrapper accepts raw DNA strings. The upstream coordinate / region / FASTA-file CLI modes (which require an hg38 reference) are intentionally not exposed and callers extract genomic sequences themselves.

## Tools

### Puffin Prediction (`puffin-prediction`)

Runs a single forward pass through Puffin and returns per-base predictions across all 10 transcription-initiation channels (5 assays × 2 strands) at single-base resolution.

#### Applications

Use this tool to score transcription start sites, rank candidate promoters, or measure the per-base effect of variants and edits across five capped-5'-end assays in one call. The fast path is the right choice when the question is *how much* signal a sequence produces rather than *why*.

#### Usage Tips

- **Per-base output length is `len(sequence) - 650`.** The model uses 325 bp of padding on each side; output coordinates run from 325 to `len(sequence) - 325` in the input frame.
- **Channel order is mirrored across strands.** The first 5 channels are FANTOM_CAGE+ → PRO_CAP+; the next 5 are PRO_CAP- → FANTOM_CAGE-. Index by name via `TRACK_NAMES.index(...)` rather than memorizing positions.
- **Outputs are in log scale.** Treat predicted values as `ln(count_scale_signal + 1)`. To compare two sequences, subtract. The difference is already in log space.

### Puffin Interpretation (`puffin-interpretation`)

Runs Puffin's gradient-based decomposition for one chosen target assay and strand. Returns the per-base prediction for that target, 18 motif-activation tracks, 18 motif-effect tracks, and per-base basepair-contribution scores both as an aggregate and decomposed two ways (contribution to the predicted signal per motif, and contribution to each motif's activation per basepair; 18 tracks each). Summed motif, initiator, trinucleotide, and total-effect tracks are also returned.

#### Applications

Use this tool to ask which motif drives a transcription start site, how a variant changes a motif activation, or how initiator and trinucleotide context shape the predicted signal. It is substantially slower than `puffin-prediction` because it computes per-base gradient contributions, so reach for it for mechanistic follow-up on specific sequences rather than for bulk scoring.

#### Usage Tips

- **`target_signal` picks which assay's prediction is decomposed.** Choose the one closest to the biological question; CAGE/RAMPAGE measure capped mRNA 5' ends, while GRO-cap/PRO-cap measure nascent transcription.
- **`reverse_strand` selects which strand head to interpret.** Defaults to forward; run it twice on the same input to analyze divergent or antisense promoters.
- **Motif dicts use strand-suffixed keys.** Access `motif_activations["TATA+"]` and `motif_activations["TATA-"]`, never the bare motif name. `MOTIF_NAMES` lists the 9 motif stems.

### Puffin Gradient (`puffin-gradient`)

Reports Puffin's predicted transcription-initiation signal and its derivative with respect to a relaxed distribution over the input bases, one quantity at a time. Each requested term names an output track and strand plus a span of output positions, and comes back with its own mean signal and its own raw gradient.

#### Applications

This is the differentiable form of Puffin prediction, intended as an objective inside a promoter-design loop: an optimizer holds a relaxed distribution over the sequence, asks for the gradients, and combines them into whatever objective it wants (concentrating initiation at a position, raising one assay's signal, or suppressing the opposite strand), then steps the sequence accordingly. Setting `compute_gradient=False` turns it into a forward-only signal score for ranking candidates.

#### Usage Tips

- **Fixed flanks go in `context_left`/`context_right`, never in the relaxed matrix.** The model input is `context_left + candidate + context_right`; the contexts are encoded exactly as `puffin-prediction` encodes a string (`N` as 0.25 per base) and are never relaxed, so any `soft`/`hard` schedule leaves them intact. Only the candidate's `(L, 4)` rows are relaxed and carry gradient. With both contexts empty (the default) the candidate is the whole model input.
- **The model input needs at least 651 bp.** Puffin trims 325 bp from each end, so the scored output span is `len(context_left) + L + len(context_right) - 650` positions and anything shorter scores nothing. 325 bp of context on each side makes the output window exactly the candidate.
- **Position spans are 1-indexed, inclusive, and relative to the output window** of the full model input, not to the candidate. Output position `p` is candidate row `p + 325 - len(context_left)`. Leaving `position_start`/`position_end` unset covers the whole output window, and the term that comes back names the span actually averaged rather than the `None` that was asked for.
- **Tracks are named with a strand suffix.** `FANTOM_CAGE+` and `FANTOM_CAGE-` select the forward and reverse heads, which are distinct model channels; the ten names match `TRACK_NAMES`.
- **Several tracks share one forward pass.** Each requested quantity gets its own backward over that shared graph, so asking for K tracks costs one forward and K backwards rather than K of each.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Puffin tool in this toolkit (`puffin-prediction`, `puffin-interpretation`, `puffin-gradient`).

- **GPU recommended but not required.** All three tools run on CPU; `puffin-interpretation` is materially slower than `puffin-prediction` on either device because it backpropagates through every output position and motif.
- **Sequence input only.** The upstream CLI's coordinate / region / FASTA-file modes require an `hg38.fa` reference and are intentionally not wrapped; callers extract DNA themselves and pass it as a string.
- **All three tools share one persistent worker.** They dispatch against the same `puffin` toolkit and load the Puffin model once per worker process; switching between prediction and interpretation does not reload weights.

## References

- Dudnyk, K., Cai, D., Shi, C., Xu, J., Zhou, J. Sequence basis of transcription initiation in the human genome. *Science* 384, eadj0116 (2024). DOI: [10.1126/science.adj0116](https://doi.org/10.1126/science.adj0116)
- Upstream repository: [jzhoulab/puffin](https://github.com/jzhoulab/puffin)
