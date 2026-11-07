<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/cherimoya"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Cherimoya

![Cherimoya](https://proto-bio.github.io/proto-assets/images/tool/cherimoya/hero.png)

> [!NOTE]
> **License:** Cherimoya uses MIT for code and CC-BY-4.0 for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/jmschrei/cherimoya/blob/main/LICENSE) and [model weights license](https://huggingface.co/programmable-genomics/CATv1/blob/main/LICENSE) for full terms.

## Overview

Cherimoya is a compact sequence-to-function model from the [Programmable Genomics Laboratory](https://github.com/jmschrei/cherimoya) that predicts base-resolution chromatin accessibility or protein binding from DNA sequence. This toolkit runs two collections, one model per ENCODE experiment with five folds each: CATv1 (1,518 DNase-seq and ATAC-seq experiments) and BATv1 (3,429 ChIP-seq experiments), and lets you search them, score and profile sequences, and take gradients for design.

## Background

Cherimoya ([Ramirez et al., 2026](https://doi.org/10.5281/zenodo.21535161)) adapts the BPNet/ChromBPNet architecture into a residual convolutional network of about 614,000 parameters. It takes one-hot DNA of 2,114 bp and predicts two quantities over the central 1,000 bp: a profile head (the shape of the read distribution, one value per base) and a count head (ln of the total reads plus one). Multiplying the softmaxed profile by the expected total gives predicted reads per base.

CATv1 is the Cherimoya Accessibility aTlas, v1: models trained on 1,149 DNase-seq and 369 ATAC-seq experiments from the [ENCODE Project](https://doi.org/10.1038/s41586-020-2493-4), on GRCh38, across 407 biosamples. Each experiment has five models, one per chromosome-held-out cross-validation fold. A model reflects only the assay and biosample of the experiment it was trained on, so pick experiments with `cherimoya-catalog` rather than by cell type name alone.

BATv1 is the Cherimoya Binding ATlas, v1: smaller models (about 46,000 parameters) trained on 3,429 ENCODE ChIP-seq experiments of 1,139 transcription factors and other DNA-associated proteins across 183 biosamples, with the same five folds. Each predicts a strand-specific profile from the sequence and the experiment's control track. This toolkit runs them with an all-zero control, as the release recommends for sequence-only predictions, and sums the two strands, so every tool returns the same shapes for both collections. A model reflects one target in one biosample; search BATv1 with the `target` filter.

### Learning Resources

- [Cherimoya: a lightweight genomic sequence-to-function model built for large-scale design and understanding](https://genomicsxai.github.io/blogs/2026-011/) (Genomics x AI) - the announcement post introducing the model, the CATv1 atlas, and the design workflows they are built for.
- [jmschrei/cherimoya](https://github.com/jmschrei/cherimoya) (Programmable Genomics Laboratory) - the reference implementation, with the model code and training scripts.
- [CATv1 model weights](https://huggingface.co/programmable-genomics/CATv1) (Hugging Face) - the released accessibility checkpoints this toolkit loads, one per ENCODE experiment and fold.
- [BATv1 model weights](https://huggingface.co/programmable-genomics/BATv1) (Hugging Face) - the released ChIP-seq checkpoints, with the metadata and quality flags behind the bundled catalog.

## Tools

### Cherimoya Catalog (`cherimoya-catalog`)

Searches the experiments in a collection by biosample, ChIP-seq target, assay, biosample classification, accession, and validation performance, and marks one default experiment per (assay, biosample) in CATv1.

#### Applications

Use this tool to find which ENCODE experiments have models for a cell type or tissue, or for a transcription factor in a cell type, before scoring, to choose between several experiments of the same biosample, and to look up the assay, target and biosample behind an accession.

#### Usage Tips

- **The biosample and target filters are regular expressions.** They are case-insensitive; `biosample` matches the biosample term name and summary, so `hepat|liver` finds hepatocytes and liver tissue, and `target` matches the ChIP-seq target, so `^GATA1$` with `collection="batv1"` finds GATA1 experiments and `GATA` the whole family.
- **Defaults come from Chorus.** In CATv1, `is_default` marks the experiment [Chorus](https://github.com/pinellolab/chorus) resolves an (assay, biosample) to: the best fold 0 count Pearson, except 9 pairs pinned to the experiment behind Chorus's ChromBPNet models. BATv1 has no defaults yet, so `defaults_only` returns nothing there; rank its matches by `count_pearson_fold_0` and read `quality_flags`.
- **Perturbed experiments are hidden from searches by default.** Set `include_perturbed=True` to see them (8 in CATv1, 282 in BATv1); named `accessions` and `defaults_only` results are returned either way, and 4 of the CATv1 defaults (MCF-7, NAMALWA, SK-N-SH and WI38 DNase-seq) are perturbed experiments.
- **Read `quality_flags` before trusting a BATv1 model.** The release flags 268 experiments: `few_test_peaks`, `early_checkpoint` (a fold whose checkpoint is from epoch 0 or 1, with near-uninformative profiles), `nonpositive_test_pearson` and `small_training_set`.

### Cherimoya Score (`cherimoya-score`)

Predicts total accessibility or binding for DNA sequences with one or more models and returns log counts over the 1,000 bp output window, plus expected reads inside a target range when one is given.

#### Applications

Use this tool to rank regulatory element designs by predicted DNase-seq or ATAC-seq signal in a chosen biosample, or by predicted ChIP-seq signal of a transcription factor, compare a sequence across many cell types or factors at once, or screen candidates before profiling them.

#### Usage Tips

- **Sequences are 2,114 bp windows.** Pass exactly 2,114 bp, or a longer sequence with a `target_range` of at most 1,000 bp and the tool cuts the window.
- **Compare experiments with care.** Raw log counts scale with each experiment's sequencing depth, so the same value means different things in two experiments, and a BATv1 model run with an all-zero control predicts lower counts than the observed track.
- **Folds and strands are options.** The default runs fold 0; list more `folds` to average them, and set `reverse_complement_average=True` to average both strands.

### Cherimoya Predict (`cherimoya-predict`)

Predicts a base-resolution accessibility or binding profile over the 1,000 bp output window for each sequence and model.

#### Applications

Use this tool to see where accessibility or a factor's binding falls within an element, compare the profile shape between cell types, or plot predicted signal next to measured tracks.

#### Usage Tips

- **Choose the values.** `values="expected_counts"` returns predicted reads per base and `values="probabilities"` returns the profile shape, which sums to 1.
- **BATv1 profiles are summed over strands.** A ChIP-seq model predicts one distribution over the plus and minus strands; the tool adds the two, so the profile is the probability (or expected reads) of a read at each base on either strand.
- **Profiles cover the output window only.** Each profile spans `output_start` to `output_end` of the input sequence, the central 1,000 bp of the 2,114 bp context.

### Cherimoya Gradient (`cherimoya-gradient`)

Computes a weighted differentiable accessibility or binding objective over one or more models and, by default, returns its gradient with respect to batched relaxed DNA logits.

#### Applications

Use this tool inside gradient-based DNA design loops, for example to raise accessibility in one biosample while lowering it in others, or to add or remove a transcription factor's predicted binding. It is designed for optimizer calls rather than final biological validation.

#### Usage Tips

- **Context fills the window.** `context_left`, the designed logits, and `context_right` must total 2,114 bp; only the designed segment receives a gradient.
- **Set the sigmoid per model.** Both quantities are ln(reads + 1), `log_counts` over the output window and `log_target_counts` over `target_start`..`target_end`, and both depend on each experiment's depth; the default `sigmoid_center` of 4.5 is the median `log_counts` of the default model over hg38 cCREs. BATv1 models predict on a lower scale with their all-zero control, so center their terms on a `cherimoya-score` value of the starting sequence.
- **Use `deterministic=True` for reproducible gradients.** The default Triton kernels are faster but do not repeat bit-for-bit on GPU.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Cherimoya tool in this toolkit (`cherimoya-catalog`, `cherimoya-score`, `cherimoya-predict`, `cherimoya-gradient`).

- **Models are named by ENCODE experiment accession.** `models` (and each gradient loss term's `model`) takes accessions such as `ENCSR000EOT` (K562 DNase-seq), validated against the bundled catalog of the configured `collection`. The default `collection` is `catv1`; set `collection="batv1"` and name the models (for example `ENCSR000EGM`, K562 CTCF) for ChIP-seq, since the default model belongs to CATv1.
- **BATv1 runs without a control track.** The ChIP-seq models were trained with the experiment's control signal as a second input, which these tools cannot take, so they receive an all-zero control: predictions are the sequence's own contribution, as the BATv1 release recommends, not a reconstruction of the observed track with its control biases.
- **Your own checkpoints work locally.** A path to a Cherimoya `.torch` file with the same 2,114 bp in, 1,000 bp out geometry can stand in for an accession; control tracks are zero-filled and output channels summed, as for BATv1. Remote devices reject paths.
- **Weights download on first use.** Each fold checkpoint (about 2.5 MB for CATv1, 0.2 MB for BATv1) is fetched from Hugging Face at a pinned revision into the managed model cache, and loaded checkpoints stay in the worker between calls.
- **Human GRCh38 only.** Every CATv1 and BATv1 model was trained on human ENCODE data; for another species, any model is a proxy.
- **Runs on CPU or GPU.** A GPU is much faster; on CPU expect tens of sequences per second per model.
