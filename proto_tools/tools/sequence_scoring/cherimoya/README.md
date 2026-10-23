<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/cherimoya"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Cherimoya

![Cherimoya](https://proto-bio.github.io/proto-assets/images/tool/cherimoya/hero.png)

> [!NOTE]
> **License:** Cherimoya uses MIT for code and CC-BY-4.0 for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/jmschrei/cherimoya/blob/main/LICENSE) and [model weights license](https://huggingface.co/programmable-genomics/CATv1/blob/main/LICENSE) for full terms.

## Overview

Cherimoya is a compact sequence-to-function model from the [Programmable Genomics Laboratory](https://github.com/jmschrei/cherimoya) that predicts base-resolution chromatin accessibility from DNA sequence. This toolkit runs the CATv1 collection, one model per ENCODE DNase-seq or ATAC-seq experiment (1,518 experiments, five folds each), and lets you search it, score and profile sequences, and take gradients for design.

## Background

Cherimoya ([Ramirez et al., 2026](https://doi.org/10.5281/zenodo.21535161)) adapts the BPNet/ChromBPNet architecture into a residual convolutional network of about 614,000 parameters. It takes one-hot DNA of 2,114 bp and predicts two quantities over the central 1,000 bp: a profile head (the shape of the read distribution, one value per base) and a count head (ln of the total reads plus one). Multiplying the softmaxed profile by the expected total gives predicted reads per base.

CATv1 is the Cherimoya Accessibility aTlas, v1: models trained on 1,149 DNase-seq and 369 ATAC-seq experiments from the [ENCODE Project](https://doi.org/10.1038/s41586-020-2493-4), on GRCh38, across 407 biosamples. Each experiment has five models, one per chromosome-held-out cross-validation fold. A model reflects only the assay and biosample of the experiment it was trained on, so pick experiments with `cherimoya-catalog` rather than by cell type name alone.

## Tools

### Cherimoya Catalog (`cherimoya-catalog`)

Searches the experiments in a collection by biosample, assay, biosample classification, accession, and validation performance, and marks one default experiment per (assay, biosample).

#### Applications

Use this tool to find which ENCODE experiments have models for a cell type or tissue before scoring, to choose between several experiments of the same biosample, and to look up the assay and biosample behind an accession.

#### Usage Tips

- **The biosample filter is a regular expression.** It is case-insensitive and matches the biosample term name and summary, so `hepat|liver` finds hepatocytes and liver tissue.
- **Defaults come from Chorus.** `is_default` marks the experiment [Chorus](https://github.com/pinellolab/chorus) resolves an (assay, biosample) to: the best fold 0 count Pearson, except 9 pairs pinned to the experiment behind Chorus's ChromBPNet models.
- **Perturbed experiments are hidden from searches by default.** Set `include_perturbed=True` to see the 8 that ENCODE marks as perturbed; named `accessions` and `defaults_only` results are returned either way, and 4 of the defaults (MCF-7, NAMALWA, SK-N-SH and WI38 DNase-seq) are perturbed experiments.

### Cherimoya Score (`cherimoya-score`)

Predicts total accessibility for DNA sequences with one or more models and returns log counts over the 1,000 bp output window, plus expected reads inside a target range when one is given.

#### Applications

Use this tool to rank regulatory element designs by predicted DNase-seq or ATAC-seq signal in a chosen biosample, compare a sequence across many cell types at once, or screen candidates before profiling them.

#### Usage Tips

- **Sequences are 2,114 bp windows.** Pass exactly 2,114 bp, or a longer sequence with a `target_range` of at most 1,000 bp and the tool cuts the window.
- **Compare experiments with care.** Raw log counts scale with each experiment's sequencing depth, so the same value means different things in two experiments.
- **Folds and strands are options.** The default runs fold 0; list more `folds` to average them, and set `reverse_complement_average=True` to average both strands.

### Cherimoya Predict (`cherimoya-predict`)

Predicts a base-resolution accessibility profile over the 1,000 bp output window for each sequence and model.

#### Applications

Use this tool to see where accessibility falls within an element, compare the profile shape between cell types, or plot predicted signal next to measured tracks.

#### Usage Tips

- **Choose the values.** `values="expected_counts"` returns predicted reads per base and `values="probabilities"` returns the profile shape, which sums to 1.
- **Profiles cover the output window only.** Each profile spans `output_start` to `output_end` of the input sequence, the central 1,000 bp of the 2,114 bp context.

### Cherimoya Gradient (`cherimoya-gradient`)

Computes a weighted differentiable accessibility objective over one or more models and, by default, returns its gradient with respect to batched relaxed DNA logits.

#### Applications

Use this tool inside gradient-based DNA design loops, for example to raise accessibility in one biosample while lowering it in others. It is designed for optimizer calls rather than final biological validation.

#### Usage Tips

- **Context fills the window.** `context_left`, the designed logits, and `context_right` must total 2,114 bp; only the designed segment receives a gradient.
- **Set the sigmoid per model.** Both quantities are ln(reads + 1), `log_counts` over the output window and `log_target_counts` over `target_start`..`target_end`, and both depend on each experiment's depth; the default `sigmoid_center` of 4.5 is the median `log_counts` of the default model over hg38 cCREs.
- **Use `deterministic=True` for reproducible gradients.** The default Triton kernels are faster but do not repeat bit-for-bit on GPU.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Cherimoya tool in this toolkit (`cherimoya-catalog`, `cherimoya-score`, `cherimoya-predict`, `cherimoya-gradient`).

- **Models are named by ENCODE experiment accession.** `models` (and each gradient loss term's `model`) takes accessions such as `ENCSR000EOT` (K562 DNase-seq), validated against the bundled catalog.
- **Your own checkpoints work locally.** A path to a Cherimoya `.torch` file with the same 2,114 bp in, 1,000 bp out geometry, no control tracks, and one output can stand in for an accession; remote devices reject paths.
- **Weights download on first use.** Each fold checkpoint (about 2.5 MB) is fetched from Hugging Face at a pinned revision into the managed model cache, and loaded checkpoints stay in the worker between calls.
- **Human GRCh38 only.** Every CATv1 model was trained on human ENCODE data; for another species, any model is a proxy.
- **Runs on CPU or GPU.** A GPU is much faster; on CPU expect tens of sequences per second per model.
