<a href="https://bio-pro.mintlify.app/tools/rna-splicing/spliceai"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# SpliceAI

> [!NOTE]
> **License:** SpliceAI uses Custom (PolyForm Strict License 1.0.0) for code and CC-BY-NC-4.0 for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/Illumina/SpliceAI/blob/master/LICENSE) and [model weights license](https://creativecommons.org/licenses/by-nc/4.0/) for full terms.

## Overview

[SpliceAI](https://github.com/Illumina/SpliceAI) is Illumina's deep residual neural network that predicts RNA splice donor and acceptor sites directly from pre-mRNA sequence, and quantifies how genetic variants alter splicing. This wrapper exposes two tools: variant delta-score annotation (the signature SpliceAI workflow) and raw per-position splice-site prediction. Inference runs locally through an isolated standalone venv on GPU or CPU.

## Background

[RNA splicing](https://en.wikipedia.org/wiki/RNA_splicing) removes introns from pre-mRNA and joins exons, guided by sequence motifs at the donor (5') and acceptor (3') splice sites. Variants that create or disrupt these motifs can cause exon skipping, intron retention, or cryptic splicing, and are a major and frequently overlooked class of disease-causing mutations. SpliceAI ([Jaganathan et al., 2019](https://doi.org/10.1016/j.cell.2018.12.015)) is a deep dilated residual convolutional network that reads 10,000 bp of flanking context (5,000 bp per side) and outputs, for every position, the probability of being an acceptor, a donor, or neither.

For variant interpretation, SpliceAI compares predictions for the reference and alternate sequences and reports four **delta scores** in [0, 1] — acceptor gain (DS_AG), acceptor loss (DS_AL), donor gain (DS_DG), and donor loss (DS_DL) — together with the **delta positions** (DP_*) of the affected sites relative to the variant. The maximum delta score is the headline number: the paper characterizes cutoffs of 0.2 (high recall), 0.5 (recommended), and 0.8 (high precision). The shipped model is an ensemble of five models whose per-position outputs are averaged. All variant coordinates follow the 1-based VCF convention.

### Learning Resources

- [SpliceAI repository](https://github.com/Illumina/SpliceAI) (Illumina) - the canonical CLI, the `Annotator`/`get_delta_scores` Python API, and the bundled GENCODE annotations and ensemble weights.
- [Jaganathan et al., 2019](https://doi.org/10.1016/j.cell.2018.12.015) (Cell) - the original paper describing the architecture, training data, and clinical validation of delta scores.

## Tools

### SpliceAI Variant Scoring (`spliceai-score`)

Scores genetic variants (chromosome / 1-based position / ref / alt) for splice-altering effects, returning per-gene delta scores and delta positions for acceptor and donor gain/loss. Requires a reference genome FASTA and a gene annotation (the bundled `grch37`/`grch38`, or a custom file).

#### Applications

Use this to triage candidate variants from a sequencing study for splicing impact, to annotate a VCF with SpliceAI predictions, or to prioritize variants of uncertain significance where a coding effect is absent but a splicing effect is plausible. The `max_delta_score` metric supports threshold-based filtering at the recommended 0.2 / 0.5 / 0.8 cutoffs.

#### Usage Tips

- **`reference_fasta` is required and `position` is 1-based.** SpliceAI extracts the wild-type window around each variant from the genome you supply, so the FASTA, the annotation, and each variant's `chromosome` must use consistent identifiers. Note this is the opposite of `AlphaGenome`, whose coordinates are 0-based.
- **`annotation` selects the gene model.** `grch37` and `grch38` load the GENCODE files bundled with SpliceAI; pass a path to score against a custom tab-separated annotation. Changing it restarts the worker.
- **`max_distance` (default 50) and `mask` mirror the SpliceAI `-D`/`-M` flags.** Widen `max_distance` to report splice sites farther from the variant; enable `mask` to suppress scores for annotated-gain and unannotated-loss positions.

### SpliceAI Splice-Site Prediction (`spliceai-predict`)

Predicts per-position `[neither, acceptor, donor]` probabilities directly from one or more DNA sequences. No reference genome is needed — the model runs on the sequence as given, padding 5,000 bp of context per side internally.

#### Applications

Use this to scan an engineered construct, a minigene, or a transcript for latent splice sites, to visualize the acceptor/donor probability landscape across a region of interest, or to compare splice-site usage between designed sequence variants without assembling a genome and annotation.

#### Usage Tips

- **Output channels are `[neither, acceptor, donor]`.** Index channel 1 for acceptor and channel 2 for donor probabilities; each per-sequence array has the same length as the corresponding input sequence.
- **Sequences may differ in length.** They are scored independently (per-item caching applies), so batching ragged sequences is fine; very short sequences still receive the full 10,000 bp `N`-padded context.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to both SpliceAI tools in this toolkit (`spliceai-score`, `spliceai-predict`).

- **Runs on GPU or CPU via TensorFlow.** SpliceAI is the only TensorFlow tool in the catalog; the standalone env pins TensorFlow 2.15 (Keras 2) so the bundled `.h5` models load, which constrains the runtime to Python 3.11. TensorFlow falls back to CPU automatically when no GPU is visible.
- **Weights and annotations ship with the package.** The five ensemble models and the GENCODE `grch37`/`grch38` annotations are bundled in `pip install spliceai`, so no weight download is needed. The reference genome FASTA for `spliceai-score` is user-supplied at call time.
- **Non-commercial license.** SpliceAI's code is PolyForm Strict and its bundled models are CC-BY-NC-4.0 — both noncommercial, so the toolkit is not hostable on Proto; commercial use requires a license from Illumina.
