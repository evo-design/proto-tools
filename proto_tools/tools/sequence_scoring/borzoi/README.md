<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/borzoi"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Borzoi

> [!NOTE]
> **License:** Borzoi uses Apache-2.0 for code and CC-BY-4.0 for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/calico/borzoi/blob/main/LICENSE) and [model weights license](https://huggingface.co/johahi/borzoi-replicate-0) for full terms.

## Overview

[Borzoi](https://github.com/calico/borzoi) is a deep learning model that predicts cell-type and tissue-specific RNA-seq coverage and other regulatory genomics tracks directly from DNA sequence, developed by the Kelley lab at [Calico Life Sciences](https://www.calicolabs.com/). Given a fixed 524,288 bp input window, it returns activity across 6,144 bins of 32 bp each for thousands of human or mouse genomic assays. This toolkit exposes single-replicate prediction and a four-replicate ensemble for uncertainty estimation.

## Background

Gene regulation acts across a wide range of genomic distances. Most promoter-proximal elements operate within a few kilobases, yet enhancers can influence genes from more than 100 kb away, and [topologically associating domains](https://en.wikipedia.org/wiki/Topologically_associating_domain) organize chromatin contacts over megabase scales. Sequence-to-function models that aim to relate noncoding variation to molecular phenotype therefore require an input window broad enough to capture these long-range relationships at fine spatial resolution.

Borzoi ([Linder et al., 2025](https://doi.org/10.1038/s41588-024-02053-6)) learns to predict cell- and tissue-specific [RNA-seq](https://en.wikipedia.org/wiki/RNA-Seq) coverage from DNA sequence, serving as a unifying model of gene regulation. Using statistics computed from its predicted coverage, Borzoi isolates and accurately scores the DNA regulatory elements that modulate transcriptional processes including transcription, splicing, and polyadenylation, with greater accuracy than comparable models. Benchmarked against state-of-the-art models, it accurately predicts the influence of variants on RNA expression and splicing and recapitulates the causal variants underlying molecular quantitative trait loci. Alongside RNA-seq, the model predicts [CAGE](https://en.wikipedia.org/wiki/Cap_analysis_of_gene_expression), [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq), [ATAC-seq](https://en.wikipedia.org/wiki/ATAC-seq), and [histone modification](https://en.wikipedia.org/wiki/Histone_modification) tracks, making it a broad regulatory genomics predictor.

The published model couples a convolutional sequence encoder with transformer-style attention to process the full 524,288 bp window, and separate output heads produce human and mouse track predictions. The Borzoi authors trained four model replicates from independent initializations, which this toolkit exposes both as a single-replicate tool and as a four-replicate ensemble. A separate [FlashAttention](https://github.com/Dao-AILab/flash-attention)-based distillation of Borzoi, named Flashzoi, reaches comparable accuracy at substantially higher speed. The human checkpoints exposed by this toolkit use the Flashzoi distillation, and the mouse checkpoints use the standard Borzoi architecture.

### Learning Resources

- [calico/borzoi](https://github.com/calico/borzoi) (Calico Life Sciences). Official repository with the reference model code, training data references, and usage documentation.
- [Borzoi PyTorch weights](https://huggingface.co/collections/johahi/borzoi-models) (Hugging Face). The PyTorch-converted Borzoi and Flashzoi checkpoints that this toolkit loads at inference time.

## Tools

### Borzoi Prediction (`borzoi-prediction`)

Predicts regulatory track activity for one or more DNA sequences using a single Borzoi replicate. Each sequence may be supplied as an exact 524,288 bp model window, or as a longer source sequence paired with a sequence-relative target range that the tool uses to extract the fixed model window. For every input, the tool returns a per-bin activity matrix together with the source-sequence coordinates of the model input window and the output-bin span, so predictions can be mapped back onto the original sequence.

#### Applications

This tool is appropriate for high-throughput screening and iterative sequence design, where a single forward pass per sequence keeps the analysis fast. Representative applications include predicting RNA-seq, CAGE, and chromatin-accessibility profiles for a locus of interest, comparing reference and alternate alleles to estimate the regulatory effect of a noncoding variant, and ranking candidate regulatory sequences inside an optimization loop. The single-replicate setting is well suited to the inner iterations of a design campaign before a final ensemble assessment.

#### Usage Tips

- **Exact-window inputs must be exactly 524,288 bp.** When no target range is supplied, the provided sequence is treated as the literal model input and is rejected unless it matches the model context length. A longer genomic region should instead be paired with a target range so the tool can extract the fixed window.
- **A target range places a region of interest inside the output bins.** Extraction is aligned to the start of the requested range rather than centered, and the window shifts left near the right edge of the source sequence so the full range remains covered. The returned context and output coordinates report where the model window landed in source coordinates.
- **The region of interest is most informative near the center of the input window.** Predictions degrade toward the edges of the 524,288 bp context, so a target gene or variant is best positioned close to the midpoint of the supplied window.
- **The `species` setting selects the checkpoint family.** A value of `"human"` loads the FlashAttention Flashzoi checkpoints and requires a CUDA device, while `"mouse"` loads the standard Borzoi checkpoints. The two heads predict different track panels, so the species must match the organism of the input sequence.
- **`output_tracks` selects which assays are returned.** Track indices address the full Borzoi output panel (7611 human, 2608 mouse). Selecting a small set of relevant tracks is appropriate when only specific assays inform the analysis.
- **`avg_output_tracks=True` collapses the selected tracks into a single composite signal.** This default is appropriate when a single objective is needed, for example when combining related assays into one optimization score. A value of `False` returns one row per requested track when per-assay resolution is required.

### Borzoi Ensemble (`borzoi-ensemble`)

Predicts regulatory track activity using all four Borzoi replicates and returns the per-replicate predictions stacked together for each input sequence. The four replicates are evaluated in sequence and share the input handling, coordinate reporting, and track-selection behavior of the single-replicate tool. The spread of predictions across replicates provides a measure of model confidence at each bin.

#### Applications

This tool is appropriate for final assessments and for any analysis that benefits from uncertainty quantification. Computing the dispersion across the four replicate predictions at each bin distinguishes positions where the model is confident from positions where the replicates disagree. Representative applications include reporting confidence intervals on a predicted regulatory profile, filtering candidate variants or designed sequences to those with consistent predicted effects, and producing the headline numbers for a locus after single-replicate screening has narrowed the candidates.

#### Usage Tips

- **The ensemble runs four full forward passes per sequence.** Inference therefore takes roughly four times as long as a single replicate. The single-replicate tool is appropriate for iteration, and the ensemble is appropriate for the final reportable result.
- **Confidence is read from agreement across replicates.** A low spread across the four predictions at a bin indicates a robust signal, while a high spread indicates lower model confidence at that position. The per-replicate predictions are returned in full so any dispersion statistic can be computed downstream.
- **Replicate selection is not exposed for the ensemble.** All four replicates are always evaluated. The single-replicate tool is the appropriate choice when only one specific replicate is needed.
- **The species, track-selection, and averaging behavior match the single-replicate tool.** The same `species`, `output_tracks`, and `avg_output_tracks` guidance applies, and the ensemble inherits the same input modes and coordinate reporting.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Borzoi tool in this toolkit (`borzoi-prediction`, `borzoi-ensemble`).

- **A CUDA GPU is required.** Both tools run on GPU, and human prediction uses FlashAttention kernels that are available only on CUDA hardware. The model checkpoints are downloaded from Hugging Face on first use and cached for subsequent runs.
- **Input sequences accept only the bases A, C, G, T, and N.** Other characters are rejected during validation. The base N is permitted but encoded as the absence of any base, so a high N content reduces prediction quality and is best minimized.
- **Predicted values are the model's raw track-activity outputs, returned without any additional post-processing.** Higher values correspond to stronger predicted signal. The values are best used for relative comparisons, for example between alleles or across positions, rather than as absolute experimental counts.
- **Output bins map to source coordinates through the reported window.** Each result reports the output-bin span in source-sequence coordinates at a resolution of 32 bp per bin, so a bin index can be converted to a genomic position using the output start and the bin resolution.
