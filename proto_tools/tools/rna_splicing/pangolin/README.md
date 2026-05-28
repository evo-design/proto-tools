<a href="https://bio-pro.mintlify.app/tools/rna-splicing/pangolin"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Pangolin

> [!NOTE]
> **License:** Pangolin has a GPL-3.0 license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/tkzeng/Pangolin/blob/main/LICENSE) for full terms.

## Overview

[Pangolin](https://github.com/tkzeng/Pangolin) ([Zeng & Li, 2022](https://doi.org/10.1186/s13059-022-02664-4)) is a deep-learning splice-prediction model built by Tony Zeng and Yang I. Li at the University of Chicago. In the [SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015) lineage of dilated convolutional networks, it predicts per-position, tissue-specific [splice-site](https://en.wikipedia.org/wiki/RNA_splicing) strength directly from DNA sequence and scores the splicing effect of genetic variants. This toolkit wraps both capabilities as typed tools: `pangolin-predict` for per-position score prediction and `pangolin-score-variants` for variant gain/loss scoring.

## Background

[Pre-mRNA splicing](https://en.wikipedia.org/wiki/RNA_splicing) removes [introns](https://en.wikipedia.org/wiki/Intron) and joins [exons](https://en.wikipedia.org/wiki/Exon), with the spliceosome recognizing the donor (5') and acceptor (3') splice sites that bound each intron. Which sites are used, and how often, varies across tissues and underlies much of the transcriptome's [alternative splicing](https://en.wikipedia.org/wiki/Alternative_splicing) diversity. Variants that create or destroy splice sites are a major and frequently under-recognized cause of genetic disease, which motivates models that can read splicing regulation straight from sequence.

Pangolin extends the [SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015) dilated-CNN approach in two directions ([Zeng & Li, 2022](https://doi.org/10.1186/s13059-022-02664-4)). First, it is trained on quantitative, tissue-specific splicing measurements (including the fraction of transcripts that use a given site), and emits a per-tissue splice-site probability rather than SpliceAI's single tissue-agnostic score. Second, it is trained across four tissues - **heart, liver, brain, and testis** - and across multiple species (human, rhesus, mouse, rat), which improves generalization and lets the model report tissue-specific predictions. The released model is an ensemble of checkpoints; predictions for a tissue average the relevant ensemble members. As with SpliceAI, the network consumes a wide window of flanking sequence to capture the long-range context that governs splice-site choice.

These tools expose the per-tissue **splice-site probability** score — the same P(splice) head Pangolin's reference CLI uses for variant scoring (not the separate transcript-usage head). Variant scoring reduces that score across the selected tissues into a per-position splice gain and loss.

### Learning Resources

- [Pangolin repository](https://github.com/tkzeng/Pangolin) (Zeng & Li, University of Chicago) - source, pretrained ensemble weights, and the reference CLI this wrapper mirrors.
- [Predicting RNA splicing from DNA sequence using Pangolin](https://doi.org/10.1186/s13059-022-02664-4) (Genome Biology, 2022) - the primary publication, with training setup, tissue/usage formulation, and variant-scoring benchmarks.
- [SpliceAI](https://doi.org/10.1016/j.cell.2018.12.015) (Jaganathan et al., 2019) - the dilated-CNN splice-prediction model that Pangolin builds on.

## Tools

### Pangolin Splice-Site Prediction (`pangolin-predict`)

Predicts per-position, tissue-specific splice-site probability scores along one or more DNA sequences.

#### Applications

Use this to scan a gene, transcript, or designed sequence for where splice sites are predicted and how strongly, resolved by tissue. Typical workflows include mapping the donor/acceptor splice-score landscape of a locus, comparing predicted scores across heart/liver/brain/testis to find tissue-specific sites, and generating per-position tracks for downstream visualization or differential analysis.

#### Usage Tips

- **Each sequence needs 5,000 bp of flanking context on each side** (`PANGOLIN_FLANK`). A length-`N` sequence yields predictions for the central `N - 10000` positions, so the minimum input is 10,001 bp. The `output_start` field reports the input index (always `5000`) of the first scored position.
- **`tissues`** selects which of `HEART`, `LIVER`, `BRAIN`, `TESTIS` are ensembled (default: all four). The score columns are emitted in the order given by `tissues`, so request only the tissues you need and read columns by that order.
- Inputs accept a single sequence string (auto-wrapped) or a list; outputs are 1:1 with inputs. Sequences are validated as DNA (A/C/G/T/N, uppercased) before scoring.

### Pangolin Variant Splice Scoring (`pangolin-score-variants`)

Scores the splicing gain/loss effect of variants by comparing the predicted splice-site probability between the reference and alternate sequence.

#### Applications

Use this to prioritize candidate splice-altering variants - SNVs and simple indels - by how much they are predicted to increase (gain) or decrease (loss) the splice-site probability near the variant. It suits variant-interpretation pipelines and saturation/screen analyses where each variant is supplied with its local reference window.

#### Usage Tips

- **Variant scoring is sequence-centric: no genome FASTA is required.** Provide each variant's reference window (`sequence`), the 0-based `variant_position`, and the `reference_bases`/`alternate_bases` alleles. The reference allele must match the window at that position, and the variant needs **5,000 bp of flank on each side** (`PANGOLIN_FLANK`).
- **`distance`** (default `50`) sets the ± reporting window around the variant. To report scores over the full window the sequence should provide `PANGOLIN_FLANK + distance` bp of flank on each side; with less context the reporting window is clipped to the available flank.
- **`tissues`** behaves as in prediction: gain and loss are reduced (max increase / max decrease) across the selected tissues. `max_gain`/`max_loss` summary metrics and the `increase_position`/`decrease_position` peaks are reported relative to the variant in bp.
- **Annotation-based score masking (the upstream CLI `--mask` option) is not supported**, because it requires exon annotations; raw gain/loss scores are returned.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to both tools in this toolkit (`pangolin-predict`, `pangolin-score-variants`).

- **GPU recommended.** Pangolin runs on GPU (default `device="cuda"`) for practical throughput; CPU works but is slow, especially for long sequences or many variants.
- **Model weights ship inside the pip package** (~180 MB) and are installed automatically with the standalone environment - no separate weight download or gated access is required.
- **Deterministic outputs.** Pangolin inference is deterministic: the same sequence and tissue selection produce the same scores, so results are cacheable and reproducible.
