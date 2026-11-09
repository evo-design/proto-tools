<a href="https://bio-pro.mintlify.app/tools/rna-splicing/spliceai2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# SpliceAI2

![SpliceAI2](https://proto-bio.github.io/proto-assets/images/tool/spliceai2/hero.png)

> [!NOTE]
> **License:** SpliceAI2 is licensed under Custom (SpliceAI2 Model Terms of Use) and has restrictions around commercial use and may require explicit attribution when utilized. Model weights are gated and require accepting the provider's terms and authenticating with a HuggingFace token. Please refer to [the license](https://github.com/Illumina/SpliceAI2/blob/main/LICENSE) for full terms.
>
> SpliceAI2 (code, weights, and precomputed scores) may only be used for non-commercial purposes by, or on behalf of, non-commercial organizations such as universities, academic research institutions, and government entities. Hosting, managed services, redistribution, and third-party access are prohibited, and any publication using its output must cite the SpliceAI2 manuscript. For commercial licensing, contact Illumina at AI_licensing@illumina.com.

## Overview

[SpliceAI2](https://github.com/Illumina/SpliceAI2) is Illumina's successor to [SpliceAI](https://github.com/Illumina/SpliceAI): a deep learning model that predicts, from DNA sequence alone, how often each splice site is used, which donor/acceptor pairs are joined into junctions, and which full-length transcripts result. This toolkit wraps two tools: `spliceai2-score` for variant splice-effect scoring against a reference genome and `spliceai2-predict` for per-position splice usage, junction, and transcript prediction on any sequence.

## Background

[Pre-mRNA splicing](https://en.wikipedia.org/wiki/RNA_splicing) removes introns and joins exons at donor (5') and acceptor (3') splice sites, and [alternative splicing](https://en.wikipedia.org/wiki/Alternative_splicing) lets one gene produce many transcript isoforms. Variants that create, weaken, or destroy splice sites are a major cause of genetic disease, often deep in introns where exome sequencing does not look. Illumina's original SpliceAI ([Jaganathan et al., 2019](https://doi.org/10.1016/j.cell.2018.12.015)) became a standard tool for finding these variants, but it answers one question per position: is it a splice site? It does not say how strongly a site is used, which sites are joined, or what transcript results.

SpliceAI2 ([Jaganathan et al., 2026](https://assets.illumina.com/content/dam/illumina-marketing/images/genomics-research/articles/spliceai2/SpliceAI2.pdf)) is Illumina's successor, built to answer those questions quantitatively. From about 200 kb of genomic sequence it predicts the **usage** of each splice site, the usage of each **splice junction** (the intron joining a donor to an acceptor), and the complete **transcripts** these imply. Its training set is more than 100 times larger than SpliceAI's: 314,745 RNA-seq samples across human and nine other mammals (more than 46 million observed junctions after filtering), together with 330 ENCODE long-read samples that show which full transcripts cells actually make. A species channel tells the single network which genome it is reading.

Internally, a stack of dilated convolutional blocks reads the sequence at progressively wider scales to predict donor and acceptor usage across the central 65,536 bp. The 1,024 strongest candidate sites are then paired to predict junction usage, and these site and junction scores form a splice graph whose highest-scoring paths are the predicted transcripts. The model has about 13 million parameters, and the release averages two trained checkpoints.

To score a variant, SpliceAI2 compares predictions for the reference and alternate alleles and reports the ten largest gains and losses in donor, acceptor, and junction usage, plus a single `summary_score` in [0, 1]. In Illumina's benchmarks it outperformed SpliceAI, Pangolin, and AlphaGenome, most clearly on cryptic splice variants deep in introns (auPRC 0.77 versus 0.66 for SpliceAI on GTEx). Eight of its output fields map one-to-one onto SpliceAI's delta scores and positions (`DS_DG`, `DS_DL`, `DS_AG`, `DS_AL`, `DP_*`), so existing SpliceAI pipelines translate directly.

This toolkit runs the released multispecies ensemble; the tissue-specific and disease-state models described in the manuscript are not publicly released.

### Learning Resources

- [Introducing SpliceAI2](https://www.illumina.com/science/genomics-research/articles/introducing-spliceai2--the-next-generation-of-splicing-and-trans.html) (Illumina) - accessible overview of the three prediction levels, the training data, and benchmark results against SpliceAI, Pangolin, and AlphaGenome.
- [SpliceAI2 repository](https://github.com/Illumina/SpliceAI2) (Illumina) - source, the reference variant-scoring CLI, output column definitions, and the transcript-decoding recipe this wrapper mirrors.
- [SpliceAI2 manuscript](https://assets.illumina.com/content/dam/illumina-marketing/images/genomics-research/articles/spliceai2/SpliceAI2.pdf) (Jaganathan et al., 2026) - architecture, training setup, and variant-effect benchmarks.

## Tools

### SpliceAI2 Variant Scoring (`spliceai2-score`)

Scores genetic variants (chromosome, 1-based position, ref, alt, gene strand) for splice-altering effects, returning the ten strongest donor, acceptor, and junction gains and losses per variant and a headline `summary_score`. Reads each variant's sequence context from a reference genome.

#### Applications

Use this to triage candidate variants from a sequencing study for splicing impact, including deep-intronic variants that create new splice sites, or to rank variants of uncertain significance where a coding effect is absent. The TSV export uses the column layout of the upstream `.spliceai2` output, so existing downstream scripts can read it.

#### Usage Tips

- **`strand` is required for every variant.** It is the strand of the gene the variant is scored against. When it is unknown, score the variant on both strands as two separate entries and aggregate the predictions (as the upstream README recommends).
- **Filter on `summary_score` with the upstream thresholds.** 0.1 for high recall, 0.25 to balance precision and recall, 0.5 for high precision (the SpliceAI equivalents are 0.2 / 0.5 / 0.8). The summary score is the maximum of the strongest donor gain, donor loss, acceptor gain, and acceptor loss delta scores; junction effects are reported but excluded from it.
- **`reference_fasta` is required.** Pass `'grch38'` (or `'grch37'`; `'hg38'`/`'hg19'` also work), which is downloaded on first use, or a path to a local FASTA. A `chr` prefix on `chromosome` is added or dropped to match the FASTA (`'chr1'` and `'1'` both work), and `ref` must match the genome at `position`.
- **Keep `assembly='GRCh38'` for any human variant, including GRCh37 coordinates.** `assembly` selects the species channel the model is conditioned on, not the coordinate system; set it to another training assembly only for non-human sequence.
- **The scoring geometry is fixed.** Each allele is scored in a 196,608 bp window centered on the variant, changes are reported within ±32,768 bp, both alleles share 1,024 candidate splice sites, and predictions average the two-model ensemble. These are the values the model was trained and evaluated with, so they are not configurable. Positions beyond a chromosome end are padded with all-zero (unknown) input.

### SpliceAI2 Splice and Transcript Prediction (`spliceai2-predict`)

Predicts per-position donor and acceptor usage along one or more sense-strand DNA sequences, the junctions between candidate splice sites, and the highest-scoring decoded transcripts. No reference genome is needed.

#### Applications

Use this to map the splice-usage landscape of a gene or locus, to see which junctions and isoforms a sequence is predicted to produce, or to compare reference and edited versions of a sequence (predict each separately and diff the outputs). It also suits designed constructs, minigenes, and saturation scans where no genome coordinates exist.

#### Usage Tips

- **Embed short constructs in genomic context where possible.** Every input is padded with 65,536 bp of unknown (all-zero) context per side, so predictions cover every input position at any length. Isolated short sequences fall outside what the model was evaluated on (it always saw real flanking genome), so prefer including the surrounding genomic sequence and reading predictions over the region of interest.
- **Coordinates are 1-based and intron-anchored.** Donor usage is scored at the intron's first base and acceptor usage at its last base; each junction's `donor` and `acceptor` are those intron boundary positions. Transcript `exons` are 1-based inclusive intervals.
- **Transcript ends are not predicted.** The first and last exon of every decoded transcript extend to the start and end of the input sequence, because SpliceAI2 does not model transcription start or end sites. `num_transcripts` (default 3) sets how many top-scoring paths are decoded; duplicates are dropped, so fewer may be returned, and none are returned when fewer than two candidate sites clear 0.01 usage.
- **Junctions are reported at usage >= 0.01** between candidate sites (positions whose donor or acceptor usage exceeds 0.01), ordered by donor then acceptor.
- **`N` bases are encoded as `A`, matching upstream.** Ambiguous bases are read as adenine rather than as missing sequence, so avoid `N` runs in regions you care about; only the padding outside the input is all-zero.
- **Use `assembly='GRCh38'` for any human sequence.** It selects the species channel the model is conditioned on; choose another training assembly only for non-human sequence.
- **GPU memory grows with input length.** Peak memory is about 5 GiB plus 4 GiB per 100 kb of input (measured on an H100: 13 GiB at 190 kb, 44 GiB at 1 Mb, 63 GiB at 1.5 Mb), so inputs longer than about 1.8 Mb do not fit on an 80 GB GPU, and smaller GPUs reach their limit proportionally sooner. For a long locus, predict overlapping windows that each keep genomic flanking sequence.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to both SpliceAI2 tools in this toolkit (`spliceai2-score`, `spliceai2-predict`).

- **CUDA GPU required.** SpliceAI2 runs only on GPU (default `device="cuda"`). As upstream does, `spliceai2-score` runs under 16-bit mixed precision and `spliceai2-predict` in full precision.
- **Gated weights.** The two ensemble checkpoints are gated on HuggingFace. Accept the terms at [illumina-ai/SpliceAI2](https://huggingface.co/illumina-ai/SpliceAI2) and set `HF_TOKEN` (or run `hf auth login`); the standalone environment then downloads both checkpoints automatically.
- **Run it yourself.** The license prohibits hosting and third-party access, so run SpliceAI2 locally on your own GPU, or deploy it on your own Modal account and use `device='modal'`.
- **Scope of the release.** This toolkit runs the released multispecies ensemble. Illumina's tissue-specific fine-tuned models are not publicly released, and the precomputed genome-wide scores (HuggingFace dataset [illumina-ai/SpliceAI2-data](https://huggingface.co/datasets/illumina-ai/SpliceAI2-data)) are not part of this toolkit.
