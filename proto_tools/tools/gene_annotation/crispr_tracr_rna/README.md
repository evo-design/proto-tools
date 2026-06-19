<a href="https://bio-pro.mintlify.app/tools/gene-annotation/crispr-tracr-rna"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# CRISPRtracrRNA

![CRISPRtracrRNA](https://proto-bio.github.io/proto-assets/images/tool/crispr_tracr_rna/hero.png)

> [!NOTE]
> **License:** CRISPRtracrRNA's own code is licensed under MIT, and it federates over bundled data sources and components, each under its own license terms.
>
> Bundled dependencies, each under its own license:
>
> - [CRISPRcasIdentifier](https://github.com/BackofenLab/CRISPRcasIdentifier/blob/master/LICENSE.txt): GPL-3.0
>
> Review each source's terms before commercial use or redistribution.

## Overview

[CRISPRtracrRNA](https://github.com/BackofenLab/CRISPRtracrRNA) is a multi-evidence pipeline from the [Bioinformatics Group at the University of Freiburg](https://www.bioinf.uni-freiburg.de/) that detects [tracrRNA](https://en.wikipedia.org/wiki/Trans-activating_crRNA) candidates in nucleotide [CRISPR](https://en.wikipedia.org/wiki/CRISPR) loci. It combines covariance-model search, CRISPR array detection, Cas-effector cassette detection, anti-repeat similarity, RNA-RNA interaction prediction, and transcription-terminator detection into a single weighted ranking score per candidate.

## Background

[CRISPRtracrRNA](https://github.com/BackofenLab/CRISPRtracrRNA) ([Mitrofanov et al., 2022](https://doi.org/10.1093/bioinformatics/btac466)) detects trans-activating CRISPR RNA (tracrRNA) sequences in nucleotide CRISPR loci. A tracrRNA is a small non-coding RNA that base-pairs with the precursor crRNA in the [Class 2](https://en.wikipedia.org/wiki/CRISPR) effector systems that depend on one, namely [Type II](https://en.wikipedia.org/wiki/Cas9) (Cas9) and the tracrRNA-bearing [Type V](https://en.wikipedia.org/wiki/Cas12a) subtypes such as Cas12b, Cas12c, and Cas12e, and the resulting RNA duplex licenses Cas-mediated cleavage of target DNA. It is also the second component fused into the [single-guide RNA](https://en.wikipedia.org/wiki/Guide_RNA) used in modern genome editing. Because tracrRNAs share little primary-sequence conservation across families, single-model approaches such as [Infernal](https://github.com/EddyRivasLab/infernal) covariance-model search alone miss divergent tracrRNAs in newly sequenced and metagenomic genomes.

Internally, the pipeline runs an array-detection step with [CRISPRidentify](https://github.com/BackofenLab/CRISPRidentify) (machine learning), a Cas-cassette step with [CRISPRcasIdentifier](https://github.com/BackofenLab/CRISPRcasIdentifier) (HMM and machine learning), a tracrRNA candidate scan with Infernal `cmsearch` against curated covariance models, an anti-repeat alignment step using fasta36, vmatch, Clustal Omega, and BLAST, an RNA-RNA interaction step with [IntaRNA](https://github.com/BackofenLab/IntaRNA), and a transcription-terminator step with erpin. A final ranking step combines the per-candidate features into a single weighted score, and a faster `model_run` mode performs only the covariance-model scan and skips the validation evidence and the ranking step.

### Learning Resources

- [BackofenLab/CRISPRtracrRNA](https://github.com/BackofenLab/CRISPRtracrRNA) (Bioinformatics Group Freiburg) - official repository with installation instructions, the canonical configuration surface, and the curated covariance models distributed with the tool.
- [EddyRivasLab/infernal](https://github.com/EddyRivasLab/infernal) (The Eddy/Rivas Laboratory, Harvard) - official repository and User's Guide for the covariance-model search engine and the `cmsearch` E-value statistics that score tracrRNA candidates.
- [BackofenLab/IntaRNA](https://github.com/BackofenLab/IntaRNA) (Bioinformatics Group Freiburg) - official repository for the RNA-RNA interaction predictor that scores the anti-repeat to repeat duplex.

## Tools

### CRISPRtracrRNA Prediction (`crispr-tracr-rna`)

Predicts tracrRNA candidates from one or more nucleotide sequences and returns, per input sequence, a list of `CrisprTracrRNAPrediction` rows sorted by ranking score. Each row carries the candidate position and sequence, CRISPR array context, anti-repeat similarity and coverage, predicted RNA-RNA interaction with the repeat, terminator location and score, distance to the nearest Cas-effector cassette, and a single weighted multi-evidence score.

#### Applications

Use this to confirm and characterize Type II and Type V CRISPR-Cas loci, since a detected tracrRNA is the component that completes a functional Class 2 locus and distinguishes a Cas9 or Cas12 system from an unaccompanied CRISPR array. Pair it with [`minced`](https://bio-pro.mintlify.app/tools/gene-annotation/minced) on a confirmed array to recover the crRNA spacers, then design a single-guide RNA by fusing a spacer-bearing crRNA with the detected tracrRNA scaffold for genome-editing experiments. Run it across metagenomes and uncultured genomes to discover novel Cas9 or Cas12 systems whose tracrRNAs are too divergent to be caught by covariance-model search alone.

#### Usage Tips

- **Provide each CRISPR locus with at least 5 kb of flanking sequence on either side.** The multi-evidence pipeline needs adjacent context to locate the Cas cassette and the downstream transcription terminator. Loci submitted as narrow windows lose those evidence channels and fall back to a covariance-model-only score.
- **`model_type` defaults to `"II"`, which only screens for Cas9 systems.** To also screen tracr-bearing Type V (Cas12b, Cas12c, Cas12e, ...) loci, set `model_type="all"` and `perform_type_v_anti_repeat_analysis=True`. The Type V path is off by default because it is slower and irrelevant when only Cas9 loci are of interest.
- **Type I and Type III CRISPR systems do not use a tracrRNA.** A `complete_run` on such a locus returns array context and Cas annotations but empty tracrRNA fields, with the ranking score reflecting only the partial evidence.
- **The ten `weight_*` ranking parameters interact.** Sweep them together against a held-out positive and negative set rather than tuning a single weight in isolation, and keep upstream's documented defaults when there is no specific objective to optimize for.
- **`run_type="model_run"` is the high-throughput pre-filter, not the final answer.** It runs only the Infernal `cmsearch` step and returns candidates with E-values but none of the array, interaction, or terminator evidence, so re-run promising candidates through `complete_run` before drawing conclusions.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every CRISPRtracrRNA tool in this toolkit (`crispr-tracr-rna`).

- **Runs on CPU only.** The pipeline drives Infernal, IntaRNA, fasta36, vmatch, Clustal Omega, BLAST, and erpin, all CPU-based programs. There is no GPU acceleration to enable.
- **Initial install pulls model archives from Google Drive.** `complete_run` mode requires the CRISPRcasIdentifier ML and HMM archives, which the standalone install fetches once. Google Drive rate-limits anonymous fetches, so on a failed install retry after a minute or follow the upstream README to place the two archives in the CRISPRcasIdentifier directory by hand. After install the runtime needs no further network access.
- **`num_workers` parallelizes across input sequences, not within a sequence.** Each worker runs the full pipeline in its own working directory to avoid file-name collisions between concurrent jobs. The default of 1 is single-process; set it explicitly when batch-scanning many loci. The wrapper caps the effective worker count at `len(sequences)`, so over-provisioning is safe.
