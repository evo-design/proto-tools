<a href="https://bio-pro.mintlify.app/tools/gene-annotation/minced"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# MinCED

![MinCED](https://proto-bio.github.io/proto-assets/images/tool/minced/hero.png)

> [!NOTE]
> **License:** MinCED has a GPL-3.0 license. Please refer to [the license](https://github.com/ctSkennerton/minced/blob/master/LICENSE) for full terms.

## Overview

[MinCED](https://github.com/ctSkennerton/minced) (Mining CRISPRs in Environmental Datasets) is a fast Java program that locates [CRISPR](https://en.wikipedia.org/wiki/CRISPR) arrays in nucleotide sequences from isolate genomes or metagenomic contigs. It returns each detected array as an ordered list of repeat-spacer units with their genomic coordinates, the repeat consensus, and per-spacer sequence.

## Background

MinCED is a derivative of the [CRISPR Recognition Tool (CRT)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1924867/) ([Bland et al., 2007](https://doi.org/10.1186/1471-2105-8-209)), maintained by [Connor Skennerton](https://github.com/ctSkennerton). CRISPR arrays are blocks of short, near-identical direct repeats (typically 23 to 47 nt) separated by unique [spacer](https://en.wikipedia.org/wiki/CRISPR#Spacer_acquisition) sequences (typically 26 to 50 nt) that record fragments of past viral and plasmid infections; they form the heritable memory of the [CRISPR-Cas adaptive immune system](https://en.wikipedia.org/wiki/CRISPR) of bacteria and archaea.

Internally, MinCED uses a [k-mer](https://en.wikipedia.org/wiki/K-mer) seed-and-extend strategy. It scans for short exact k-mer matches that recur at a consistent spacing, then extends each seed bidirectionally to the actual repeat length, and finally validates the candidate by checking that the inter-repeat spacers fall within the configured length window. The algorithm runs on raw DNA, has linear time complexity in sequence length, and finishes in seconds on a typical 5 Mb bacterial genome on commodity CPU hardware.

### Learning Resources

- [ctSkennerton/minced](https://github.com/ctSkennerton/minced) (Connor Skennerton) - official repository with the canonical command-line flag surface, installation instructions, and example output.
- [PMC1924867 (CRT paper)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1924867/) (Bland et al.) - the full text of the algorithm description, including the seed-and-extend mechanism and the comparison against PatScan and PILER-CR.

## Tools

### MinCED CRISPR Array Detection (`minced-crispr`)

Detects CRISPR arrays in one or more nucleotide sequences. Returns, per input sequence, a list of `CrisprArray` objects; each carries an ordered list of `CrisprRepeatSpacer` units with the repeat's start position, the repeat sequence, and the following spacer (the last unit has no spacer).

#### Applications

Use this to confirm and catalog CRISPR loci across newly sequenced bacterial and archaeal genomes, or to mine spacer libraries from metagenomic assemblies for phage-host interaction studies. As a pre-filter, run `minced-crispr` first to verify that a candidate contig actually carries a CRISPR array before spending compute on downstream Cas and tracrRNA analysis with [`pyhmmer-hmmsearch`](https://bio-pro.mintlify.app/tools/gene-annotation/pyhmmer) for Cas effector domains and [`crispr-tracr-rna`](https://bio-pro.mintlify.app/tools/gene-annotation/crispr-tracr-rna) for tracrRNA on the same locus. The spacer set returned for each array can then be aligned against phage or plasmid sequence databases to reconstruct the host's immune history.

#### Usage Tips

- **`min_num_repeats` controls the sensitivity-versus-specificity trade-off.** The default of 3 balances both for typical bacterial and archaeal genomes. Lower it to 2 to catch partial or degraded arrays at the cost of more false positives, and raise it to 4 or more when only high-confidence arrays should pass through.
- **The 23 to 47 nt repeat and 26 to 50 nt spacer windows match canonical CRISPR loci.** Widen `max_repeat_length` and `max_spacer_length` to detect atypical families such as Type IV-A or CRISPR systems with unusually long spacers, and lower `min_repeat_length` only when chasing partial repeats since values below 23 nt start to pick up generic tandem repeats.
- **MinCED only locates the array; it does not identify Cas genes or classify the CRISPR system.** Type assignment requires downstream Cas-effector annotation, typically `pyhmmer-hmmsearch` against curated Cas HMMs or a dedicated classifier such as CRISPRcasIdentifier.
- **Inverted length ranges are caught at config time.** Setting `max_repeat_length < min_repeat_length` or `max_spacer_length < min_spacer_length` raises `ValueError` before the run starts, so the call fails fast instead of completing with an empty result set.
- **Spacer count is not an immunity-breadth metric.** Multiple spacers in an array can target the same phage, and many spacers are degraded remnants of historical encounters, so the number of spacers overestimates how many distinct threats the host can recognize today.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every MinCED tool in this toolkit (`minced-crispr`).

- **Runs on CPU only.** MinCED is a Java program; the standalone install bundles a Java runtime alongside the `minced` program. There is no GPU acceleration to enable, and runtime is seconds per typical bacterial genome.
- **Self-contained after install.** The standalone `setup.sh` downloads the `minced` program once; subsequent runs need no network access and no model weights or reference databases.
- **Sequences are processed one at a time.** The wrapper iterates over `inputs.sequences` sequentially rather than parallelizing across them. For large batches, run independent calls in parallel from the caller side.
