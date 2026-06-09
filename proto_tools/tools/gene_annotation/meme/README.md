<a href="https://bio-pro.mintlify.app/tools/gene-annotation/meme"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# MEME Suite (FIMO)

![MEME Suite (FIMO)](https://proto-bio.github.io/proto-assets/images/tool/meme/hero.png)

> [!NOTE]
> **License:** MEME Suite (FIMO) is licensed under Custom (MEME Suite Academic License) and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to [the license](https://github.com/althonos/pymemesuite/blob/main/vendor/meme/COPYING) for full terms.

## Overview

[FIMO](https://meme-suite.org/meme/doc/fimo.html) (Find Individual Motif Occurrences) is a tool from the [MEME Suite](https://meme-suite.org/) that scans DNA or protein sequences for occurrences of known motifs described as position weight matrices. This toolkit exposes FIMO through [pymemesuite](https://github.com/althonos/pymemesuite), a Cython binding to the MEME Suite C library, so scans run entirely in-process with no separate MEME installation required.

## Background

[FIMO](https://meme-suite.org/meme/doc/fimo.html) ([Grant, Bailey & Noble, 2011](https://doi.org/10.1093/bioinformatics/btr064)) treats a motif as a [position weight matrix](https://en.wikipedia.org/wiki/Position_weight_matrix) (PWM) and slides it across every position of each target sequence. At each position it computes a log-odds [score](https://en.wikipedia.org/wiki/Log_odds) of the windowed subsequence under the motif model versus a background nucleotide (or amino-acid) distribution, then converts that score to a [p-value](https://en.wikipedia.org/wiki/P-value) using the exact null distribution of scores for the motif. Because many positions across many sequences and motifs are tested, FIMO also reports a Benjamini-Hochberg [q-value](https://en.wikipedia.org/wiki/False_discovery_rate) so that hits can be filtered at a controlled [false discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate).

Motifs are supplied in [MEME format](https://meme-suite.org/meme/doc/meme-format.html), the text PWM format shared across the MEME Suite, and large curated collections such as the [JASPAR](https://jaspar.elixir.no/) transcription-factor database publish their matrices in this format directly. For nucleotide motifs FIMO scans both the given strand and its reverse complement by default, since a regulatory motif may occur on either DNA strand; protein and strand-specific scans disable the reverse strand. Coordinates of each match are reported as 1-indexed, inclusive intervals to match biological residue selection conventions. [pymemesuite](https://github.com/althonos/pymemesuite) preserves the FIMO scoring algorithm exactly while returning structured per-match results in Python.

### Learning Resources

- [FIMO documentation](https://meme-suite.org/meme/doc/fimo.html) (The MEME Suite) - the canonical reference for FIMO's inputs, p-value/q-value statistics, threshold options, and output columns.
- [MEME motif format guide](https://meme-suite.org/meme/doc/meme-format.html) (The MEME Suite) - describes the text PWM format FIMO consumes, including how to convert matrices from other databases.
- [JASPAR](https://jaspar.elixir.no/) (JASPAR Consortium) - the standard open-access database of curated transcription-factor binding profiles, downloadable as MEME-format motifs ready to feed to FIMO.

## Tools

### MEME FIMO Motif Scan (`meme-fimo-scan`)

Scans one or more target sequences against every position weight matrix in a MEME-format motif file and returns each occurrence with its motif id, 1-indexed coordinates, strand, log-odds score, p-value, and q-value.

#### Applications

Use this when the question is "where does motif X occur in these sequences." Typical workflows include locating transcription-factor binding sites in promoters or enhancers, screening a designed regulatory library for unwanted or intended motif occurrences, and annotating candidate sites for downstream filtering on score or q-value.

#### Usage Tips

- **`threshold` is the p-value cutoff and is the main sensitivity knob.** It defaults to `1e-4`, reproducing FIMO's command-line default (`--thresh`); only matches with a p-value at or below this value are reported. Loosen it (e.g. `1e-3`) to recover weaker sites at the cost of more false positives, or tighten it for stringent calls. The reported q-value gives the false discovery rate for filtering after the scan.
- **`both_strands` controls strand coverage for nucleotide motifs.** It defaults to `True`, scanning the forward strand and its reverse complement (the right choice for DNA/RNA motifs, which can bind on either strand). Set it to `False` for single-strand scans (this maps to FIMO's `--norc`). For protein and other non-complementable motifs the reverse complement is meaningless, so it is ignored automatically and scanning is always forward-only — matching the FIMO CLI.
- **Motifs come from MEME-format files, such as those exported from JASPAR.** Supply a `.meme` PWM file; matrices from JASPAR and other databases can be converted to this format. The scan iterates over every motif in the file against every target sequence.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to the MEME FIMO tool in this toolkit (`meme-fimo-scan`).

- **Runs on CPU.** FIMO scanning is a CPU operation; `pymemesuite` compiles the MEME Suite C library into its wheel, so there is no GPU acceleration to enable and no separate MEME install or PATH lookup is needed.
- **Motifs are user-supplied.** FIMO ships no motif database; provide your own MEME-format PWM file (e.g. exported from [JASPAR](https://jaspar.elixir.no/)) via `motifs`. Every motif in the file is scanned against every target sequence.
- **Results are returned per input sequence.** `results[i]` holds the matches found in input sequence `i`, positionally aligned to the input — a sequence with no occurrences yields an empty bundle. Scanning is deterministic — identical inputs return identical matches on repeated calls.
