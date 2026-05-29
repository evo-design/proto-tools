<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/segmasker"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Segmasker

> [!NOTE]
> **License:** Segmasker is licensed under Custom (NCBI BLAST+ public domain). Please refer to [the license](https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/scripts/projects/blast/LICENSE) for full terms.

## Overview

Segmasker measures the low-complexity content of protein sequences using the SEG algorithm. [Low-complexity regions](https://en.wikipedia.org/wiki/Low_complexity_regions_in_proteins) are stretches of biased amino acid composition, such as homopolymeric runs or short-period repeats, that can produce spurious matches during sequence comparison. For each input sequence, segmasker identifies the residues that fall within low-complexity regions and reports their count, their fraction of the sequence, and the sequence length, giving a quantitative measure of compositional bias.

## Background

Most natural [proteins](https://en.wikipedia.org/wiki/Protein) contain regions whose [amino acid](https://en.wikipedia.org/wiki/Amino_acid) composition is strongly biased, including homopolymeric runs, short-period repeats, and segments dominated by a few residue types. These [low-complexity regions](https://en.wikipedia.org/wiki/Low_complexity_regions_in_proteins) are biologically real but cause difficulty in [sequence alignment](https://en.wikipedia.org/wiki/Sequence_alignment), because their similarity is driven by shared composition rather than by common ancestry, which inflates the apparent significance of matches between unrelated sequences.

The SEG algorithm ([Wootton and Federhen, 1993](https://doi.org/10.1016/0097-8485(93)85006-x)) quantifies local compositional complexity along a protein sequence using a sliding window and partitions the sequence into segments of low and high complexity. Masking or down-weighting the low-complexity segments before a similarity search improves the specificity of the results. Segmasker is the SEG implementation distributed as a command-line program within the NCBI [BLAST+](https://en.wikipedia.org/wiki/BLAST_(biotechnology)) suite ([Camacho et al., 2009](https://doi.org/10.1186/1471-2105-10-421)), which reorganized the original BLAST applications into modular command-line tools. Within that suite, segmasker applies the SEG procedure to protein sequences and reports the low-complexity regions it identifies, which can then be excluded from similarity searches or used to flag compositionally biased designs.

### Learning Resources

- [NCBI BLAST+ Command Line Applications User Manual](https://www.ncbi.nlm.nih.gov/books/NBK279690/) - the reference manual for the BLAST+ suite that segmasker ships with, including its masking applications.
- [BLAST Help (NCBI)](https://blast.ncbi.nlm.nih.gov/doc/blast-help/) - NCBI's documentation hub for BLAST concepts, including low-complexity filtering.

## Tools

### Segmasker Low-Complexity Detection (`segmasker-score`)

Applies the SEG algorithm to one or more protein sequences and returns, for each sequence, the number of residues classified as low-complexity, the fraction of the sequence those residues represent, and the sequence length. The low-complexity fraction is the primary metric for ranking sequences by compositional bias.

#### Applications

- Screening designed protein sequences for compositional bias before further analysis.
- Quantifying low-complexity content to flag homopolymeric runs or short-period repeats.
- Prioritizing sequences for masking ahead of a protein similarity search to reduce spurious matches.

#### Usage Tips

- **`window` sets the scale of the regions detected.** A larger window targets broader low-complexity stretches, while a smaller window resolves shorter runs.
- **`locut` and `hicut` set how aggressively regions are flagged.** Raising the cutoffs classifies more of the sequence as low-complexity, while lowering them applies a stricter criterion that flags only the most biased regions. `hicut` must be greater than or equal to `locut`.
- **Very short and empty sequences are limited.** A sequence shorter than the window cannot be assessed reliably, and an empty sequence reports a low-complexity fraction of zero.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **Detection runs on CPU and is deterministic.** Segmasker takes only protein sequences, runs without a GPU, and returns the same values for identical inputs on repeated calls.
- **Results are index-aligned with the input.** Each result corresponds to the input sequence at the same position, so a batch of sequences returns metrics in the order they were supplied.
