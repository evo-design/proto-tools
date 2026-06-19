<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/mafft"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# MAFFT

![MAFFT](https://proto-bio.github.io/proto-assets/images/tool/mafft/hero.png)

> [!NOTE]
> **License:** MAFFT is open source and free for academic and commercial use under a BSD-3-Clause license. Please refer to [the license](https://mafft.cbrc.jp/alignment/software/license.txt) for full terms.

## Overview

[MAFFT](https://mafft.cbrc.jp/alignment/software/) (Multiple Alignment using Fast Fourier Transform) is a multiple sequence alignment program developed by Kazutaka Katoh and collaborators at Osaka University. It aligns multiple protein or nucleotide sequences by inserting gap characters so that homologous residues occupy the same alignment column, and offers a family of algorithms that trade speed for accuracy. This toolkit runs the MAFFT command-line program and returns typed MSA results.

## Background

MAFFT ([Katoh and Standley, 2013](https://doi.org/10.1093/molbev/mst010)) is a multiple sequence alignment program that constructs an alignment through progressive alignment along a guide tree followed by optional iterative refinement. Pairwise distances between input sequences are first estimated rapidly using either k-mer counting or a Fast Fourier Transform that detects homologous segments in compositionally transformed sequences. A guide tree is built from these distances, sequences are progressively aligned along the tree, and the alignment is optionally refined by an iterative cycle that repeatedly removes and re-aligns subsets of sequences.

MAFFT exposes several algorithm variants that differ in pairwise scoring and refinement strategy. `FFT-NS-i` is the default progressive method with iterative refinement on FFT-derived distances and is appropriate for large datasets. `L-INS-i` (`localpair`) performs local pairwise alignment with iterative refinement and is appropriate for sequences with one alignable domain flanked by variable regions. `G-INS-i` (`globalpair`) performs global pairwise alignment with iterative refinement and is appropriate for sequences of similar length. `E-INS-i` (`genafpair`) is a local-alignment variant that handles sequences with multiple conserved domains separated by long unalignable regions.

### Learning Resources

- [MAFFT software homepage](https://mafft.cbrc.jp/alignment/software/) (Osaka University). Official distribution site and user documentation for the command-line program that this toolkit invokes.
- [MAFFT algorithm comparison](https://mafft.cbrc.jp/alignment/software/algorithms/algorithms.html) (Osaka University). A side-by-side comparison of the alignment algorithm variants that the `align_method` field selects.
- [MAFFT online server](https://mafft.cbrc.jp/alignment/server/) (Osaka University). Hosted entry point to the same MAFFT pipeline, useful for a quick browser-based alignment before scripting against the tool.

## Tools

### MAFFT Alignment (`mafft-align`)

Performs multiple sequence alignment over two or more input sequences using the bundled `mafft` command-line program. The selected algorithm variant is controlled by the `align_method` configuration field. The tool returns a typed `MSA` object containing the aligned sequences and their identifiers, with helpers for column statistics and serialisation to FASTA or A3M.

#### Applications

This tool is appropriate for any analysis that benefits from a multiple sequence alignment of homologous protein or nucleotide sequences. Common downstream uses include phylogenetic-tree inference, conservation analysis over alignment columns to identify functionally important residues, homology modelling against a related reference, motif and domain discovery across a protein family, and variant-effect analysis in the context of the conserved structural and functional positions revealed by the alignment.

#### Usage Tips

- **`align_method="auto"` is the default and lets MAFFT select an algorithm based on input size.** Use `localpair` for sequences with a single conserved domain flanked by variable regions, `globalpair` for full-length homologs of similar length, and `genafpair` for multi-domain sequences separated by long unalignable regions. The `*pair` variants run in O(N^2) time and are appropriate for up to a few hundred sequences.
- **`max_iterations=0` (the default) skips iterative refinement.** Raise it to enable the full `*-INS-i` refinement pipeline when paired with one of the `*pair` methods. A value around `1000` is appropriate for high-accuracy alignments of small to medium datasets.
- **`threads=1` is the default; raise it on large alignments.** MAFFT parallelises both the all-against-all distance computation and the iterative refinement passes, so increasing the thread count yields substantial wall-time reductions on alignments of hundreds of sequences or longer.
- **Inputs must contain at least two non-empty sequences.** The input validator hard-errors otherwise. Auto-generated identifiers default to `seq_0`, `seq_1`, and so on when `sequence_ids` is omitted.
- **`extra_args` accepts verbatim `mafft` CLI tokens.** Pass any CLI flag not exposed as a typed field through this list (for example `["--retree", "3", "--reorder"]` to control the guide-tree rebuild schedule). Tokens are inserted before the input FASTA path and take precedence over MAFFT's own defaults.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every MAFFT tool in this toolkit (`mafft-align`).

- **Outputs are returned as typed `MSA` objects.** The `msa` field of `MafftOutput` exposes the aligned sequences, their identifiers, alignment dimensions, column-level conservation statistics, and gap-statistics properties. The result serialises to FASTA or A3M through the standard export interface.
