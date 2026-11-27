<a href="https://bio-pro.mintlify.app/tools/structure-alignment/tmalign"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# TMalign

![TMalign](https://proto-bio.github.io/proto-assets/images/tool/tmalign/hero.png)

> [!NOTE]
> **License:** TMalign is licensed under Custom (Zhang Lab academic-use license) and may require explicit attribution when utilized. Please refer to [the license](https://github.com/pylelab/USalign/blob/master/LICENSE) for full terms.

## Overview

[TMalign](https://zhanggroup.org/TM-align/) is a pairwise protein structure alignment program developed by the [Zhang Lab](https://zhanggroup.org/). It identifies the optimal structural superposition of two protein structures by directly maximising the Template Modeling score (TM-score), and reports the score normalised by the length of each input chain. This toolkit compiles TMalign from the canonical [pylelab/USalign](https://github.com/pylelab/USalign) distribution and runs it through a single registered tool that returns both length-normalised TM-scores.

## Background

The Template Modeling score (TM-score) ([Zhang and Skolnick, 2004](https://doi.org/10.1002/prot.20264)) is a length-independent measure of topological similarity between two protein structures. It scores each pair of corresponding residues with a distance-based weight that uses a protein-size-dependent normalisation, which eliminates the inherent length dependence of RMSD-style scores and lets the same TM-score value be compared across proteins of different sizes. The score ranges from 0 to 1, with 1 indicating identical structures.

TMalign ([Zhang and Skolnick, 2005](https://doi.org/10.1093/nar/gki524)) is a structure alignment algorithm that identifies the optimal pairwise structural superposition by combining a TM-score-based rotation matrix with dynamic programming. Three initial alignments are seeded from secondary-structure matching, gapless threading, and a hybrid scoring matrix, and the residue-to-residue correspondence is then iteratively refined by alternating rigid-body rotation with dynamic programming on the TM-score-weighted distance matrix until the alignment converges. Unlike alignment methods that optimise RMSD, TMalign directly optimises the TM-score, which decouples the alignment objective from chain length. The published benchmark reports that TMalign produces alignments with higher coverage and accuracy than CE, DALI, and SAL while running approximately four times faster than CE and twenty times faster than DALI on the same workload.

A subsequent statistical analysis of the TM-score ([Xu and Zhang, 2010](https://doi.org/10.1093/bioinformatics/btq066)) provides quantitative interpretation guidance. The authors compare TM-scores across all pairs in a non-redundant set of 6,684 single-domain protein structures and report that the score follows an extreme value distribution. They show that a TM-score above 0.5 is a strong probabilistic indicator of shared SCOP and CATH fold classification, while scores below 0.5 mostly indicate different folds.

### Learning Resources

- [pylelab/USalign](https://github.com/pylelab/USalign) (Pyle Lab, Yale University). The canonical distribution that bundles TMalign together with USalign, MMalign, and TMscore. This toolkit compiles the TMalign program from this repository.
- [Zhang Lab TMalign page](https://zhanggroup.org/TM-align/) (Zhang Lab). Background documentation and an online TMalign web service maintained by the original developers.

## Tools

### TMalign Structure Alignment (`tmalign-alignment`)

Aligns two protein structures with TMalign and returns the Template Modeling score normalised by the length of each input chain. The tool takes a query and reference `Structure`, runs the compiled TMalign program, and reports `tm_score_chain_1` (normalised by the query length) and `tm_score_chain_2` (normalised by the reference length). It also returns a `superposition` transform (rotation + translation) that superposes the query onto the reference, so the two structures can be overlaid. Set the `include_superimposed_pdb` config option to also return a multi-model PDB of the overlay for download.

#### Applications

This tool is the standard method for pairwise protein structure comparison. Representative applications include validating that a designed protein adopts the intended fold, ranking predicted structures by topological similarity to a reference, classifying experimentally determined structures into known folds, and detecting distant structural homology where sequence similarity is too low for sequence-based comparison.

#### Usage Tips

- **The two TM-scores differ when the query and reference have different lengths.** Each score is normalised by the length of the named chain, so the score normalised by the shorter chain is typically the larger of the two. Use the score normalised by the chain whose length matters for the comparison, typically the reference or target when ranking candidates against a fixed structure.
- **A TM-score above 0.5 indicates the structures share the same fold.** This threshold is statistically derived from a non-redundant analysis of the Protein Data Bank ([Xu and Zhang, 2010](https://doi.org/10.1093/bioinformatics/btq066)) and is the standard fold-similarity cutoff in the literature. Scores above 0.3 are significantly above random with a P-value below 0.001, while scores below 0.17 are indistinguishable from random pairs (the random-pair distribution is centred near a TM-score of 0.15).
- **TMalign is designed for monomeric protein chains.** Multi-chain assemblies are processed as a single chain and chain breaks are not preserved. For genuine multi-chain alignment use the [USalign](../usalign) tool in this category, which is built for protein complexes.
- **Very short inputs produce unreliable scores.** The TM-score `d0` length-normalisation factor is calibrated for chains of approximately 15 residues and above and saturates rapidly for shorter chains, so short-chain comparisons lose the standard topological interpretation. Restrict comparison to chains of meaningful length before drawing fold-level conclusions.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every TMalign tool in this toolkit (`tmalign-alignment`).

- **Outputs are returned as typed metric objects.** Each `TMalignMetrics` result carries both `tm_score_chain_1` and `tm_score_chain_2`, and the output's `superposition` field carries the rotation/translation that overlays the query onto the reference (`None` if TMalign emitted no parseable matrix). Results can be exported to JSON through the standard export method.
- **Inputs accept a `Structure` object, a file path, or raw PDB or mmCIF content.** Each input is normalised to a `Structure` before scoring, and the corresponding PDB text is passed to TMalign through a temporary file.
- **TMalign runs on CPU and is fast enough for all-against-all comparison of large structure sets.** No GPU is used, and per-pair runtime scales with the product of the two chain lengths.
