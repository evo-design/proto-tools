<a href="https://bio-pro.mintlify.app/tools/structure-alignment/usalign"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# USalign

![USalign](https://proto-bio.github.io/proto-assets/images/tool/usalign/hero.png)

> [!NOTE]
> **License:** USalign is licensed under Custom (Zhang Lab academic-use license) and may require explicit attribution when utilized. Please refer to [the license](https://github.com/pylelab/USalign/blob/master/LICENSE) for full terms.

## Overview

[USalign](https://github.com/pylelab/USalign) is a universal structure alignment program that extends the TMalign approach to multi-chain complexes and nucleic acid structures. It is developed jointly by the [Zhang Lab](https://zhanggroup.org/) and the [Pyle Lab](https://pylelab.org/) and unifies pairwise structure alignment of proteins, RNA, DNA, and mixed-molecule complexes under a single Template Modeling (TM-score) objective. This toolkit compiles USalign from the canonical [pylelab/USalign](https://github.com/pylelab/USalign) distribution and runs it through a single registered tool that returns both length-normalised TM-scores in oligomeric multi-chain mode.

## Background

USalign ([Zhang, Shine, Pyle, and Zhang, 2022](https://doi.org/10.1038/s41592-022-01585-1)) is a universal structure alignment platform that aligns monomer and complex structures of proteins, RNA, and DNA under a single Template Modeling (TM-score) objective. Multi-chain complexes are aligned jointly by combining residue-level structural alignment with a chain-to-chain mapping search, and nucleic acid residues are anchored on the C3' backbone atom in place of the protein Cα. The published benchmark reports consistent advantages over state-of-the-art methods in pairwise and multiple structure alignment across these molecular types, and demonstrates that heterogeneous oligomeric complexes such as protein-RNA assemblies can be aligned within the same framework.

The Template Modeling score (TM-score) ([Zhang and Skolnick, 2004](https://doi.org/10.1002/prot.20264)) is a length-independent measure of topological similarity between two structures. The score lies between 0 and 1, with 1 indicating identical structures. A subsequent statistical analysis ([Xu and Zhang, 2010](https://doi.org/10.1093/bioinformatics/btq066)) established that a TM-score above 0.5 is a strong probabilistic indicator of shared SCOP and CATH fold classification for single-domain proteins, while scores below 0.17 are indistinguishable from random pairs (the random-pair distribution is centred near a TM-score of 0.15). The same 0.5 threshold is the convention used in the literature for multi-chain complex alignment, although the underlying statistical study was performed on monomers.

### Learning Resources

- [pylelab/USalign](https://github.com/pylelab/USalign) (Pyle Lab, Yale University). The canonical distribution that bundles USalign together with TMalign, MMalign, and TMscore. This toolkit compiles the USalign program from this repository.
- [Zhang Lab US-align page](https://zhanggroup.org/US-align/) (Zhang Lab). Background documentation, command-line reference, and an online USalign web service maintained by the original developers.

## Tools

### USalign Structure Alignment (`usalign-alignment`)

Aligns two macromolecular structures with USalign and returns the Template Modeling score normalised by the length of each input structure. The tool takes a query and reference `Structure`, runs the compiled USalign program in multi-chain oligomeric mode (`-mm 1 -ter 1`), and reports `tm_score_structure_1` (normalised by the query length) and `tm_score_structure_2` (normalised by the reference length). It also returns a `superposition` transform (rotation + translation) that superposes the query onto the reference, so the two structures can be overlaid. Set the `include_superimposed_pdb` config option to also return a multi-model PDB of the overlay for download.

#### Applications

This tool is the appropriate choice for any pairwise structure comparison that may include multiple chains, nucleic acid components, or a mix of protein and nucleic acid. Representative applications include validating a predicted multimeric complex against an experimental reference, ranking designed binder-target poses by interface architecture, comparing predicted RNA tertiary structures against known folds, and assessing predicted protein-nucleic acid assemblies against experimentally determined complexes.

#### Usage Tips

- **The two TM-scores differ when the query and reference have different total lengths.** Each score is normalised by the length of the named structure (summed across all aligned chains), so the score normalised by the shorter structure is typically the larger of the two. Use the score normalised by the structure whose length matters for the comparison, typically the reference or target when ranking candidates against a fixed structure.
- **A TM-score above 0.5 indicates the structures share the same fold or complex architecture.** This threshold is statistically derived from a non-redundant analysis of the Protein Data Bank ([Xu and Zhang, 2010](https://doi.org/10.1093/bioinformatics/btq066)) and is the standard fold-similarity cutoff in the literature. The same interpretation applies to monomers, multimers, and nucleic acid structures.
- **For single-chain protein-only inputs, prefer the [TMalign](../tmalign) tool.** TMalign runs the original single-chain TM-score alignment algorithm and is faster for guaranteed single-chain protein inputs. USalign is the appropriate choice when either input may be multi-chain or may contain nucleic acid residues.
- **The tool always runs in multi-chain oligomeric mode (`-mm 1`) and aligns every chain of the first model in each input (`-ter 1`).** This is the recommended mode for complex structure comparison and is also valid for monomer inputs. Molecule type is auto-detected from the input residues, so the same call handles proteins, RNA, DNA, and mixed assemblies without explicit configuration.
- **Multi-chain inputs should carry distinct chain identifiers.** The chain-to-chain mapping algorithm uses the chain IDs from the input PDB to track the joint alignment, so duplicated or missing chain IDs in a multi-chain input can result in suboptimal chain pairings.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every USalign tool in this toolkit (`usalign-alignment`).

- **Outputs are returned as typed metric objects.** Each `USalignMetrics` result carries both `tm_score_structure_1` and `tm_score_structure_2`, and the output's `superposition` field carries the rotation/translation that overlays the query onto the reference (`None` if USalign emitted no parseable matrix). Results can be exported to JSON through the standard export method.
- **Inputs accept a `Structure` object, a file path, or raw PDB or mmCIF content.** Each input is normalised to a `Structure` before scoring, and the corresponding PDB text is passed to USalign through a temporary file.
- **USalign runs on CPU and is fast enough for batch comparison of large structure sets.** No GPU is used, and per-pair runtime scales with the combined length of the two structures and the number of chains.
