<a href="https://bio-pro.mintlify.app/tools/structure-alignment/foldmason"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# FoldMason

![FoldMason](https://proto-bio.github.io/proto-assets/images/tool/foldmason/hero.png)

> [!NOTE]
> **License:** FoldMason has a GPL-3.0 license. Please refer to [the license](https://github.com/steineggerlab/foldmason/blob/master/LICENSE.md) for full terms.

## Overview

[FoldMason](https://github.com/steineggerlab/foldmason) is a multiple protein-structure alignment tool from the [Steinegger Lab](https://steineggerlab.com/) at Seoul National University. It produces a structural multiple-sequence alignment over an arbitrary set of [PDB](https://www.rcsb.org/) inputs and returns the alignment in both the amino-acid alphabet and the [3Di structural alphabet](https://www.nature.com/articles/s41587-023-01773-0) shared with [Foldseek](https://bio-pro.mintlify.app/tools/structure-alignment/foldseek). A separate tool in the same toolkit scores an existing MSA against its structures using a per-column [LDDT](https://doi.org/10.1093/bioinformatics/btt473) metric.

## Background

[FoldMason](https://github.com/steineggerlab/foldmason) ([Gilchrist, Mirdita & Steinegger, 2026](https://doi.org/10.1126/science.ads6733)) is a progressive multiple-structure alignment method that scales to hundreds of thousands of protein structures. Each input structure is first encoded as a string over the [3Di alphabet](https://www.nature.com/articles/s41587-023-01773-0), the structural alphabet introduced with Foldseek that represents the local backbone geometry of each residue as a discrete letter. FoldMason then aligns the 3Di strings alongside their amino-acid sequences through a progressive procedure that follows a structural guide tree, using [Foldseek](https://bio-pro.mintlify.app/tools/structure-alignment/foldseek) and [TM-align](https://bio-pro.mintlify.app/tools/structure-alignment/tmalign) as the pairwise structural aligners at each merge step. An optional iterative refinement procedure can re-align the result to maximise its LDDT score. The output is a column-by-column alignment expressed in both alphabets together with the Newick guide tree.

Alignment quality is summarised with the [Local Distance Difference Test (lDDT)](https://academic.oup.com/bioinformatics/article-abstract/29/21/2722/195896) ([Mariani et al., 2013](https://doi.org/10.1093/bioinformatics/btt473)), a superposition-free metric that scores local atomic-distance agreement between two structures. FoldMason's `msa2lddt` computes LDDT on each pairwise sub-alignment, maps the per-residue scores back to MSA columns, and averages across pairs to produce one column-wise score and one overall average. The reference implementation is released as open source by the [Steinegger Lab](https://steineggerlab.com/) at [steineggerlab/foldmason](https://github.com/steineggerlab/foldmason). The same group operates a public web service at [search.foldseek.com/foldmason](https://search.foldseek.com/foldmason) that the remote execution mode of this toolkit targets.

### Learning Resources

- [steineggerlab/foldmason](https://github.com/steineggerlab/foldmason) (Steinegger Lab, Seoul National University) - official repository, command-line interface for `easy-msa`, `structuremsa`, `refinemsa`, and `msa2lddt`, and the FASTA output format that this toolkit parses.
- [search.foldseek.com/foldmason](https://search.foldseek.com/foldmason) (Steinegger Lab) - the public web service that the remote execution mode targets, useful for a single browser-based alignment before scripting against the tool.

## Tools

### FoldMason MSA (`foldmason-msa`)

Aligns two or more PDB structures and returns the amino-acid and 3Di MSAs as FASTA strings together with the Newick guide tree, the alignment length, and the number of sequences aligned. The tool executes against the public Steinegger Lab web service in `remote` mode and against the bundled `foldmason easy-msa` program in `local` mode.

#### Applications

This tool is appropriate for aligning a fold family retrieved from a Foldseek search, for comparing designed scaffolds against their target backbone, or for assembling a multi-structure template ensemble for downstream template-based modelling. It also applies to AlphaFold predictions across an evolutionary set, where the alignment can identify residues that are structurally conserved as well as loops that vary in conformation.

#### Usage Tips

- **`foldmason-msa` supports both remote (`search_mode="remote"`, the default) and local (`search_mode="local"`) execution.** Remote mode targets the Steinegger Lab web service. Local mode runs the bundled FoldMason program and accepts the full set of alignment parameters.
- **The Steinegger Lab web service does not accept alignment parameters.** The configuration fields `gap_open`, `gap_extend`, `refine_iters`, `precluster`, and `guide_tree_newick` therefore require `search_mode="local"`.
- **`refine_iters` controls how many iterative LDDT-maximising refinement passes run after the initial progressive alignment.** Each pass adds runtime, and the default of `0` is appropriate for most workflows. Increase it only when an alignment shows poor quality in difficult regions.
- **The remote service has no authentication and no published rate limit.** `search.foldseek.com/foldmason` is a free public academic resource. High-throughput or batch workloads should be performed in `local` mode to avoid overloading the shared service.

### FoldMason Score MSA (`foldmason-score-msa`)

Accepts a precomputed amino-acid MSA in FASTA format together with the underlying PDB structures, and returns the average MSA-wide LDDT score, the per-column LDDT scores, the number of columns considered, and the total alignment length.

#### Applications

This tool is appropriate for assigning a structural quality score to an MSA produced elsewhere, for identifying low-LDDT columns that should be masked or treated as variable loops before downstream analysis, or for comparing two candidate alignments of the same structures using a single summary score.

#### Usage Tips

- **FASTA record headers must match `structure_ids`.** `msa2lddt` resolves each MSA row to its corresponding structure by matching the header against the supplied identifiers. Headers that do not correspond to a supplied structure are not scored, which can produce a misleadingly high score derived from a partial alignment.
- **`only_scoring_cols=True` normalises the average LDDT by the number of scored columns rather than by the total alignment length.** Use this option when comparing alignments with different gap content. Leaving it `False` (the default) includes gap columns in the denominator.
- **This tool runs only in local mode.** The public web service does not provide an `msa2lddt` endpoint, so every `foldmason-score-msa` call requires the local FoldMason program.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every FoldMason tool in this toolkit (`foldmason-msa`, `foldmason-score-msa`).

- **FoldMason runs on CPU only.** Neither the remote service nor the local program uses a GPU. Local-mode runtime grows with both the number of structures and their lengths, since each progressive merge step performs a pairwise structural alignment.
- **Inputs are normalised to PDB before alignment.** Each `structures` entry may be a `Structure`, a file path, or raw PDB/CIF text; every entry is serialised to PDB and written to disk as `{structure_id}.pdb` before FoldMason runs.
