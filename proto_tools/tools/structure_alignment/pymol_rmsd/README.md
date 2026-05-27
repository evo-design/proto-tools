<a href="https://bio-pro.mintlify.app/tools/structure-alignment/pymol-rmsd"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# PyMOL RMSD

> [!NOTE]
> **License:** PyMOL RMSD is licensed under Custom (Open-Source PyMOL Copyright Notice) and may require explicit attribution when utilized. Please refer to [the license](https://github.com/schrodinger/pymol-open-source/blob/master/LICENSE) for full terms.

## Overview

[PyMOL](https://www.pymol.org/) is a molecular visualization system originally written by Warren L. DeLano and maintained as open source by [Schrödinger, LLC](https://www.schrodinger.com/). This toolkit uses Open-Source PyMOL as a headless structural-alignment backend through a single registered tool that aligns two structures and reports the post-alignment RMSD together with method-specific alignment statistics. Two alignment routines are exposed: the Combinatorial Extension `cealign` for distantly related folds and the sequence-aware `align` for closely related structures.

## Background

Root mean square deviation (RMSD) measures the average distance between corresponding atoms of two structures after optimal rigid-body superposition. It is the standard summary statistic for assessing how closely a model recapitulates a reference structure, comparing alternative poses of the same complex, and quantifying conformational change between two states of the same protein. Computing a meaningful RMSD requires a residue correspondence between the two structures, which the upstream alignment routine establishes before the superposition is performed.

Open-Source PyMOL exposes two distinct alignment routines that this toolkit invokes. The `cealign` command implements the Combinatorial Extension (CE) algorithm ([Shindyalov and Bourne, 1998](https://doi.org/10.1093/protein/11.9.739)), which builds a structural alignment from aligned fragment pairs based on local geometry rather than a global sequence alignment. CE was developed to detect remote structural similarity below the sequence-similarity twilight zone, where the underlying sequences share too little identity for a meaningful sequence-based alignment. The `align` command in contrast first computes a sequence alignment using the BLOSUM62 substitution matrix, performs a structural superposition over the aligned residues, and then iterates several cycles of outlier rejection to remove residues with poor structural agreement. This routine is appropriate when the two structures share substantial sequence identity and the goal is a residue-matched superposition that excludes locally divergent regions.

### Learning Resources

- [schrodinger/pymol-open-source](https://github.com/schrodinger/pymol-open-source) (Schrödinger, LLC). The official Open-Source PyMOL repository and the source of the `cealign` and `align` commands invoked by this toolkit.
- [PyMOL Wiki](https://pymolwiki.org/) (community-maintained). Reference documentation for the `align` and `cealign` commands and the broader PyMOL scripting interface.

## Tools

### PyMOL RMSD Alignment (`pymol-rmsd-alignment`)

Aligns two `Structure` inputs with Open-Source PyMOL and returns the post-alignment RMSD together with method-specific alignment statistics. The `method` configuration field selects between the CE-based `cealign` and the sequence-aware `align` routine.

#### Applications

This tool is appropriate for any analysis that needs a pairwise structural superposition of two proteins. Representative applications include scoring designed structures against a reference template, quantifying conformational drift across molecular dynamics snapshots, comparing predicted poses against experimental structures, and evaluating backbone or side-chain changes introduced by mutation.

#### Usage Tips

- **`method` selects the alignment routine and should match the expected sequence relationship between the two inputs.** The default `cealign` runs the Combinatorial Extension algorithm and is appropriate for proteins with low sequence similarity. `align` performs a sequence alignment followed by structural superposition and iterative outlier rejection, and is more appropriate when the two inputs share substantial sequence identity.
- **`cealign` and `align` populate different metric fields.** A `cealign` call returns `rmsd` and `aligned_length` (the length of the CE alignment in CA atoms, equivalent to aligned residues since `cealign` operates only on CA atoms). An `align` call returns `rmsd` (after refinement), `aligned_atoms`, `alignment_cycles`, `alignment_score`, `pre_refinement_rmsd`, `pre_refinement_aligned_atoms`, and `aligned_residues`. Metric fields that do not apply to the selected method are returned as `None`.
- **`target_selection` and `mobile_selection` accept arbitrary PyMOL selection strings.** The defaults select the full target and mobile objects. Pass a refined selection such as `"target and name CA"` or `"target and chain A"` to restrict the alignment to a specific residue subset, chain, or atom set. Selection syntax follows the standard PyMOL grammar documented on the [PyMOL Wiki](https://pymolwiki.org/index.php/Selection_Algebra).
- **A failed alignment returns `failure_rmsd` rather than raising an exception.** When PyMOL cannot align the two structures (for example, when the structures are too dissimilar for CE to converge), the call returns the configured `failure_rmsd` value (default `999.0`) and attaches the underlying error message to the result metadata as `alignment_error`. This sentinel-value approach lets calling code distinguish a failed alignment from a near-zero RMSD between two essentially identical structures.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PyMOL RMSD tool in this toolkit (`pymol-rmsd-alignment`).

- **Outputs are returned as typed metric objects.** Each `PyMOLRMSDMetrics` result carries the post-alignment `rmsd` together with the method-specific metrics described under Usage Tips. The headline `primary_metric` is `rmsd`, and results can be exported to JSON through the standard export method.
- **Inputs accept a `Structure` object, a file path, or raw PDB or mmCIF content.** Each input is normalised to a `Structure` before scoring, and the corresponding PDB text is passed to PyMOL through a temporary file.
