<a href="https://bio-pro.mintlify.app/tools/structure-scoring/dssp"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# DSSP

> [!NOTE]
> **License:** DSSP is open source and free for academic and commercial use under a BSD-2-Clause license. Please refer to [the license](https://github.com/PDB-REDO/dssp/blob/trunk/LICENSE) for full terms.

## Overview

[DSSP](https://github.com/PDB-REDO/dssp) (Dictionary of Protein Secondary Structure) is a standard program for assigning protein secondary structure from atomic coordinates. It identifies hydrogen-bonding and geometric patterns in a protein backbone and labels each residue as helix, strand, turn, or another defined state. This toolkit runs DSSP and collapses those assignments into helix, sheet, and loop percentages for a selected chain in each input structure.

## Background

DSSP ([Kabsch and Sander, 1983](https://doi.org/10.1002/bip.360221211)) is an assignment program that classifies each residue of a protein into a secondary-structure state by inspecting the geometry and the hydrogen-bond pattern of the protein backbone. Recurring turns are assigned as helices (states `H`, `G`, and `I` for alpha, 3-10, and pi helices respectively), recurring bridges between residues form ladders that are assigned as strand (`E`), and isolated bridges, turns, bends, and unassigned residues form the remaining states. DSSP works only from atomic coordinates and does not predict secondary structure from sequence alone. The modern implementation ([Hekkelman et al., 2025](https://doi.org/10.1002/pro.70208)) is maintained by the [PDB-REDO project](https://github.com/PDB-REDO) at the Netherlands Cancer Institute and ships as the `mkdssp` command-line program with extended mmCIF support and FAIR annotation.

This toolkit collapses the per-residue DSSP states into a coarse three-class summary for the chain of interest. Helix percentage counts residues assigned `H`, `G`, or `I`. Sheet percentage counts residues assigned `E`. Loop percentage counts every other DSSP state (`B`, `T`, `S`, the unassigned state, and the `P` polyproline-II state introduced in DSSP 4). The three percentages sum to 100 for the counted residues of the selected chain.

### Learning Resources

- [PDB-REDO/dssp](https://github.com/PDB-REDO/dssp) (PDB-REDO project, Netherlands Cancer Institute). Official repository and the source of the `mkdssp` command-line program that this toolkit invokes.
- [`mkdssp` command-line reference](https://github.com/PDB-REDO/dssp/blob/trunk/doc/mkdssp.md) (PDB-REDO project). Reference documentation for the DSSP state alphabet and the command-line interface of the program that this toolkit invokes.

## Tools

### DSSP Secondary Structure (`dssp-secondary-structure`)

Assigns secondary structure with the `mkdssp` program for a selected chain in each input structure and returns the resulting helix, sheet, and loop percentages. Inputs are supplied as one or more `DSSPStructureInput` objects, each carrying a `Structure` (or a path / coordinate string accepted by `Structure`) plus the chain identifier to analyse. Multiple structures in one call are processed independently and the results are returned in input order.

#### Applications

This tool is appropriate for filtering designed proteins by their secondary-structure composition, for summarising the helical or beta-sheet content of a predicted structure ensemble, and for any analysis that requires a DSSP-backed secondary-structure assignment as an upstream step in a larger pipeline. The collapsed three-class summary is the right shape for ranking or filtering large structure batches by composition.

#### Usage Tips

- **Select the chain to analyze, or leave `chain` empty to analyze the first chain.** `chain` is a `SingleChainSelection` (e.g. `"A"`); each input structure yields one result row for its selected chain. The input validator hard-errors when the selected chain is not present and lists the available chains. Omitting `chain` analyzes the first chain in the structure.
- **Structures with more than 62 chains are rejected.** The DSSP standalone runs on PDB-format text, which represents a chain identifier as a single character from `A-Z`, `a-z`, or `0-9`. Structures exceeding this limit cannot be dispatched through the wrapper.
- **Helix percentage counts DSSP states `H`, `G`, and `I`.** Sheet percentage counts state `E`. Loop percentage counts every remaining state, including `B`, `T`, `S`, the unassigned state, and the `P` polyproline-II state introduced in DSSP 4. The three percentages sum to 100 for the counted residues of the selected chain.
- **The first model is used for multi-model structures.** Only the first model parsed by Biopython contributes to the residue counts. To analyse a specific model in an NMR ensemble or a multi-state file, extract that model into its own `Structure` before passing it in.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every DSSP tool in this toolkit (`dssp-secondary-structure`).

- **Structure inputs accept either typed `Structure` objects or a path / coordinate string.** A field validator normalises raw paths and `Structure`-coercible values into `Structure` instances at input time. The wrapper writes each parsed structure to a temporary PDB file for the DSSP program and removes it after the call.
- **Outputs are returned as typed metric objects.** Each `DSSPSecondaryStructureMetrics` result carries the analysed chain identifier and the three secondary-structure percentages, with `helix_pct`, `sheet_pct`, and `loop_pct` constrained to the range 0 to 100. Results serialise to CSV or JSON through the standard export interface.
