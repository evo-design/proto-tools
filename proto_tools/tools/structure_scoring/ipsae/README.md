<a href="https://bio-pro.mintlify.app/tools/structure-scoring/ipsae"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# IPSAE

![IPSAE](https://proto-bio.github.io/proto-assets/images/tool/ipsae/hero.png)

> [!NOTE]
> **License:** IPSAE is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/DunbrackLab/IPSAE/blob/main/LICENSE) for full terms.

## Overview

[IPSAE](https://github.com/DunbrackLab/IPSAE) is a scoring program for protein-protein interfaces in cofolded complexes from the [Dunbrack Lab](https://dunbrack.fccc.edu/lab/) at the Fox Chase Cancer Center. It takes a predicted complex along with its per-residue confidence scores and the predicted aligned error (PAE) matrix and reports five complementary interface-quality scores for a selected binder-target chain pair. This toolkit runs IPSAE through a single registered tool that returns the ipSAE score together with pDockQ, pDockQ2, the local interaction score (LIS), and interface pTM.

## Background

IPSAE ([Dunbrack, 2025](https://doi.org/10.1101/2025.02.10.637595)) is a single program that computes five interface-quality scores for a cofolded protein complex from the structure together with its per-residue pLDDT and pairwise PAE matrix. The primary metric, ipSAE, recomputes an interface pTM-style score from PAE alone while replacing the chain-length-based reference distance with an adaptive reference distance derived from the size of the well-predicted interface. The published benchmark reports that ipSAE separates true and false complexes more efficiently than AlphaFold's interface pTM, in particular for domain-domain and domain-peptide interactions inside larger constructs that contain disordered or accessory regions.

Alongside ipSAE the program computes four additional interface-quality scores that the literature has converged on for cofolded complex assessment. The pDockQ2 score ([Zhu et al., 2023](https://doi.org/10.1093/bioinformatics/btad424)) estimates the quality of each interface in a multimer from interface pLDDT and PAE-derived signal. The pDockQ score ([Bryant et al., 2022](https://doi.org/10.1038/s41467-022-28865-w)) is an earlier variant that uses interface pLDDT and the logarithm of the number of contacts. The local interaction score ([Kim et al., 2024](https://doi.org/10.1101/2024.02.19.580970)) reports the fraction of interface residues with low mean cross-chain PAE. The interface pTM (`iptm_d0chn`) reports the original AlphaFold-style interface pTM recomputed from PAE with the chain-length reference distance.

### Learning Resources

- [DunbrackLab/IPSAE](https://github.com/DunbrackLab/IPSAE) (Dunbrack Lab, Fox Chase Cancer Center). Official repository and the source of the reference scoring script that this toolkit invokes.

## Tools

### IPSAE Interface Scoring (`ipsae-scoring`)

Scores a cofolded protein complex by computing ipSAE, pDockQ2, LIS, pDockQ, and interface pTM for a designated binder chain against one or more target chains. The tool takes a `Structure` with per-residue pLDDT in the B-factor column and the PAE matrix attached at `structure.metrics["pae"]`, runs the IPSAE scoring program, and returns the headline scores together with a full per-chain-pair breakdown.

#### Applications

This tool is appropriate for ranking and filtering cofolded complexes from structure-prediction tools such as AlphaFold 3, Chai-1, Boltz, or Protenix. Representative applications include scoring candidate protein binders from a design pipeline, identifying the most promising poses in a multi-chain prediction ensemble, and any analysis that benefits from multiple complementary interface-quality scores in a single call rather than running several scoring programs in sequence.

#### Usage Tips

- **The PAE matrix is required and must be attached at `structure.metrics["pae"]` as a square `list[list[float]]`.** The dimension should match the total residue count of the structure. The input is rejected when the matrix is missing or not square.
- **Per-residue pLDDT must be supplied via the B-factor column.** Structure predictors in proto-tools return the correct `b_factor_type` automatically, and `Structure.from_file()` auto-detects it for AlphaFold DB and ModelArchive files. For manually provided structures from other sources, pass `b_factor_type=BFactorType.PLDDT` (raw 0 to 100) or `BFactorType.NORMALIZED_PLDDT` (0 to 1) explicitly. The input is rejected when `b_factor_type` is any other value, since pDockQ and pDockQ2 would otherwise be computed incorrectly.
- **Missing or duplicated chains are rejected.** The binder chain must not also appear in `target_chains`, and every requested chain must be present in the structure. Single-character chain identifiers are also required because IPSAE reads PDB-format input. Multi-character mmCIF chain labels should be shortened with `Structure.to_pdb_with_chain_mapping()` before scoring.
- **The tool implementation does not check that the PAE matrix is aligned with the structure residue by residue.** It only confirms the matrix is square and two-dimensional. The caller is responsible for ensuring the PAE rows and columns match the residue order in the structure. A misaligned PAE matrix produces silently meaningless scores rather than an error.
- **The top-level scores report the symmetric maximum value for the binder-target interface.** The full directional and symmetric breakdown for every chain pair is available on `result.metrics.chain_pair_results`. When no symmetric pair matches the binder-target combination, the tool raises a `ValueError` listing the available chain pairs so the chain labels can be corrected.
- **`pae_cutoff` and `distance_cutoff` control the interface definition.** Both default to `10.0` Å. Lower values define a tighter interface and reduce the number of residues contributing to each score. Use a tighter cutoff when comparing interfaces of similar overall size or when assessing buried interfaces.
- **An empty interface returns zeros for every score.** When no residues fall within both cutoffs across the binder-target chain pair, every metric is `0.0`. Verify the chain identifiers and cutoff values before interpreting an all-zero result as a poor interface.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every IPSAE tool in this toolkit (`ipsae-scoring`).

- **Outputs are returned as typed metric objects.** Each `IPSAEMetrics` result carries the five top-level scores, the `chain_pair_results` breakdown for every chain pair, and the headline `primary_metric` (`ipsae`). Results can be exported to JSON through the standard export method.
