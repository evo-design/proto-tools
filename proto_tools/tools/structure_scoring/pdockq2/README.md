<a href="https://bio-pro.mintlify.app/tools/structure-scoring/pdockq2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# pDockQ2

![pDockQ2](https://proto-bio.github.io/proto-assets/images/tool/pdockq2/hero.png)

> [!NOTE]
> **License:** pDockQ2 has an AGPL-3.0 license and may require explicit attribution when utilized. Please refer to [the license](https://gitlab.com/ElofssonLab/afm-benchmark/-/blob/main/LICENSE) for full terms.

## Overview

[pDockQ2](https://gitlab.com/ElofssonLab/afm-benchmark) is an interface-quality score for cofolded protein complexes developed by the [Elofsson Lab](https://www.bioinfo.se/) at SciLifeLab and Stockholm University. It combines per-residue pLDDT and the Predicted Aligned Error (PAE) matrix into a single estimate of the per-interface DockQ score in the range 0 to 1, with higher values indicating more reliably predicted interfaces. This toolkit re-implements the published scoring formula in pure Python and exposes it through a single registered tool that returns the overall pDockQ2 score together with a per-interface breakdown.

## Background

DockQ ([Basu and Wallner, 2016](https://doi.org/10.1371/journal.pone.0161879)) is a continuous interface-quality measure for protein-protein docking models that combines the CAPRI quality indicators (fraction of native contacts, interface RMSD, and ligand RMSD) into a single score in the range 0 to 1. The published thresholds approximate the CAPRI quality classes of Acceptable (DockQ ≥ 0.23), Medium (DockQ ≥ 0.49), and High (DockQ ≥ 0.80). DockQ requires a known reference complex and cannot be computed when only the predicted structure is available.

pDockQ ([Bryant, Pozzati, and Elofsson, 2022](https://doi.org/10.1038/s41467-022-28865-w)) was introduced as a predicted version of DockQ that uses only AlphaFold2 outputs, with no reference complex required. It estimates DockQ for a dimer from the mean pLDDT of interface residues together with the logarithm of the number of interface contacts, calibrated against ground-truth DockQ values on a benchmark of heterodimers.

pDockQ2 ([Zhu, Shenoy, Kundrotas, and Elofsson, 2023](https://doi.org/10.1093/bioinformatics/btad424)) generalises pDockQ to larger multi-chain complexes and replaces the contact-count term with the Predicted Aligned Error (PAE) matrix, which captures pairwise residue-position uncertainty across chains. For each interface, the score combines the contact-weighted mean interface pLDDT with the mean of `1 / (1 + (PAE / 10)²)` over interface residue pairs, then passes the product through a logistic sigmoid whose parameters were fit against ground-truth DockQ values on the AlphaFold-Multimer benchmark. The published analysis demonstrates that pDockQ2 estimates DockQ for each interface in a multimer rather than only for a single dimer.

### Learning Resources

- [ElofssonLab/afm-benchmark](https://gitlab.com/ElofssonLab/afm-benchmark) (Elofsson Lab, Stockholm University). Reference implementation of pDockQ2 and the benchmark data from the original publication.
- [bjornwallner/DockQ](https://github.com/bjornwallner/DockQ) (Wallner Lab, Linköping University). Reference implementation of the underlying DockQ measure that pDockQ2 estimates.

## Tools

### pDockQ2 Interface Quality (`pdockq2`)

Scores the per-interface quality of a cofolded protein complex by computing pDockQ2 for each chain pair and aggregating the per-chain scores into a single overall score. The tool takes a `Structure` with per-residue pLDDT in the B-factor column and the PAE matrix attached at `structure.metrics["pae"]`, identifies CA-CA contacts between every pair of chains within a configurable distance cutoff, applies the published sigmoid, and returns the overall score together with a per-chain interface breakdown.

#### Applications

This tool is appropriate for filtering and ranking cofolded complexes from structure-prediction tools such as AlphaFold-Multimer, AlphaFold 3, Chai-1, Boltz-2, and Protenix. Representative applications include gating candidate protein binders from a design pipeline by predicted interface quality, ranking the most promising poses in a multi-chain prediction ensemble, and screening large sets of predicted complexes before committing to more expensive downstream analyses.

#### Usage Tips

- **The PAE matrix is required and must be attached at `structure.metrics["pae"]` as a square `list[list[float]]` whose dimension matches the total residue count of the structure.** The input is rejected when the matrix is missing, not square, or of the wrong dimension.
- **Per-residue pLDDT must be supplied via the B-factor column.** Structure predictors in proto-tools return the correct `b_factor_type` automatically, and `Structure.from_file()` auto-detects it for AlphaFold DB and ModelArchive files. For manually provided structures from other sources, pass `b_factor_type=BFactorType.PLDDT` (raw 0 to 100) or `BFactorType.NORMALIZED_PLDDT` (0 to 1) explicitly. The input is rejected when `b_factor_type` is any other value, since the published sigmoid was fit on a 0 to 100 pLDDT scale.
- **A pDockQ2 score above 0.23 corresponds to the "Acceptable" DockQ quality class.** The thresholds derive from the underlying DockQ measure ([Basu and Wallner, 2016](https://doi.org/10.1371/journal.pone.0161879)): scores above 0.49 correspond to "Medium" quality and scores above 0.80 to "High" quality. Scores below 0.23 typically reflect either low interface pLDDT or high cross-chain PAE.
- **The overall score is the mean of `pmidockq` over target chains that contact the binder chain.** When no target chain in `target_chains` is within the distance cutoff of `binder_chain`, the overall score is set to `0.0`, `num_interface_contacts` is reported as `0`, and a warning is logged. Verify the chain identifiers and the cutoff before interpreting an all-zero result as a poor interface.
- **`distance_cutoff` controls the CA-CA contact distance used to define interface residues.** The wrapper default of `10.0` Å is more permissive than the `8.0` Å default used by the Elofsson Lab reference implementation against which the published sigmoid was calibrated. The qualitative DockQ-quality interpretation still applies at `10.0` Å, but quantitative scores will not exactly match the published values. Set `distance_cutoff=8.0` for scores that match the original pDockQ2 calibration. The PAE normalisation distance inside the sigmoid is independently fixed at 10 Å per the published formula and is not affected by this setting.
- **The interface pLDDT is contact-pair weighted, not residue-deduplicated.** A residue that contacts `k` cross-chain partners contributes its pLDDT `k` times to the interface mean. This matches the published pDockQ2 definition and is preserved by the wrapper.
- **The per-chain breakdown is available on `result.metrics.interfaces`.** Each `InterfacePDockQ2` entry exposes `chain_id`, `neighbor_chains`, `if_plddt` (0 to 100 pLDDT scale), `norm_pae` (0 to 1 normalised confidence, higher is more confident), and `pmidockq` (0 to 1 DockQ-scale prediction) for one chain. Inspect this list when debugging multi-chain targets or when the overall mean masks variation across interfaces.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every pDockQ2 tool in this toolkit (`pdockq2`).

- **Outputs are returned as typed metric objects.** Each `PDockQ2Metrics` result carries the overall `pdockq2` score (0 to 1), `avg_interface_plddt` (0 to 100 pLDDT scale), `avg_interface_pae` (0 to 1 normalised confidence), and `num_interface_contacts` (integer count) together with a per-chain `interfaces` breakdown. The headline `primary_metric` is `pdockq2`, and results can be exported to JSON through the standard export method.
- **The tool implementation runs entirely in-process and uses CPU only.** The scoring formula is re-implemented in pure Python with numpy, and no standalone environment or separate program is invoked. Per-call runtime is sub-second for typical complex sizes and scales quadratically with the total residue count because of the all-against-all CA-CA distance computation.
