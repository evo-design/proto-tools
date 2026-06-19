<a href="https://bio-pro.mintlify.app/tools/structure-scoring/structure-metrics"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Structure Metrics

![Structure Metrics](https://proto-bio.github.io/proto-assets/images/tool/structure_metrics/hero.png)

> [!NOTE]
> **License:** Structure Metrics is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/evo-design/proto-tools) for full terms.

## Overview

Structure Metrics computes coarse geometric and secondary-structure descriptors for a protein structure: secondary-structure percentages (helix, sheet, loop), the length of the longest contiguous alpha helix, and the radius of gyration. The metrics are computed using the [Biotite](https://www.biotite-python.org/) Python library and are appropriate as a fast first-pass filter for catching common structure-prediction artifacts such as unrealistically long helices or extended, non-compact conformations.

## Background

The tool assigns secondary structure using the P-SEA algorithm ([Labesse, Colloc'h, Pothier, and Mornon, 1997](https://doi.org/10.1093/bioinformatics/13.3.291)) implemented in [Biotite](https://www.biotite-python.org/), which classifies each protein residue as alpha helix, beta sheet, or loop from the Cα-atom trace alone using distance and angle patterns. The reported helix, sheet, and loop percentages summarise the overall secondary-structure composition of the structure, while the longest contiguous alpha-helix length is a separate scalar that captures the longest single helix without averaging across the structure.

Radius of gyration is the mass-weighted root-mean-square distance of all atoms from the centre of mass and is a standard scalar measure of overall structural compactness used in small-angle X-ray scattering, polymer physics, and protein structural analysis. For proteins of a given length, compact native folds produce smaller gyration radii than disordered or partially folded conformations, which is what makes the metric useful as an artifact filter for predicted structures.

Both metrics are useful as inexpensive sanity checks on structures produced by sequence-based predictors such as ESMFold, AlphaFold, Chai, Boltz, and Protenix. Predictors can default to extended helical bundles for low-confidence regions, and failed folds frequently appear as extended conformations with elevated gyration radii. The Biotite Python library ([Kunzmann et al., 2023](https://doi.org/10.1186/s12859-023-05345-6)) provides the underlying secondary-structure annotation and gyration-radius implementations used by this tool.

### Learning Resources

- [Biotite documentation](https://www.biotite-python.org/) (TU Darmstadt). API reference for the secondary-structure annotation and gyration-radius computations used by this tool.

## Tools

### Structure Quality Metrics (`structure-metrics`)

Computes five quality metrics for each input structure: `helix_pct`, `sheet_pct`, `loop_pct` (secondary-structure composition on the 0 to 100 scale), `longest_alpha_helix` (residue count of the longest contiguous alpha-helical segment), and `gyration_radius` (radius of gyration in Å). Inputs are passed as a list of structures and results are returned in the same order.

#### Applications

This tool is appropriate as a fast first-pass filter for batch screening of predicted protein structures. Representative applications include flagging predicted structures with unrealistically long alpha helices that often arise as artifacts on low-confidence regions, identifying extended or disordered conformations that fail to fold compactly, summarising the secondary-structure composition of a designed protein, and ranking generated structures by structural plausibility before more expensive downstream analyses.

#### Usage Tips

- **Inputs accept a list of `Structure` objects, file paths, or raw PDB or mmCIF content strings.** A single bare input is automatically wrapped in a list. Each item is coerced to a `Structure` before analysis.
- **All five metrics are computed over every chain of the input structure.** There is no per-chain breakdown at the tool level. To analyse a specific chain, extract that chain into its own `Structure` using `Structure.select_chain()` before passing it in.
- **Filter thresholds depend on the protein family.** A 50-residue alpha helix is a strong artifact signal for a typical globular protein but is normal for coiled-coil and fibrous proteins. A gyration radius above 45 Å indicates failed folding for a 1000-residue protein but is expected for naturally elongated proteins. Calibrate thresholds against known structures of the protein family being screened.
- **The `secondary_structure_percentages` summary and `longest_alpha_helix` use the same P-SEA assignment.** The helix percentage and longest contiguous helix length are derived from the same per-residue annotation, so a structure with `helix_pct=80` and `longest_alpha_helix=200` indicates that nearly the entire structure is one continuous helix.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Structure Metrics tool in this toolkit (`structure-metrics`).

- **Outputs are returned as typed metric objects.** Each `StructureQualityMetrics` entry carries `longest_alpha_helix` (integer residue count), `gyration_radius` (Å), `helix_pct`, `sheet_pct`, and `loop_pct` (all on the 0 to 100 scale). Results can be exported to CSV or JSON through the standard export method.
- **The tool implementation runs entirely in-process and uses CPU only.** Computation is performed in pure Python through Biotite, with no standalone environment or separate program invoked. Per-structure runtime is sub-second for typical protein sizes and scales linearly with the number of input structures.
