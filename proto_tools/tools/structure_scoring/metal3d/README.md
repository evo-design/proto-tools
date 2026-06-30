<a href="https://bio-pro.mintlify.app/tools/structure-scoring/metal3d"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Metal3D

> [!NOTE]
> **License:** Metal3D uses MIT for code and CC-BY-4.0 for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/gelnesr/dEVA/blob/main/LICENSE) and [model weights license](https://github.com/lcbc-epfl/metal-site-prediction#license) for full terms.

## Overview

Metal3D predicts metal-ion locations in protein structures from residue-centered 3D voxel features. This toolkit defaults to the original Metal3D checkpoint and additionally exposes dEVA's retrained Metal3D-Cat and Metal3D-Clean checkpoints through the Proto tool interface, returning clustered metal-site probabilities, optional per-candidate residue probabilities, and an annotated PDB containing the top predicted zinc site when it passes the configured threshold.

## Background

Metal3D ([Dürr, Levy, and Rothlisberger, 2023](https://doi.org/10.1038/s41467-023-37870-6)) is a 3D convolutional neural network for locating likely zinc-binding sites in protein structures. It evaluates residue-centered voxel boxes around candidate metal-binding residues and converts the predicted grid probabilities into metal-site coordinates.

The `metal3d-original` checkpoint is the original Metal3D network; `metal3d-cat` (catalytic) and `metal3d-clean` (cleaned) are dEVA retrains that use a modified architecture (4³ convolution kernels versus the original's 3³) and a wider grid-probability averaging radius. All three checkpoints are downloaded from `gelnesr/dEVA` during standalone setup.

### Learning Resources

- [Metal3D paper](https://doi.org/10.1038/s41467-023-37870-6). Nature Communications article describing the original Metal3D model.
- [Metal3D repository](https://github.com/lcbc-epfl/metal-site-prediction). Original Metal3D code and the `metal3d-original` checkpoint.
- [dEVA paper](https://www.biorxiv.org/content/10.1101/2026.04.23.720277). El Nesr et al., describing the dEVA framework behind the `metal3d-cat` and `metal3d-clean` checkpoints.
- [dEVA repository](https://github.com/gelnesr/dEVA). Source of the inference code and the checkpoint files used by this wrapper.

## Tools

### Metal3D Prediction (`metal3d-prediction`)

Predicts metal-ion sites for one or more input protein structures. Each input can optionally include a `candidate_residues` selection keyed by chain identifier; when omitted, the standalone worker evaluates canonical metal-binding residue types across the protein.

#### Applications

This tool is appropriate for scoring enzyme-design proposals by predicted metal-site strength, checking whether a redesigned structure still supports a target metal pocket, and annotating likely zinc-site coordinates before downstream structural inspection.

#### Usage Tips

- **`metal3d-original` (default) is the original Metal3D zinc model.** Switch to `metal3d-cat` for catalytic metal sites or `metal3d-clean` for the cleaned variant; both are dEVA retrains on the modified architecture.
- **Pass `candidate_residues` when the pocket is known.** Candidate filtering reduces the scored residue set and returns per-residue probabilities for the configured pocket positions.
- **Tune `probability_threshold` for reporting, not model inference.** The model always produces grid probabilities; the threshold controls which clustered sites are returned and whether the top zinc site is appended to the annotated PDB.
- **Use persistent tool instances for repeated calls.** The worker keeps the selected checkpoint loaded when reused through `ToolInstance.persist_tool("metal3d")`.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Metal3D tool in this toolkit (`metal3d-prediction`).

- **Structure inputs accept typed `Structure` objects or path / coordinate strings.** The wrapper writes PDB text to the standalone worker and remaps temporary PDB-safe chain identifiers back to the original chain identifiers in the returned residue probabilities.
- **Outputs are returned as typed metric objects.** Each result carries `pmetal`, a `found` flag, clustered `sites`, optional `residue_probabilities`, and an `annotated_structure`. JSON and PDB export are supported through the standard export interface.
