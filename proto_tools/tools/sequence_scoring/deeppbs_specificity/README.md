<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/deeppbs-specificity"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# DeepPBS Specificity

> [!NOTE]
> **License:** DeepPBS Specificity is open source and free for academic and commercial use under a BSD-3-Clause license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/timkartar/DeepPBS/blob/main/LICENSE.txt) for full terms.

## Overview

DeepPBS (Deep Predictor of Binding Specificity) predicts the DNA base preferences of a protein directly from a protein-DNA complex structure. Given one or more PDB files, the `deeppbs-specificity` tool runs DeepPBS over each structure and returns a canonical DNA-only position probability matrix (PPM) in `A,C,G,T` order, alongside the true DNA sequence, residue masks, and per-base chain labels.

## Background

Sequence-specific recognition of [DNA](https://en.wikipedia.org/wiki/DNA) by proteins underlies transcriptional regulation, and predicting a [protein](https://en.wikipedia.org/wiki/Protein)'s binding preference directly from a co-crystal structure is a long-standing goal. DeepPBS ([Mitra et al., 2024](https://doi.org/10.1038/s41592-024-02372-w)) applies [geometric deep learning](https://en.wikipedia.org/wiki/Geometric_deep_learning) over a graph representation of the protein-DNA interface to predict per-position base preferences that generalize across protein families, on experimental or predicted complex structures. The model consumes a processed representation of the complex built from [DSSR/X3DNA](https://en.wikipedia.org/wiki/Nucleic_acid_structure_determination) geometry, so the wrapper depends on a local DeepPBS repository and a local X3DNA install.

### Learning Resources

- [DeepPBS GitHub repository](https://github.com/timkartar/DeepPBS) - source code, processing scripts, and pretrained weights.
- [DeepPBS paper (Nature Methods, 2024)](https://doi.org/10.1038/s41592-024-02372-w) - the method, benchmarks, and applications.

## Tools

### DeepPBS Specificity (`deeppbs-specificity`)

Runs DeepPBS preprocessing and prediction on each input protein-DNA structure and returns, per structure, a canonical DNA PPM (`L x 4`, `A,C,G,T` order), the true DNA sequence indices, residue and DNA masks, per-base chain labels, and the path to a canonical `.npz` artifact. When a required DeepPBS dependency (the X3DNA `x3dna-dssr`/`analyze` binaries) is missing, or preprocessing or prediction fails to produce output, the tool raises by default. Set `allow_fallback=True` to instead emit a conservative fallback result: a uniform PPM (`0.25` per base) derived from the DNA residues in the input PDB, flagged with `used_fallback=True` and a human-readable `fallback_reason`.

#### Applications

- Estimating the DNA base preference of a designed or natural protein-DNA complex.
- Scoring protein-DNA designs for specificity against a target motif.
- Generating canonical PPMs for downstream motif comparison.

#### Usage Tips

- **Inputs are full protein-DNA PDB structures.** Provide a clean co-crystal containing both DNA strands; missing strands or non-standard residues can trigger the fallback path.
- **DeepPBS, X3DNA, and the weights are provisioned for you.** The standalone setup clones a pinned revision of the DeepPBS repository (which bundles the X3DNA/DSSR binaries, the process/predict configs, and the trained inference weights) into the managed weights cache and installs it, so no repository or X3DNA path is configured. Point `PROTO_DEEPPBS_SPECIFICITY_WEIGHTS_DIR` at an existing checkout to reuse it.
- **Fallback is opt-in.** By default a missing dependency or failed run raises; pass `allow_fallback=True` to get a uniform fallback PPM instead. A `used_fallback=True` result carries a uniform PPM, not a real prediction.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **The DeepPBS checkout is self-contained and auto-provisioned.** The standalone setup clones a pinned DeepPBS revision (bundling the X3DNA/DSSR binaries, configs, and inference weights) into the managed weights cache and installs it editable. The tool shells out to those local scripts and binaries, so it cannot run on `device='cloud'`. If the clone cannot reach GitHub, the environment setup signals a clean test skip rather than a hard failure.
- **Failures raise by default.** When a dependency is missing or processing fails, the tool raises; set `allow_fallback=True` to instead return a uniform fallback PPM flagged with `used_fallback` so downstream code can filter or re-run.
- **Results are index-aligned with the input.** Each result corresponds to the input structure at the same position.
