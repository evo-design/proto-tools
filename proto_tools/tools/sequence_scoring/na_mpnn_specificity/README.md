<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/na-mpnn-specificity"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# NA-MPNN

> [!NOTE]
> **License:** NA-MPNN uses MIT for code and BSD-3-Clause for model weights and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/baker-laboratory/NA-MPNN/blob/main/LICENSE) and [model weights license](https://github.com/baker-laboratory/NA-MPNN/blob/main/LICENSE) for full terms.

## Overview

[NA-MPNN](https://github.com/baker-laboratory/NA-MPNN) is a message-passing neural network for nucleic-acid sequence design and protein–DNA specificity prediction. Given a protein–DNA complex structure, it predicts, for each DNA position, the base preference of the bound protein. This toolkit exposes that specificity prediction as a single tool that returns a canonical DNA-only position probability matrix (PPM) in `A,C,G,T` order.

## Background

Sequence-specific DNA-binding proteins recognize their target sites through a combination of direct base contacts and indirect shape readout. Predicting the base preference a protein imposes at each position of a bound DNA site is central to engineering transcription factors, characterizing binding specificity, and designing synthetic regulatory elements such as repressors and promoters.

NA-MPNN ([Kubaney et al., 2025](https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2)) is a message-passing neural network that represents proteins, DNA, and RNA within a single unified biopolymer graph, framing both RNA sequence design and protein–DNA binding specificity as nucleic-acid inverse-folding problems (recovering likely sequences for a fixed three-dimensional structure). For specificity prediction, the model conditions on a fixed-docked protein–DNA complex and emits per-position nucleotide probabilities over the DNA chains, which this toolkit canonicalizes into a DNA-only `L x 4` PPM together with the recovered true sequence and chain labels.

### Learning Resources

- [baker-laboratory/NA-MPNN](https://github.com/baker-laboratory/NA-MPNN). Official repository with the training and inference code, installation instructions, and usage examples.

## Tools

### NA-MPNN Specificity (`na-mpnn-specificity`)

Predicts the DNA base preference of a bound protein from one or more protein–DNA complex structures. For each input PDB, the tool runs NA-MPNN specificity inference, restricts the output to valid DNA positions, and returns a canonical PPM (`L x 4`, `A,C,G,T` order), the recovered true DNA sequence (indices `0..3`), per-row masks, and chain labels. Each result is also written to a canonical `.npz` file whose path is returned in `output_npz_path`.

#### Applications

This tool is appropriate for scoring and ranking candidate protein–DNA binder designs by their predicted specificity, for comparing the base preference of a designed binder against a target motif inside an optimization loop, and for characterizing the specificity landscape of natural or engineered DNA-binding proteins. The per-position PPM can be compared to a desired motif to drive specificity-based selection.

#### Usage Tips

- **The NA-MPNN checkout and checkpoint are provisioned for you.** The standalone setup clones a pinned revision of the repository (which ships the specificity checkpoint in-tree) into the managed weights cache, so no repository or checkpoint path is configured. Point `PROTO_NA_MPNN_SPECIFICITY_WEIGHTS_DIR` at an existing checkout to reuse it. A run that uses `device='cloud'` fails fast because these local resources cannot be staged.
- **`predicted_ppm` rows are DNA-only and renormalized.** Only positions that are both valid and DNA are kept, and each row is renormalized over `A,C,G,T`, so the returned matrix already excludes protein and masked positions.
- **`temperature` controls sampling sharpness.** Lower values (default `0.1`) concentrate probability on the most-preferred base; raise it to soften the distribution.
- **`output_directory` and `keep_intermediate` control artifacts.** Leave `output_directory` unset to write canonical `.npz` files to a temporary directory, or set it to persist them. Set `keep_intermediate=True` to retain the raw NA-MPNN output for debugging.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- NA-MPNN runs as an isolated standalone environment that shells out to the upstream NA-MPNN inference CLI; the heavy model dependencies stay out of the main environment.
- The standalone setup clones a pinned NA-MPNN revision (with its in-tree specificity checkpoint) into the managed weights cache. If the clone cannot reach GitHub, the environment setup signals a clean test skip rather than a hard failure.
