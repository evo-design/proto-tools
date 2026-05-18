<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/malinois"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Malinois

> [!NOTE]
> **License:** Malinois is open source and free for academic and commercial use under an MIT license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/sjgosai/boda2/blob/main/LICENSE.mit) for full terms.

## Overview

Malinois is the CODA/BODA2 convolutional neural network for predicting MPRA-measured regulatory DNA activity from 200 bp inserts. This toolkit scores sequences in K562, HepG2, and SK-N-SH cell contexts and exposes a differentiable activity-gradient call for sequence design.

## Background

Malinois is the regulatory sequence model used in CODA ([Gosai et al., 2024](https://doi.org/10.1038/s41586-024-08070-z)) for machine-guided design of cell-type-targeting cis-regulatory elements. The model adapts the Basset-style convolutional architecture to MPRA data and predicts activity from a fixed 200 nucleotide insert after adding the assay flanks expected by the published checkpoint.

The model returns one raw activity value for each supported cell context: K562, HepG2, and SK-N-SH. The scoring wrapper averages forward and reverse-complement predictions and returns selected raw outputs. The gradient wrapper applies max/min sigmoid objective terms to these raw scores and backpropagates through relaxed A,C,G,T logits, matching the Fast SeqProp-style design path used for regulatory DNA optimization.

## Tools

### Malinois Score (`malinois-score`)

Scores one or more 200 bp DNA inserts and returns raw Malinois predictions keyed by requested cell type.

#### Applications

Use this tool to rank regulatory DNA designs by predicted activity in K562, HepG2, or SK-N-SH cells, screen MPRA insert candidates, or compare candidate designs before selecting sequences for downstream validation.

#### Usage Tips

- **Sequence length is fixed by default.** Inputs must match `seq_length`, which defaults to 200 bp.
- **Cell type keys are canonical.** Request outputs as `K562`, `HepG2`, and `SKNSH`; `SKNSH` maps to the SK-N-SH model output.
- **Batch size affects throughput.** Increase `batch_size` for many same-length inserts when GPU memory allows.

### Malinois Gradient (`malinois-gradient`)

Computes a weighted differentiable activity objective and, by default, returns the gradient with respect to batched relaxed DNA logits.

#### Applications

Use this tool inside gradient-based DNA design loops to maximize activity in an on-target cell type while minimizing activity in off-target cell types. It is designed for optimizer calls rather than final biological validation.

#### Usage Tips

- **Logits are batched.** Pass logits with shape `B x L x 4` in `A,C,G,T` order; use `B=1` for a single candidate.
- **Directions are per term.** `direction="max"` minimizes `1 - sigmoid(raw)` and `direction="min"` minimizes `sigmoid(raw)` after centering and scaling.
- **Soft/hard mixing controls relaxation.** `soft=1.0, hard=0.0` is fully soft; increasing `hard` uses a straight-through hard-forward estimator.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Malinois tool in this toolkit (`malinois-score`, `malinois-gradient`).

- **Requires a GPU.** Both tools load a PyTorch Malinois checkpoint and run most practically on CUDA.
- **Weights are provisioned automatically.** By default, the standalone worker downloads the CODA Zenodo artifact into the managed model cache and verifies its MD5 checksum.
- **The gradient tool is a single evaluation.** It returns one loss and optional gradient for the provided logits; run it from an optimizer for iterative design.

## References

- Gosai, S. J. et al. Machine-guided design of cell-type-targeting cis-regulatory elements. *Nature* 634, 1211-1220 (2024). DOI: [10.1038/s41586-024-08070-z](https://doi.org/10.1038/s41586-024-08070-z)
- CODA/BODA2 repository: [sjgosai/boda2](https://github.com/sjgosai/boda2)
- CODA supplemental data and resources: [Zenodo record 10698014](https://doi.org/10.5281/zenodo.10698014)
