<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/rs3"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Rule Set 3

> [!NOTE]
> **License:** Rule Set 3 is open source and free for academic and commercial use under an Apache-2.0 license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/gpp-rnd/rs3/blob/master/LICENSE) for full terms.

## Overview

[Rule Set 3](https://github.com/gpp-rnd/rs3) predicts SpCas9 guide activity from a 30-nucleotide target context and the experimental tracrRNA architecture. This wrapper exposes the published sequence model without adding hand-designed guide heuristics.

## Background

Rule Set 3 was trained to predict guide activity in CRISPR screening and explicitly accounts for differences between common tracrRNA scaffolds. Its raw output is a relative activity score rather than a probability, so comparisons should stay within the model's SpCas9 screening domain.

## Tools

### Rule Set 3 On-Target Score (`rs3-score`)

Scores contexts arranged as four upstream bases, a 20-base protospacer, a three-base NGG PAM, and three downstream bases. Select `Hsu2013` or `Chen2013` to match the tracrRNA scaffold. Scores are raw mean-centered model outputs, not calibrated probabilities; higher values predict greater activity within the model domain.

The tool validates the context and reports the exact model/package version. It does not evaluate genomic specificity, chromatin accessibility, target-gene features, or non-SpCas9 nucleases.

#### Applications

Use this tool to rank PAM-compatible SpCas9 guide candidates after exact target matching and before separate genome-wide off-target evaluation.

#### Usage Tips

- Supply the complete 30-nt context in the documented orientation; do not score an isolated 20-mer.
- Select the tracrRNA architecture that matches the experimental construct.
- Retain raw scores and model version in candidate provenance rather than interpreting the score as a calibrated success probability.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **Runs on CPU in an isolated Python 3.10 environment,** because the published package pins its training-era scikit-learn dependency.
- **Batched.** One call scores many contexts and results preserve input order.
- **Not available on Apple Silicon.** The pinned `lightgbm==3.3.5` ships no macOS arm64 wheel, so the environment build falls back to compiling it from source and needs `cmake` and `libomp`. Linux and hosted execution use the published wheels.
