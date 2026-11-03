<a href="https://bio-pro.mintlify.app/tools/gene-annotation/crispr-guides"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# CRISPR Guide Enumeration

> [!NOTE]
> **License:** CRISPR Guide Enumeration is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/evo-design/proto-tools) for full terms.

## Overview

CRISPR Guide Enumeration scans reference-oriented DNA intervals for protospacers adjacent to an IUPAC PAM grammar. It examines either or both reference strands and returns sequence-exact candidates with biological coordinates and flanking context, without scoring or selecting them.

## Background

RNA-guided CRISPR nucleases recognize a protospacer together with a nuclease-specific protospacer-adjacent motif (PAM). The PAM's sequence and position relative to the protospacer vary by nuclease, so the operation accepts both an IUPAC PAM grammar and an explicit five-prime or three-prime orientation. The default `NGG` three-prime PAM and 20-nt protospacer describe SpCas9-style enumeration.

This implementation performs deterministic string matching rather than efficacy or off-target prediction. It uses standard [IUPAC-IUB nucleic-acid ambiguity symbols](https://iupac.qmul.ac.uk/misc/naabb.html) for the PAM grammar, but emits only candidates whose complete protospacer and observed PAM contain canonical A/C/G/T bases. Returned coordinates are always 1-indexed and inclusive, regardless of PAM strand.

## Tools

### CRISPR Guide Enumeration (`crispr-guides-enumerate`)

Enumerates every PAM-compatible protospacer in each supplied reference interval and bundles the candidates by input region.

#### Applications

Use this operation to create an auditable candidate pool before applying on-target activity models, mismatch/bulge-aware off-target searches, feature-window constraints, or portfolio selection. Its returned exterior context can feed models such as Rule Set 3 when the configured context dimensions match that model's contract.

#### Usage Tips

- **Supply intervals in reference plus-strand orientation.** Set each region's 1-indexed `start`; the tool derives all plus- and minus-strand coordinates from that representation.
- **Write the PAM in nuclease orientation.** Use `pam_side="three_prime"` for patterns such as SpCas9 `NGG`, or `pam_side="five_prime"` for patterns such as Cas12a `TTTV`.
- **`pam_strand` is the strand carrying the protospacer and PAM.** The guide RNA matches the protospacer and pairs with the opposite strand.
- **Fetch enough exterior sequence.** A candidate near an interval edge keeps whatever flank is available, so its `context_sequence` is shorter than `context_5p + protospacer + PAM + context_3p`.
- **Chain the output.** Under the defaults, `context_sequence` is the 30-nt window Rule Set 3 scores (4-nt flank, 20-nt protospacer, 3-nt PAM, 3-nt flank), and `protospacer` is what a mismatch-aware off-target search takes.
- **Score downstream.** This tool intentionally performs no guide ranking, efficacy prediction, off-target search, or candidate selection.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These notes apply to every tool in the CRISPR Guide Enumeration toolkit.

- **Pure CPU operation.** Enumeration runs in process with no separate environment, model weights, network access, or local database.
- **Naturally batched.** One call accepts multiple independently identified intervals and returns exactly one candidate bundle per interval.
- **Deterministic.** Identical normalized sequence and configuration inputs produce identical ordered candidates.
