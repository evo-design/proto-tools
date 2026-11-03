<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold2-inversion"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESMFold2 Inversion

> [!NOTE]
> **License:** ESMFold2 Inversion is open source and free for academic and commercial use under a MIT license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/Biohub/esm/blob/main/LICENSE.md) for full terms.

## Overview

Folding a sequence tells you what a design is likely to do; inverting that prediction tells you how to change the design. This toolkit exposes the ESMFold2 inversion checkpoints as a differentiable objective: given a fixed target chain and a relaxed distribution over a designed binder, it folds the complex once and reports each requested structural quantity together with its derivative with respect to the binder's logits.

It is a single evaluation, not a design loop: stepping the logits, annealing, and rescoring candidates all belong to the caller.

## Background

The inversion checkpoints ([Candido et al., 2026](https://biohub.ai/papers/esm_protein.pdf)) are a separate family from the maintained ESMFold2 prediction checkpoints. They accept a relaxed, continuous description of the designed chain rather than a discrete sequence, which is what makes the fold differentiable end to end: a gradient taken through the folding trunk says how to move each residue's distribution to improve the predicted complex.

The quantities themselves are read off the predicted distogram, the model's distribution over pairwise distances. Contact objectives score how confidently a residue is placed near a partner, globularity compares the binder's radius of gyration against the empirical scaling for a packed protein of the same length, and the epitope objective aims the binder at named target residues. A language-model term scores the designed chain's plausibility as a protein sequence, which counteracts the tendency of a purely structural objective to drift toward sequences that fold well on paper and not at all in reality.

## Tools

### ESMFold2 Inversion Gradient (`esmfold2-inversion-gradient`)

Folds one target/binder complex per candidate and differentiates each requested quantity against the relaxed binder state.

| term | measures | units |
|---|---|---|
| `intra_contact` | binder self-contact over each residue's two best intra-chain partners | — |
| `inter_contact` | interface contact between the target and the binder | — |
| `globularity` | binder radius of gyration in excess of a packed protein | angstroms |
| `epitope` | contact surprisal between the binder and named target hotspots | — |
| `plm_nll` | ESM C masked pseudo-perplexity of the designed chain | nats per masked position |

#### Applications

Use this inside a gradient-based binder design loop, where an optimizer holds a relaxed sequence state, asks for the gradient, and steps it. The terms are reported separately so an optimizer can weigh structure against sequence plausibility, or drive the binder onto a chosen epitope, without the tool deciding the trade-off.

#### Usage Tips

- **Every quantity is lower-is-better**, so composing without naming goals already minimizes them. Weights are how you balance them: `result.compose(weights={...})` reproduces any fixed weighting, and changing it costs nothing because the terms are reported raw.
- **One fold, many gradients.** All requested quantities come from a single forward pass, with one backward each. Asking for five terms costs one fold and five backwards, not five folds.
- **`epitope` needs hotspots.** Each is a residue letter and a 1-indexed position, such as `L15`, and is checked against the target sequence — a hotspot that names a residue the target does not have at that position is rejected rather than silently aimed elsewhere.
- **`plm_nll` loads a second model.** Leave it out of `gradient_terms` and ESM C is never loaded.
- **The alphabet is yours.** Give `alphabet` as any permutation of the canonical amino acids and the returned gradient comes back in that same column order.
- **The binder is the only mutable chain.** The target is supplied as a discrete sequence and never receives a gradient.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **GPU only, and expensive.** Each call folds a complex and runs one backward per requested term. Measure a single call before choosing how many optimization steps to budget.
- **Shares the `biohub_esm` environment** with ESM C, ESM3 and ESMFold2 prediction, so it adds no new dependency and tracks the same pinned `esm` revision.
- **Seeded, but deliberately stochastic.** The seed scopes checkpoint loading, dropout and the pseudo-perplexity mask draws; the upstream recipe keeps language-model dropout active, so repeated calls are close rather than bit-identical.
- **Use `ToolInstance.persist_tool`** across the steps of an optimization so the checkpoints are loaded once.

<!-- docs:ignore start -->
### Provenance

The structure objectives are reimplemented from the MIT-licensed ESMFold2 binder-design recipe published by the Chan Zuckerberg Biohub, at [`cookbook/tutorials/binder_design.py`](https://github.com/bviggiano/esm/blob/1ad44674ef25f5a783273751ee3261617f4c5504/cookbook/tutorials/binder_design.py). The "Algorithm N" labels in `standalone/objective.py` refer to that reference. Each ported quantity is checked against it in float64; see the PR for the comparison.
<!-- docs:ignore end -->

## References

- Candido, S. et al. Language Modeling Materializes a World Model of Protein Biology. *Preprint* (2026). [biohub.ai/papers/esm_protein.pdf](https://biohub.ai/papers/esm_protein.pdf)
