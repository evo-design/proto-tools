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

Both tools fold a complex once and differentiate each requested quantity against the relaxed
state, sharing one worker, one checkpoint family and one set of objective implementations. They
differ in what they take in and what they offer.

The objectives read ESMFold2's **differentiable distogram prediction** — its distribution over
pairwise distances. They are not gradients through diffusion-generated coordinates, and no
coordinates are produced or differentiated.

### ESMFold2 Inversion Gradient (`esmfold2-inversion-gradient`)

General protein-complex inversion. A complex is a list of chains in fold order, each either
**fixed** (a discrete `sequence`) or **designable** (relaxed `logits`). Any number of chains may
be designed at once; fixed chains are folded as context and receive no gradient.

| term | measures | units |
|---|---|---|
| `intra_contact` | self-contact over a chain's two best contacts inside that same chain | — |
| `globularity` | that chain's radius of gyration in excess of a packed protein | angstroms |
| `plm_nll` | ESM C masked pseudo-perplexity of that chain, scored on its own | nats per masked position |

Each term names the chain it evaluates through `chain_id`; with exactly one designable chain the
default resolves on its own.

#### Applications

Hallucinating a monomer, redesigning one chain inside a fixed assembly, or co-designing several
chains of a complex in one optimization.

#### Usage Tips

- **One gradient spans every designable chain.** Rows are those chains' matrices concatenated in
  fold order, and `chain_rows` on each result gives the `[start, stop)` rows per chain. Terms
  therefore share one coordinate space and compose directly.
- **A chain boundary is a real boundary.** `intra_contact` counts only partners inside its own
  chain, `globularity` reads only its own sub-block, and `plm_nll` scores each chain as its own
  sequence rather than reading a boundary as a peptide bond.
- **Several terms on several chains coexist**, so `intra_contact` on chain B and on chain C are
  two separate quantities with two separate gradients.

### ESMFold2 Inversion Binder Gradient (`esmfold2-inversion-binder-gradient`)

Binder design against one fixed target. The two chain roles are part of the model rather than
something to configure, so the input is a target sequence, a relaxed binder, and optional
hotspots.

Offers the three single-chain terms above, applied to the binder, plus two interface terms:

| term | measures | units |
|---|---|---|
| `inter_contact` | interface contact between the target and the binder | — |
| `epitope` | contact surprisal between the binder and named target hotspots | — |

#### Applications

A gradient-based binder design loop: an optimizer holds the relaxed binder, asks for the
gradient, and steps it, weighing interface contact against the binder's own foldability.

#### Usage Tips

- **Every quantity is lower-is-better**, so composing without naming goals already minimizes
  them. `result.compose(weights={...})` reproduces any fixed weighting.
- **`inter_contact` is deliberately asymmetric.** It is scored over the target and partnered
  against the binder, so it rewards the binder reaching the target rather than the reverse. It
  is not a generic inter-chain objective, which is why the general tool does not offer it.
- **`epitope` needs hotspots**, each a residue letter and a 1-indexed position such as `L15`,
  checked against the target sequence so a stale selection is rejected rather than re-aimed.
- **`plm_nll` loads a second model.** Leave it out and ESM C is never loaded.
- **The alphabet is yours.** Give any permutation of the canonical amino acids and the gradient
  comes back in that same column order.

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
