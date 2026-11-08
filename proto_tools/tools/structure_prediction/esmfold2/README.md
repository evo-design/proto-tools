<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ESMFold2

![ESMFold2](https://proto-bio.github.io/proto-assets/images/tool/esmfold2/hero.png)

> [!NOTE]
> **License:** ESMFold2 is open source and free for academic and commercial use under an MIT license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/Biohub/esm/blob/main/LICENSE.md) for full terms.

## Overview

ESMFold2 is [Biohub](https://biohub.ai)'s all-atom biomolecular structure predictor and successor to ESMFold. It folds complexes containing proteins, DNA, RNA, and small-molecule ligands, from sequence alone or with an optional multiple-sequence alignment (MSA). This toolkit runs ESMFold2 on a local GPU, both to predict structures and to differentiate structural objectives with respect to a designed sequence, across the MSA-capable `esmfold2` checkpoint and the inference-optimized single-sequence `esmfold2-fast` checkpoint.

## Background

ESMFold2 ([Candido et al., 2026](https://biohub.ai/papers/esm_protein.pdf)) extends the ESM family from protein-only single-sequence folding to all-atom prediction of biomolecular complexes. Where the original ESMFold ([Lin et al., 2023](https://doi.org/10.1126/science.ade2574)) used the ESM-2 protein language model as a learned substitute for an MSA and folded a single protein chain into a backbone, ESMFold2 supports proteins, DNA, RNA, small-molecule ligands, modified residues, and covalent bonds in a single joint prediction, comparable in scope to AlphaFold3 and Boltz-2. The model can be run in single-sequence mode, or, when an MSA is available for a protein chain, conditioned on the alignment to recover the evolutionary signal that aids prediction of difficult or sparsely-engineered targets.

Architecturally, ESMFold2 conditions on representations from the frozen ESMC 6B language model, pools them into a two-dimensional pair representation refined through a stack of folding layers with a stabilized recurrent update, and concludes with a diffusion transformer that denoises directly into all-atom coordinates. Two inference-time parameters, the number of refinement loops through the folding stack and the number of diffusion sampling steps, trade computation time for accuracy and can materially improve predictions on difficult targets, especially antibody-antigen complexes. Alongside the structure, ESMFold2 reports calibrated confidence: a per-residue predicted local distance difference test (pLDDT), a predicted aligned error (PAE) for the relative placement of any two tokens, and predicted template-modeling (pTM) and interface predicted template-modeling (ipTM) scores that summarize overall and interface accuracy.

Two checkpoints are available. `esmfold2` is the larger, MSA-capable model recommended for difficult or long targets where alignment signal aids prediction; `esmfold2-fast` is an inference-optimized single-sequence variant intended for high-throughput applications. Both are distributed under the MIT license at [Biohub/esm](https://github.com/Biohub/esm), the consolidated package that also distributes ESM3 and ESM C.

Folding a sequence says what a design is likely to do; inverting that prediction says how to change it. The gradient tools take a relaxed, continuous distribution over the designed chain instead of a discrete sequence, which makes the fold differentiable end to end: a derivative taken through the folding trunk says how to move each residue's distribution to improve the predicted complex. The quantities are read off the predicted distogram, the model's distribution over pairwise distances. Contact objectives score how confidently a residue is placed near a partner, globularity compares a chain's radius of gyration against the empirical scaling for a packed protein of the same length, and the epitope objective aims a binder at named target residues. A language-model term scores the designed chain's plausibility as a sequence, which counteracts the tendency of a purely structural objective to drift toward sequences that fold well on paper and not at all in reality.

### Learning Resources

- [ESMFold2 model card](https://huggingface.co/biohub/ESMFold2) (Biohub) - architecture details, training data, benchmark results, and intended-use guidance for the MSA-capable checkpoint.

## Tools

### ESMFold2 Structure Prediction (`esmfold2-prediction`)

Predicts the all-atom 3D structure of a biomolecular complex. Each input complex can combine protein, DNA, RNA, and ligand chains (with optional chain-level modifications); the assembly is folded by ESMFold2 and returned as a predicted `Structure` per complex with confidence metrics: pLDDT, pTM, interface pTM (for multi-chain complexes), and predicted aligned error.

#### Applications

This tool predicts the structure of multi-component assemblies such as protein-protein, protein-DNA, protein-RNA, and protein-ligand complexes, including antibody-antigen interfaces where ESMFold2 is reported to be competitive with AlphaFold3. Running it on a multi-chain complex also estimates how confidently the components are placed relative to each other through interface pTM and PAE, which is informative for assessing predicted interfaces.

#### Usage Tips

- **`model_checkpoint` selects the variant.** `esmfold2-fast` (default) is the inference-optimized single-sequence model and is appropriate for most high-throughput applications; select `esmfold2` (with `use_msa=True`, or by attaching precomputed `msas` on the input) for the larger MSA-capable model on difficult or long targets. Setting `use_msa=True` with `esmfold2-fast` raises a validation error, and `msas` supplied with `esmfold2-fast` are ignored with a logged warning.
- **`num_loops` (default `20`) and `num_sampling_steps` (default `100`) trade computation for accuracy.** Both parameters materially affect prediction quality, with the largest gains on difficult targets such as antibody-antigen complexes. Increasing either improves accuracy but extends runtime; decreasing them accelerates high-throughput screens at some accuracy cost. The ESMfold2 paper uses 20 loops by default, while their folding examples typically use 10 loops to balance speed and accuracy.
- **Multi-modal inputs.** Protein, DNA, RNA, and small-molecule ligand chains are supported; ligands can be specified by CCD code or SMILES, and chain-level modifications are accepted. SMILES-based ligand input is supported but currently has known accuracy issues; CCD codes are recommended.
- **Confidence is reported as pLDDT, pTM, ipTM, and PAE.** Mean pLDDT (0 to 1) is the primary per-structure quality metric; `iptm` is emitted only for multi-chain complexes, and `avg_pae` is in angstroms (0 to about 32). Set `include_pae_matrix=True` to attach the full per-token PAE matrix.

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
- **`globularity` is compressed on extended chains.** It infers a radius of gyration from
  expected pairwise distances, and the distogram cannot represent beyond 22 A, so a chain whose
  real pair distances run past that is scored as more compact than it is. It still ranks compact
  against extended, but treat its absolute value as a lower bound rather than an estimate.

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
- **`inter_contact` discriminates weakly.** Its 22 A contact cutoff sits at the edge of what the
  distogram represents, so it admits all but the clamped top bin and separates "in contact" from
  "not" far less sharply than the shorter-range terms do. Weight it against `epitope`, whose
  12 A default cutoff is well inside the range, rather than relying on it alone.
- **`epitope` needs hotspots**, each a residue letter and a 1-indexed position such as `L15`,
  checked against the target sequence so a stale selection is rejected rather than re-aimed.
- **`plm_nll` loads a second model.** Leave it out and ESM C is never loaded.
- **The alphabet is yours.** Give any permutation of the canonical amino acids and the gradient
  comes back in that same column order.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ESMFold2 tool in this toolkit (`esmfold2-prediction`, `esmfold2-inversion-gradient`, `esmfold2-inversion-binder-gradient`).

- **Requires a GPU.** ESMFold2 runs through a PyTorch backend and needs an NVIDIA GPU; CPU execution is not practical.
- **Shared `biohub_esm` environment.** ESMFold2 is part of the consolidated [Biohub/esm](https://github.com/Biohub/esm) package and shares its standalone environment with the ESM3 and ESM C toolkits, so installing any one of them provisions the others.
- **AlphaFold3-style diffusion with optional MSAs.** Predictions are stochastic, so set `seed` for reproducibility across runs. MSAs are only consumed by the `esmfold2` checkpoint; the `esmfold2-fast` checkpoint is single-sequence by construction.
- **One checkpoint serves folding and gradients.** All three tools share one worker and one loaded copy of the weights, so a prediction and a gradient call at the same `model_checkpoint` do not reload the model. Use `ToolInstance.persist_tool` across the steps of an optimization to keep it loaded.
- **The gradient tools are GPU-only and expensive.** Each call folds a complex and runs one backward per requested term. Measure a single call before choosing how many optimization steps to budget.
- **Gradients read the distogram, not coordinates.** The objectives differentiate ESMFold2's distribution over pairwise distances. They are not gradients through diffusion-generated coordinates, and no coordinates are produced or differentiated.
- **The distogram reaches 22 angstroms.** It is 64 bins dividing 2-22 A, and anything farther is clamped into the top bin, so no objective can distinguish 25 A from 35 A. Short-range quantities are unaffected and are in fact resolved slightly more finely than a wider head would manage; quantities that depend on long distances are compressed. See the per-tool tips.
- **Seeded, but deliberately stochastic.** The seed scopes checkpoint loading, dropout and the pseudo-perplexity mask draws; the reference recipe keeps language-model dropout active, so repeated calls are close rather than bit-identical.
- **Folding only.** This toolkit provides ESMFold2's structure prediction and its differentiable objectives; the broader ESM family's language-model, generation, and embedding capabilities are provided by the sibling ESM3 and ESM C toolkits.

<!-- docs:ignore start -->
### Provenance

The inversion structure objectives are reimplemented from the MIT-licensed ESMFold2
binder-design recipe published by the Chan Zuckerberg Biohub, at
[`cookbook/tutorials/binder_design.py`](https://github.com/bviggiano/esm/blob/1ad44674ef25f5a783273751ee3261617f4c5504/cookbook/tutorials/binder_design.py).
The "Algorithm N" labels in `standalone/objective.py` refer to that reference. The
recipe is written against `biohub/ESMFold2-Experimental-Fast`; the objectives here are
recalibrated for the production checkpoints' distogram.
<!-- docs:ignore end -->

## References

- Candido, S. et al. Language Modeling Materializes a World Model of Protein Biology. *Preprint* (2026). [biohub.ai/papers/esm_protein.pdf](https://biohub.ai/papers/esm_protein.pdf)
- Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* (2023). [doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
