<a href="https://bio-pro.mintlify.app/tools/inverse-folding/caliby"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Caliby

![Caliby](https://proto-bio.github.io/proto-assets/images/tool/caliby/hero.png)

> [!NOTE]
> **License:** Caliby is open source and free for academic and commercial use under an Apache-2.0 license and may require explicit attribution when utilized. Please refer to [the license](https://github.com/ProteinDesignLab/caliby/blob/main/LICENSE) for full terms.

## Overview

[Caliby](https://github.com/ProteinDesignLab/caliby) designs sequences for a fixed protein backbone with a Potts model: a graph neural network reads backbone geometry and emits an explicit pairwise energy function, which is then sampled by Monte Carlo. Because that energy is explicit, it can be averaged across a set of conformers, so a sequence is designed against a distribution of backbones rather than one static snapshot. The toolkit also scores the native sequence of a structure and packs sidechains onto a designed backbone.

## Background

Caliby ([Shuai et al., 2025](https://doi.org/10.1101/2025.09.30.679633)) starts from the hypothesis that training a sequence designer to maximize native sequence recovery rewards reproducing non-structural signal, such as phylogenetic relatedness, neutral drift, and dataset sampling bias, instead of a broadly generalizable structure-to-sequence mapping. Its response is to make the learned energy explicit. A [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)-style graph network, three encoder layers and three decoder layers with the decoder's causal mask removed, encodes the backbone, and a Potts decoder adapted from ChromaDesign ([Ingraham et al., 2023](https://www.nature.com/articles/s41586-023-06728-8)) projects the node and edge embeddings into the sitewise fields and pairwise couplings of a [Potts model](https://en.wikipedia.org/wiki/Potts_model) over amino-acid identities. Training maximizes the composite pseudo-likelihood of the native sequence; design then samples that energy by discrete Langevin Monte Carlo at a temperature annealed from 1.0, under a local composition restraint that penalizes low-complexity sequences.

An energy built only from sitewise and pairwise terms is linear in its parameters, so the average of the energies of several backbones is itself one Potts model whose fields and couplings are the per-structure averages. Those averaged parameters are computed once, which makes ensemble-conditioned sampling no more expensive than single-structure sampling, and the distribution they define is the normalized geometric mean of the per-structure distributions. The paper conditions on the input backbone plus 15 conformers drawn from Protpardelle-1c partial diffusion ([Lu et al., 2025](https://doi.org/10.1101/2025.08.18.670959)), and the reference implementation's batch pipeline defaults to 32 per ensemble. Averaging in this way suppresses the native-sequence bias carried by any single snapshot: ensemble-conditioned design lowers native sequence recovery yet substantially raises AlphaFold2 self-consistency, outperforming ProteinMPNN and ChromaDesign on both native and de novo backbones. A soluble-only variant further shows, in silico, that Protpardelle-1c binder designs previously filtered out as undesignable by SolubleMPNN ([Goverde et al., 2024](https://www.nature.com/articles/s41586-024-07601-y)) are designable, which the authors read as a limitation of the filtering pipelines rather than of the backbones. That result is computational; the paper reports no experimental validation of those binders.

That explicit, pairwise form is what makes the averaging possible, and it is the structural difference from the autoregressive decoding of ProteinMPNN and LigandMPNN or the iterative unmasking of FAMPNN, where there is no single closed-form scoring function to combine across structures. Design itself only assigns residue identities and leaves coordinates untouched, so Caliby ships a separate diffusion-based sidechain packer that predicts sidechain coordinates by denoising in each residue's local backbone frame. The reference implementation is maintained at [ProteinDesignLab/caliby](https://github.com/ProteinDesignLab/caliby) and was developed at Stanford University.

### Learning Resources

- [Caliby Python API guide](https://github.com/ProteinDesignLab/caliby/blob/main/python_api.md) (Protein Design Lab, Stanford) - the upstream programmatic interface, covering checkpoint loading, ensemble layout, positional constraints, scoring, and the input-structure cleaner.
- [Caliby Colab notebook](https://colab.research.google.com/github/ProteinDesignLab/caliby/blob/main/notebooks/Caliby.ipynb) (Protein Design Lab, Stanford) - an interactive walk-through of ensemble generation followed by ensemble-conditioned design.
- [Protpardelle-1c](https://github.com/ProteinDesignLab/protpardelle-1c) (Protein Design Lab, Stanford) - the partial-diffusion backbone model used to generate the synthetic ensembles Caliby conditions on, and the source of the binder designs re-evaluated in the paper.

## Tools

### Caliby Sampling (`caliby-sample`)

Designs sequences for one or more fixed backbones by sampling the Potts model, optionally conditioned on a conformer ensemble supplied per structure. Each design comes back as a complex with its threaded structure, its Potts energy, and its recovery against the input structure's native sequence.

#### Applications

Use this for fixed-backbone design, and in particular for backbones that are not idealized: native scaffolds, loop-rich folds, and diffusion-generated binders whose designs tend to fail refolding checks. Conditioning on an ensemble is the lever for exactly those cases, since the sequence is then scored against the conformational freedom the backbone actually has rather than one snapshot of it. For a first pass over many candidate backbones, a `_distill` checkpoint gives ensemble-conditioned behavior without the cost of generating an ensemble.

#### Usage Tips

- **`conformers` is what switches on ensemble-conditioned design.** Leave it empty for ordinary single-structure design. When it is set, `structure` acts as the primary conformer: its residue identities are the ones broadcast across the ensemble, and it is the reference frame the other members align to. Every conformer must carry the same chain IDs and the same per-chain residue counts, which is checked when the input is constructed. Upstream recommends at least 32 conformers, with 8 or 16 also giving good results.
- **Designs are backbone-only.** Caliby threads the designed residue identities onto the input coordinates without generating sidechains, so `design.structure` has no sidechain atoms. Pass it to `caliby-pack` to place them before docking, full-atom scoring, or molecular dynamics.
- **`temperature` (default `0.01`) and `potts_sweeps` (default `500`) set the diversity and convergence trade-off.** Sampling anneals from 1.0 down to `temperature`, so raising it to 0.1 or 0.2 yields more diverse sequences at some cost in energy. More sweeps converge closer to the Potts optimum at proportional runtime.
- **`model_checkpoint` selects the training distribution.** `caliby` is the default, trained on PDB chains with 0.3 Å Gaussian backbone noise, and `soluble_caliby` is its analog trained with annotated transmembrane proteins excluded; both see monomers only, so `soluble_caliby_v1`, trained on monomers and interfaces, is the better fit for interface design. The distilled `caliby_distill` and `soluble_caliby_distill` checkpoints skip ensemble generation and suit fast screening, but they are less sensitive to small backbone changes and are not recommended for multistate design.

### Caliby Scoring (`caliby-score`)

Evaluates the Potts energy of the sequence an input structure already carries, optionally against a conformer ensemble. There is no separate sequence field: the sequence is read from the structure, so scoring a variant means supplying a structure that carries it.

#### Applications

Use this to rank sequence-structure pairs under the same energy that drives design, for example comparing a design against the native sequence on the same backbone, checking how a candidate holds up across a conformational ensemble before committing to it, or choosing between checkpoints on a held-out set. `return_local_energies` additionally exposes the per-position conditional energy matrix for residue-level analysis.

#### Usage Tips

- **The Potts energy is unnormalized, not a log-likelihood.** Probability is proportional to `exp(-U/T)` for a sampling temperature `T` that scoring never fixes, so there is no perplexity to read off; the number is a ranking quantity, not a calibrated one. Lower is more favorable.
- **`potts_energy` scales with sequence length, so compare `avg_potts_energy` across structures of different sizes.** The summed energy is the right quantity only when ranking sequences on one backbone.
- **Energies are comparable only within one checkpoint.** Each checkpoint defines a different energy function, so designs scored under `caliby` cannot be ranked against designs scored under `soluble_caliby`.
- **`return_local_energies` (default `False`) adds a per-position matrix of shape `(seq_len, len(potts_tokens))`.** Read its columns through the returned `potts_tokens` ordering, which is a reduced alphabet rather than a full token vocabulary. Leave it off when scoring many structures, since it adds roughly `seq_len x 22` floats per input.
- **With an ensemble, only the primary conformer's sequence is scored.** The other members contribute backbone geometry alone; their residue identities are ignored.

### Caliby Sidechain Packing (`caliby-pack`)

Places sidechain atoms for the residue identities each input structure already carries, by diffusion in that residue's local backbone frame. Returns one packed full-atom structure per input.

#### Applications

Use this as the companion step to `caliby-sample`, whose designs carry no sidechains, and more generally to build a full-atom model from a backbone plus a known sequence before docking, full-atom scoring, or molecular dynamics. It also serves to rebuild sidechains on structures that arrive incomplete from another pipeline.

#### Usage Tips

- **Packing requires a `caliby_packer_*` checkpoint.** Packing and design are distinct model tasks trained with different objectives, which is why they are separate tools; a design checkpoint passed here is rejected during config validation, as is a packer checkpoint passed to `caliby-sample` or `caliby-score`.
- **The packer suffix is the training noise level in angstroms.** `caliby_packer_010` (0.1 Å) is the default and the upstream recommendation for most cases, `caliby_packer_000` targets noise-free backbones, and `caliby_packer_030` noisier ones.
- **`diffusion_steps` (default `50`) and `step_scale` (default `1.5`) match the reference implementation.** They control the denoising trajectory and are worth changing only for a specific packing experiment.

## Toolkit Notes
<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Caliby tool in this toolkit (`caliby-sample`, `caliby-score`, `caliby-pack`).

- **Requires a GPU.** Both the Potts network and the diffusion packer are impractical on CPU for anything beyond a smoke test.
- **Checkpoints download automatically when the tool environment is built.** They are hosted on [HuggingFace](https://huggingface.co/ProteinDesignLab/caliby-weights) under the same Apache-2.0 terms as the code, with no token or terms acceptance required. A `model_checkpoint` value ending in `.ckpt` is treated as a path to a custom local checkpoint instead of a bundled name.
- **Clean externally sourced PDB and CIF files before design.** Upstream recommends it for any structure that came out of another pipeline, and especially for ensembles: blank chain IDs, unresolved atoms, and inconsistent residue indexing are what make conformers misalign with the primary conformer, and a misalignment silently degrades the aggregated Potts parameters rather than failing loudly.
- **The ensemble is declared per structure, not per run.** `conformers` lives on each structure input, so one call can mix single-structure and ensemble-conditioned designs.
- **`batch_size` counts conformers, not designs.** It sets how many ensemble members pass through the model at once, so it applies to ensemble-conditioned design only and is the first thing to lower when such a run hits GPU out-of-memory. Designs are sampled one at a time from a single set of Potts parameters, so raising `num_sequences_per_structure` costs sampling time but no extra memory.
- **Positions are 1-indexed ordinals within each chain, not residue numbers.** Position `i` in `fixed_positions` or `pinned_residues` is the chain's `i`-th residue, whatever the file numbers it, so offset numbering, gaps and insertion codes (`10`, `10A`, `11`) all resolve correctly. Structures reach Caliby renumbered 1..N per chain, because its position syntax cannot express insertion codes; designed and packed structures come back in the input's original numbering. The same ordinal applies to every conformer, so conformers only need matching per-chain residue counts.
- **Tie the chains of a homooligomer with `symmetric_chains`.** Chains are otherwise designed independently, so the copies of a homooligomer come out with different sequences and the reported energy describes an assembly that cannot be built from one gene. `symmetric_chains=["A", "B", "C"]` folds the tied positions into a single variable before sampling, so the copies are identical by construction and the energy covers the interface couplings between them. Chains tied together must share residue numbering, insertion codes included.
- **`pinned_residues` places a residue; `fixed_positions` keeps the one already there.** `fixed_positions` conditions on whatever identity the input structure carries, so it cannot pin a residue the structure lacks. `pinned_residues={"A": {26: "W"}}` supplies the identity, writes it into the design, and conditions on it. A position cannot appear in both. Pinned positions also shift `sequence_recovery`: it is measured against the supplied identities rather than the structure's original sequence, and those positions count as recovered by construction.
- **Protein chains of at least five residues are all the model sees.** Caliby's sequence-design featurizer keeps peptide polymers of five residues or more and discards everything else, so ligands, ions, and nucleic acids never reach the model. A bound ligand is dropped silently: design succeeds, but the binding site is designed as though it were empty. A DNA or RNA chain, or a peptide shorter than five residues, is counted as a polymer chain here but dropped by Caliby, so such a structure is rejected with a message naming the mismatch. To design against a bound ligand or nucleic acid, use [LigandMPNN](../ligandmpnn/README.md) instead.
