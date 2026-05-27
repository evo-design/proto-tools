<a href="https://bio-pro.mintlify.app/tools/inverse-folding/ligandmpnn"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# LigandMPNN

![LigandMPNN](https://camo.githubusercontent.com/d2e58bb7f520b9393e38084cbe2a5de762583096e74bb3a9a85057c3056c9756/68747470733a2f2f646f63732e676f6f676c652e636f6d2f64726177696e67732f642f652f32504143582d317654746e4d42444f71385470484963745566474e38566c3332783549534e63504b6c786a63514a4632713730506c61483275466c6a3241633473336b686e5a71473159787070644d72306954796b2d2f7075623f773d38383926683d333538)

> *Image source: [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)*

> [!NOTE]
> **License:** LigandMPNN is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/dauparas/LigandMPNN/blob/main/LICENSE) for full terms.

## Overview

Released in 2023, LigandMPNN is an inverse-folding model that designs a sequence for a protein backbone while explicitly accounting for the non-protein atoms around it: small-molecule ligands, nucleotides, and metal ions. It extends ProteinMPNN, which ignores those atoms, and substantially improves sequence recovery for residues that contact ligands, nucleic acids, or metals, making it a model of choice for binding-site and cofactor-aware design. It can design sequences for a backbone and score how well a sequence fits a structure.

## Background

LigandMPNN ([Dauparas et al., 2025](https://doi.org/10.1038/s41592-025-02626-1)) solves the inverse-folding problem for biomolecular assemblies: given a protein backbone together with the non-protein atoms around it, it predicts an amino-acid sequence compatible with that environment. It is a direct extension of ProteinMPNN, which sees only protein backbone atoms and is therefore blind to the bound ligands, nucleic acids, and metals that strongly shape which residues fit.

Internally, LigandMPNN keeps ProteinMPNN's message-passing design model and adds a second graph over the non-protein atoms. Residues and nearby ligand atoms exchange messages, and the model reads each atom's chemical element, which is what lets it reason about coordinating a metal or packing against a large or unusual ligand. It generates the sequence autoregressively and can also produce sidechain conformations so binding interactions can be inspected directly. On native backbones it recovers roughly 63% of the native residues that contact small molecules, 51% of those contacting nucleotides, and 78% of those coordinating metals.

The reference implementation is maintained by the [Institute for Protein Design](https://www.ipd.uw.edu/) at [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN).

### Learning Resources

- [Introducing LigandMPNN](https://www.ipd.uw.edu/2025/03/introducing-ligandmpnn/) (Institute for Protein Design) - an accessible overview of what LigandMPNN adds over ProteinMPNN and when to use it.

## Tools

### LigandMPNN Sampling (`ligandmpnn-sample`)

Designs new sequences for a backbone in the presence of its non-protein context. Each input structure is encoded once, with any ligand, nucleotide, or metal atoms included, and decoded into one or more candidate sequences with a perplexity and sequence recovery score.

#### Applications

Use this to design or redesign binding sites, enzyme active sites, nucleic-acid-binding interfaces, and metal-coordination sites, where the identity of nearby non-protein atoms determines which residues work. It is the right choice over backbone-only ProteinMPNN whenever a ligand, cofactor, nucleic acid, or metal is part of the target.

#### Usage Tips

- **Keep `ligand_mpnn_use_atom_context` enabled.** It defaults to `True` and is the whole point of LigandMPNN: it encodes the surrounding ligand, nucleotide, and metal atoms. Turning it off makes the model effectively ligand-blind, close to plain ProteinMPNN.
- **Set `ligand_mpnn_use_side_chain_context` to `True` to honor a fixed motif.** It conditions on the sidechain atoms of fixed residues, which helps when redesigning around a preserved catalytic or binding motif. It defaults to `False`.
- **`fixed_positions` is counted from 1, not 0**, to match biological residue selection conventions. Listed positions keep their input residue, and chains or atoms you do not redesign still act as context rather than being removed.

### LigandMPNN Scoring (`ligandmpnn-score`)

Evaluates how well existing sequences fit a structure and its non-protein context, returning log-likelihood-based metrics with optional per-position logits.

#### Applications

Use this to rank designs or assess mutations near ligands, nucleic acids, or metals, where backbone-only scoring would miss the very interactions that matter. Lower perplexity indicates a better fit to the structure and its bound environment.

#### Usage Tips

- **`scoring_mode` changes what the score means.** `single_aa` (the default) scores each position from its own conditional probability and is order-independent, which is what you usually want for ranking. `autoregressive` scores along one seed-determined decoding order, so it depends on the seed.
- **`fixed_positions` excludes residues from the aggregate score.** Set it per (sequence, structure) input pair as a `{chain: [positions]}` selection counted from 1, not 0, to match biological residue selection conventions, so the score reflects only the residues you care about.
- **`return_logits` (default `False`) has a size trade-off.** Enabling it adds a per-position logit array per sequence for residue-level analysis, which dominates output size and memory for long sequences, so leave it off unless you need it.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every LigandMPNN tool in this toolkit (`ligandmpnn-sample`, `ligandmpnn-score`).

- **A GPU is recommended.** LigandMPNN is a small message-passing model that also runs on CPU, but a GPU is much faster when designing or scoring many sequences.
- **The non-protein context must be in the input structure.** LigandMPNN only conditions on ligands, nucleotides, or metals that are present in the supplied structure; if they are absent, it behaves like backbone-only ProteinMPNN.
