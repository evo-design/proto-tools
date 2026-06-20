<a href="https://bio-pro.mintlify.app/tools/mutagenesis/random-protein"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Random Protein Sampling

![Random Protein Sampling](https://proto-bio.github.io/proto-assets/images/tool/random_protein/hero.png)

> [!NOTE]
> **License:** Random Protein Sampling is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/evo-design/proto-tools) for full terms.

## Overview

Random Protein Sampling fills masked positions in protein sequences with amino acids drawn from a codon scheme. Positions can be marked directly with `_` or selected automatically by a masking strategy. The codon scheme sets amino-acid frequencies: `UNIFORM` weights all twenty equally, while degenerate schemes such as `NNK` or `NDT` weight each amino acid by how many codons encode it. Optional amino-acid exclusions remove unwanted residues such as cysteine. It runs on CPU with no model and no external dependencies.

## Background

Random Protein Sampling performs random mutagenesis at the protein level: it takes a protein sequence, determines which positions are designable, and replaces each with an amino acid sampled from the distribution implied by a codon scheme. It generates protein-sequence diversity without any learned model, the simplest possible baseline against which model-guided designers can be compared.

Internally, designable positions are either the `_` characters already present in the input or, when none are present, positions chosen by the configured masking strategy. The codon scheme is expanded to its concrete codons, and each amino acid's sampling weight is set proportional to the number of codons in the scheme that encode it, with stop codons excluded by default. `UNIFORM` instead assigns equal weight to all twenty standard amino acids. If `excluded_amino_acids` is set, those residue types are removed after codon-scheme and stop-codon handling; the sampler raises an error if no reachable residue remains. Each masked position is filled independently by a weighted random draw. With a fixed seed the output is deterministic.

This tool is original proto-tools code maintained by [Proto](https://github.com/evo-design/proto-tools).

## Tools

### Random Protein Sampling (`random-protein-sample`)

Fills every masked position in each input sequence with a random amino acid drawn from the configured codon scheme, returning one filled sequence per input.

#### Applications

Use this to build randomized protein libraries that mimic experimental degenerate-codon mutagenesis, for example `NNK` saturation at chosen positions for directed-evolution and combinatorial screening. It also serves as an unbiased random baseline for judging whether a model-guided designer beats chance.

#### Usage Tips

- **`codon_scheme` (default `UNIFORM`) sets the amino-acid distribution.** `UNIFORM` draws all twenty amino acids equally; degenerate schemes (`NNK`, `NNS`, `NDT`, `DBK`, `NRT`) weight each amino acid by how many of the scheme's codons encode it, so residues such as leucine, serine, and arginine appear more often than methionine or tryptophan.
- **`NDT` gives an even 12-amino-acid library.** It encodes twelve amino acids with no codon redundancy, so each is equally likely; useful for small focused libraries.
- **Stop codons are excluded by default.** Set `allow_stop_codons` to `True` to include the stop symbol `*` in the distribution: for degenerate schemes it is weighted by its stop-codon count, and for `UNIFORM` it is an equally weighted 21st symbol.
- **`excluded_amino_acids` removes residues from sampling.** For example, set `["C"]` to avoid cysteine or `["C", "A"]` to remove cysteine and alanine from every masked position. An empty list is treated like `None`; excluding every residue reachable by the selected `codon_scheme` raises an error.
- **`_` masks override the masking strategy.** If an input already contains `_`, exactly those positions are filled and `masking_strategy` is ignored; remove the `_` characters to let the strategy choose positions instead.
- **`masking_strategy.fixed_positions` are 1-indexed.** Positions listed there are never mutated; they are specified using 1-based indexing to match biological residue selection conventions.
- **Set `seed` for reproducibility.** Sampling is otherwise nondeterministic; a fixed seed makes the filled sequences reproducible across runs.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Random Protein Sampling tool in this toolkit (`random-protein-sample`).

- **Runs on CPU.** The sampler is pure Python with no model and no external dependencies; execution is near-instant.
- **Deterministic only with a seed.** Without a `seed` the filled positions differ every run; set one when you need reproducible libraries.
