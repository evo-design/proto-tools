<a href="https://bio-pro.mintlify.app/tools/mutagenesis/random-nucleotide"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Random Nucleotide Sampling

![Random Nucleotide Sampling](https://proto-bio.github.io/proto-assets/images/tool/random_nucleotide/hero.png)

> [!NOTE]
> **License:** Random Nucleotide Sampling is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/evo-design/proto-tools) for full terms.

## Overview

Random Nucleotide Sampling fills the masked positions of a DNA or RNA sequence with random bases drawn from an IUPAC ambiguity-code pool. Positions can be marked directly with `_` or selected automatically by a masking strategy, and the substitution alphabet is set by a single IUPAC code: `N` for any base, `R` for purines, `S` for strong pairs, and so on. It runs on CPU with no model and no external dependencies, and serves as an unbiased random baseline for sequence-design and library-generation workflows.

## Background

Random Nucleotide Sampling performs random mutagenesis at the nucleotide level: it takes a DNA or RNA sequence, determines which positions are designable, and replaces each with a base drawn uniformly from a chosen IUPAC degenerate-base pool. It generates nucleotide diversity without any learned model, the simplest possible baseline against which model-guided generators can be compared.

Internally, designable positions are either the `_` characters already present in the input or, when none are present, positions chosen by the configured masking strategy. Each masked position is filled independently by drawing one base uniformly at random from the pool that the IUPAC code expands to: `N` expands to A/C/G/T, `R` to A/G, `S` to G/C, and so on. Sampling is uniform within the pool, with no frequency weighting. When the input is RNA, sampled `T` bases are converted to `U`. With a fixed seed the output is deterministic.

This tool is original proto-tools code maintained by [Proto](https://github.com/evo-design/proto-tools).

## Tools

### Random Nucleotide Sampling (`random-nucleotide-sample`)

Fills every masked position in each input sequence with a random base from the configured IUPAC substitution pool, returning one filled sequence per input.

#### Applications

Use this to build randomized nucleotide libraries: degenerate positions in promoters, ribosome binding sites, UTRs, or coding regions for directed-evolution and combinatorial-screening campaigns. It also serves as an unbiased random baseline for judging whether a model-guided generator produces better-than-chance sequences.

#### Usage Tips

- **`substitution_scheme` (default `N`) sets the substitution alphabet.** `N` allows any base for maximum diversity; restrict it to bias the library, for example `R` for purines (A/G), `S` for strong pairs (G/C), or `W` for weak pairs (A/T).
- **`_` masks override the masking strategy.** If an input already contains `_`, exactly those positions are filled and `masking_strategy` is ignored; remove the `_` characters to let the strategy choose positions instead.
- **`sequence_type` (default `auto`) controls RNA handling.** `auto` treats the sequence as RNA only when it contains `U`; force it with `dna` or `rna`. In RNA mode sampled `T` bases are written as `U`, so set `rna` explicitly when the input is fully masked.
- **`masking_strategy.fixed_positions` are 1-indexed.** Positions listed there are never mutated; they are specified using 1-based indexing to match biological residue selection conventions.
- **Set `seed` for reproducibility.** Sampling is otherwise nondeterministic; a fixed seed makes the filled sequences reproducible across runs.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Random Nucleotide Sampling tool in this toolkit (`random-nucleotide-sample`).

- **Runs on CPU.** The sampler is pure Python with no model and no external dependencies; execution is near-instant.
- **Deterministic only with a seed.** Without a `seed` the filled positions differ every run; set one when you need reproducible libraries.
