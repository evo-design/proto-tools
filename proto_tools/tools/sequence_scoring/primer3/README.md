<a href="https://bio-pro.mintlify.app/tools/sequence-scoring/primer3"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Primer3

> [!NOTE]
> **License:** Primer3 has a GPL-2.0 license. Please refer to [the license](https://github.com/libnano/primer3-py/blob/master/LICENSE) for full terms.

## Overview

[primer3-py](https://github.com/libnano/primer3-py) is a Cython binding to the [Primer3](https://primer3.org/) C library for oligonucleotide thermodynamics. This tool wraps its nearest-neighbor calculators to score a DNA oligo (and an optional partner) for melting temperature, hairpin and dimer stability, GC content, and 3' GC-clamp — the core filters used to judge whether a primer is fit for [qPCR](https://en.wikipedia.org/wiki/Real-time_polymerase_chain_reaction) and general PCR.

## Background

Primer3 ([Untergasser et al., 2012](https://doi.org/10.1093/nar/gks596)) is the de-facto standard engine for primer design and evaluation. Its thermodynamic calculations use the [nearest-neighbor model](https://en.wikipedia.org/wiki/Nucleic_acid_thermodynamics), which predicts duplex stability from the stacking energies of adjacent base pairs rather than a naive base count, and applies salt and concentration corrections so that the predicted [melting temperature](https://en.wikipedia.org/wiki/Nucleic_acid_thermodynamics#Melting_temperature) reflects the actual reaction buffer. The free energy (ΔG) of a self-folded hairpin, a self-dimer (homodimer), or a cross-dimer between two oligos (heterodimer) is reported in kcal/mol: a more negative ΔG means a more stable — and therefore more problematic — secondary structure that competes with productive priming.

For a primer to amplify cleanly, its melting temperature must sit in a workable band, its two ends should not fold back on themselves or pair with a partner, and its 3' end should anchor stably to the template. These constraints are what the metrics below quantify. A ΔG of `0.0` with the corresponding structure flag `False` is the favorable case: no significant structure was found. Thermodynamic conditions (monovalent and divalent cation, dNTP, and oligo concentrations) shift every prediction, so they are exposed as configuration and default to Primer3's own defaults for reproducibility against the upstream tool.

### Learning Resources

- [primer3-py documentation](https://libnano.github.io/primer3-py/) (libnano) - API reference for the `calc_tm`, `calc_hairpin`, `calc_homodimer`, and `calc_heterodimer` functions wrapped here, including every thermodynamic parameter.
- [Primer3 manual](https://primer3.org/manual.html) (Untergasser, Rozen, et al.) - the authoritative description of Primer3's parameters, salt-correction formulas, and design logic.
- [Top ten pitfalls in quantitative real-time PCR primer/probe design](https://www.thermofisher.com/us/en/home/references/ambion-tech-support/rtpcr-analysis/general-articles/top-ten-pitfalls-in-quantitative-real-time-pcr-primer.html) (Thermo Fisher) - practical target ranges and failure modes for qPCR assay design.

## Tools

### Primer3 Thermodynamics (`primer3-thermodynamics`)

Scores each input DNA oligo for melting temperature, hairpin/homodimer ΔG, GC content, and 3' GC-clamp, plus heterodimer ΔG against an optional partner oligo.

#### Applications

Use this to screen candidate PCR and qPCR primers before ordering them, or as the scoring step inside a primer-selection pipeline. Pair a forward primer with its reverse as its `partner` to check the primer *pair* for cross-dimerization, the most common cause of a failed or noisy amplification.

#### Usage Tips

- **For qPCR, aim for Tm 58–62 °C, GC 40–60%, and a GC clamp.** Keep the two primers of a pair within ~1 °C of each other. `gc_clamp=True` (a G or C in the last two 3' bases) helps 3' anchoring, but avoid more than three G/C in the last five bases to prevent mispriming.
- **Treat ΔG thresholds as guidelines: hairpin > −2 kcal/mol, homodimer and heterodimer > −6 kcal/mol.** More negative values indicate stable competing structures. Because ΔG depends on temperature, set `temp_c` to your annealing temperature (default 37 °C) for the most relevant hairpin/dimer numbers.
- **Defaults match primer3-py, not a qPCR preset.** `dv_conc=1.5`, `dntp_conc=0.6`, and `dna_conc=50` reproduce Primer3 directly. Typical qPCR conditions are closer to `dv_conc≈3`, `dntp_conc≈0.8`, `dna_conc≈200–250`; set them explicitly to match your master mix, since they shift Tm and every ΔG.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to the Primer3 tool in this toolkit (`primer3-thermodynamics`).

- **Runs on CPU, no model weights.** primer3-py compiles the Primer3 C library into its wheel, so a single `pip install` provides everything; there is no GPU path and nothing to download at first use.
- **Inputs are strictly A/C/G/T.** Sequences are uppercased and validated; degenerate bases (`N`, IUPAC ambiguity codes) are rejected because the nearest-neighbor model needs concrete bases.
- **Scoring is per-oligo and batchable.** Pass a list of oligos to score them in one call; results are returned in input order. Bundle a `partner` with an oligo to compute its heterodimer ΔG.
