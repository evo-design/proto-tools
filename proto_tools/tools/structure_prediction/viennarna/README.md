<a href="https://bio-pro.mintlify.app/tools/structure-prediction/viennarna"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# ViennaRNA

![ViennaRNA](https://proto-bio.github.io/proto-assets/images/tool/viennarna/hero.png)

> [!NOTE]
> **License:** ViennaRNA is licensed under Custom (ViennaRNA Package License) and may require explicit attribution when utilized. Please refer to [the license](https://github.com/ViennaRNA/ViennaRNA/blob/master/COPYING) for full terms.

## Overview

First released in 1994, ViennaRNA is a thermodynamic RNA secondary-structure prediction package. From an RNA or DNA sequence it computes the minimum-free-energy secondary structure and its free energy using a nearest-neighbor energy model, with no training and no GPU required. It is widely used to predict and compare the base-pairing of messenger RNAs, non-coding RNAs, riboswitches, and designed RNA constructs.

## Background

ViennaRNA ([Lorenz et al., 2011](https://doi.org/10.1186/1748-7188-6-26)) predicts the secondary structure of a nucleic-acid sequence: the set of intramolecular base pairs that form within a single strand. Secondary structure sits between sequence and three-dimensional shape and governs how many functional RNAs behave, so predicting it from sequence alone is a core step in RNA analysis and design.

Internally, ViennaRNA folds each sequence with the minimum-free-energy dynamic program at the core of RNAfold (cubic in the sequence length) under a nearest-neighbor thermodynamic model. RNA sequences use the Turner 2004 parameters. Selecting the DNA option instead loads the Mathews 2004 DNA parameters. The predicted structure is returned in dot-bracket notation together with its minimum free energy in kcal/mol, where a more negative value indicates a more stable predicted fold. Because it is a thermodynamic rather than a learned method, it is deterministic, runs on CPU, and predicts secondary structure only, not three-dimensional coordinates.

The reference implementation is the ViennaRNA Package, maintained by [TBI Vienna](https://www.tbi.univie.ac.at/) at [ViennaRNA/ViennaRNA](https://github.com/ViennaRNA/ViennaRNA).

### Learning Resources

- [ViennaRNA Package documentation and tutorials](https://www.tbi.univie.ac.at/RNA/) (TBI Vienna) - the official documentation, worked tutorials, and the RNAfold web server for trying predictions interactively.

## Tools

### ViennaRNA Secondary Structure Prediction (`viennarna-prediction`)

Folds each input sequence to its minimum-free-energy secondary structure, returning the structure in dot-bracket notation and the minimum free energy in kcal/mol for every sequence.

#### Applications

Use this to predict the base-pairing of mRNAs, non-coding RNAs, riboswitches, aptamers, or designed RNA constructs from sequence alone, for example to check whether a designed UTR or guide RNA folds as intended, or to rank candidate sequences by the stability of their predicted fold.

#### Usage Tips

- **`temperature` (default `37.0`, degrees Celsius) sets the folding temperature.** The energy model is temperature-dependent, so the predicted structure and free energy change with it. Keep the default for physiological predictions and change it to model other conditions.
- **`use_dna_params` (default `False`) also changes how the input is read.** When `False`, any `T` in a sequence is converted to `U` and the sequence is folded as RNA with the Turner 2004 parameters. Set it `True` to fold the sequence as DNA with the Mathews 2004 DNA parameters and no `T`-to-`U` conversion.
- **Set `circ` to `True` for circular molecules.** Plasmids, viroids, and circular RNAs fold differently from linear strands, and the default treats the sequence as linear.
- **`no_lonely_pairs` (default `False`) forbids isolated base pairs.** Enabling it removes length-one helices, which often yields more physically realistic structures.
- **`dangles` (default `2`) sets dangling-end energy treatment.** Choose `0` to ignore dangling ends, `1` for minimal, `2` for multibranch, or `3` for the most accurate model.
- **`max_bp_span` (default `-1`, unlimited) caps the base-pair span.** Set a positive value to forbid long-range pairs, which is useful for very long sequences or local-structure analysis.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ViennaRNA tool in this toolkit (`viennarna-prediction`).

- **Runs on CPU.** ViennaRNA is a fast C package and does not use a GPU. Folding is near-instant for typical sequences, and runtime grows as the cube of sequence length, so very long inputs are slower.
- **Predicts secondary structure only.** The output is a base-pairing pattern and a free energy, not three-dimensional atomic coordinates; use a tertiary-structure method when 3D is needed.
