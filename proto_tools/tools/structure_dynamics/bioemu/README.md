<a href="https://bio-pro.mintlify.app/tools/structure-dynamics/bioemu"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# BioEmu

> [!NOTE]
> **License:** BioEmu is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/microsoft/bioemu/blob/main/LICENSE) for full terms.

## Overview

BioEmu is a generative deep learning model from Microsoft Research that samples the [equilibrium](https://en.wikipedia.org/wiki/Boltzmann_distribution) conformational ensemble of a protein from its sequence. Rather than predicting a single static structure, it draws many independent backbone conformations that approximate the distribution of states a protein populates in solution, providing a fast alternative to [molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) for surveying [conformational flexibility](https://en.wikipedia.org/wiki/Protein_dynamics). This tool implementation exposes a single operation that samples ensembles for one or more monomeric protein sequences.

## Background

A protein in solution is not a single fixed shape. It fluctuates among many [conformations](https://en.wikipedia.org/wiki/Protein_dynamics), and this flexibility underlies catalysis, [allosteric regulation](https://en.wikipedia.org/wiki/Allosteric_regulation), and molecular recognition. Characterizing this ensemble experimentally is difficult, and physically simulating it with [molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) is accurate but computationally demanding, since the timescales of biologically relevant motions can require enormous amounts of simulation.

BioEmu ([Lewis et al., 2025](https://doi.org/10.1126/science.adv9817)) approaches the problem with a [diffusion-based generative model](https://en.wikipedia.org/wiki/Diffusion_model) that learns to emulate protein equilibrium ensembles directly. Starting from noise, the model iteratively denoises protein backbone coordinates conditioned on a sequence embedding, producing thousands of statistically independent structures per hour on a single [graphics processing unit](https://en.wikipedia.org/wiki/Graphics_processing_unit). The published model was trained on a large corpus of molecular dynamics simulation alongside static structures and experimental protein stability measurements, and it reproduces functional motions such as cryptic pocket formation, local unfolding, and domain rearrangements while approximating relative free energies. The conditioning sequence embedding is derived from a [multiple sequence alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment), so each sequence is first searched against sequence databases to assemble its alignment.

### Learning Resources

- [BioEmu repository](https://github.com/microsoft/bioemu) (Microsoft Research) - the reference implementation, model checkpoints, and usage examples.

## Tools

### Conformational Ensemble Sampling (`bioemu-sample`)

Samples a conformational ensemble of protein backbone structures for one or more single-chain protein sequences. Each sequence yields an independent ensemble whose members represent distinct conformations drawn from the model's learned equilibrium distribution.

#### Applications

- Surveying the conformational flexibility of a protein, including the relative populations of folded and alternative states.
- Revealing functional motions such as cryptic pocket opening, local unfolding, and domain rearrangements that a single predicted structure does not show.
- Generating a structural ensemble for downstream analysis such as clustering into metastable states or estimating per-residue flexibility.

#### Usage Tips

- **The input must be a single-chain monomer of standard amino acids.** Multi-chain complexes, non-protein chains, and non-standard residues are rejected, and sequences beyond roughly 500 residues raise a warning because quality and cost both degrade with length.
- **`num_samples` sets the size of the ensemble.** A few tens of samples give a quick read on conformational diversity, while several hundred or more give the coverage needed to estimate state populations or free-energy differences.
- **`filter_samples` removes unphysical structures.** Leaving it enabled drops samples with steric clashes or broken chain geometry, so the returned ensemble may hold fewer structures than `num_samples` requested. Disabling it returns the raw samples for inspection.
- **`model_name` selects the checkpoint.** The default `bioemu-v1.1` matches the published Science paper. `bioemu-v1.2` is trained on additional molecular dynamics and folding free-energy data and is preferable when folding-state thermodynamics matter, while `bioemu-v1.0` reproduces the earlier preprint.
- **`denoiser_config` enables physical steering.** Pointing it at a steering configuration biases sampling toward more physically plausible structures and overrides `denoiser_type`, which otherwise selects the deterministic `dpm` or stochastic `heun` sampler.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

- **A multiple sequence alignment is always required.** Each sequence is searched against the ColabFold MMseqs2 server during preprocessing to build its alignment, unless an alignment is supplied directly on the input, so network access is needed when alignments are not provided.
- **Sampling is stochastic and seeded.** Results depend on the configured seed, so repeating a run with the same seed reproduces the ensemble while changing it explores new conformations.
- **Output is backbone only and runs on GPU.** The model returns backbone coordinates without side chains and requires a CUDA GPU, since diffusion sampling is impractical on CPU.
