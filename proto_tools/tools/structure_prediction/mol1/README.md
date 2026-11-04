<a href="https://bio-pro.mintlify.app/tools/structure-prediction/mol1"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# Mol-1

> [!NOTE]
> **License:** Mol-1 is open source and free for academic and commercial use under an Apache-2.0 license. Please refer to [the license](https://github.com/molence/Mol-1/blob/main/LICENSE) for full terms.

## Overview

[Mol-1](https://github.com/molence/Mol-1) is Molence Bio's all-atom model for predicting biomolecular complexes containing proteins, DNA, RNA, and ligands. This toolkit exposes structure prediction with optional protein alignments and returns one ranked structure per complex with confidence metrics.

## Background

Mol-1 ([Molence Bio, 2026](https://github.com/molence/Mol-1/blob/main/assets/report.pdf)) jointly predicts the coordinates of interacting molecular components. Its implementation derives substantially from Protenix, as documented in the upstream [third-party notices](https://github.com/molence/Mol-1/blob/main/THIRD_PARTY_NOTICES.md). The model refines representations of the input through recycling, then samples atomic coordinates with a diffusion process. Several candidate structures can be generated from the same complex and ranked by the model's confidence.

Protein multiple-sequence alignments (MSAs) add evolutionary information to the input. The returned confidence measures describe complementary aspects of a prediction: pLDDT estimates local reliability, pTM summarizes overall structural confidence, ipTM summarizes interface confidence, and global predicted distance error (gPDE) measures uncertainty in relative placement. Use these measures to prioritize structural hypotheses for downstream checks and experiments.

### Learning Resources

- [Mol-1 technical report](https://github.com/molence/Mol-1/blob/main/assets/report.pdf) (Molence Bio) - model background and evaluations across antibody-antigen, protein-ligand, peptide, and molecular-glue complexes.
- [Confidence output reference](https://github.com/molence/Mol-1/blob/main/docs/output_format.md) (Molence Bio) - score definitions, scales, and token-level confidence indexing.
- [Inference memory guide](https://github.com/molence/Mol-1/blob/main/docs/inference_memory.md) (Molence Bio) - measured memory use and the effects of chunking.

## Tools

### Mol-1 Structure Prediction (`mol1-prediction`)

`run_mol1_prediction` accepts `Mol1Input` and an optional `Mol1Config`, and returns `Mol1Output` containing the highest-ranked `Structure` for each input complex. Each structure carries `Mol1Metrics`.

#### Applications

Predict protein assemblies, protein-nucleic-acid interactions, and ligand binding poses within mixed molecular complexes. The shared `Complex` and `Chain` inputs also support residue modifications, allowing modified polymers to be modeled alongside their interaction partners.

#### Usage Tips

- **`num_samples=5` returns the best candidate.** Five diffusion samples are generated per complex and the structure with the highest `ranking_score` is retained. Increasing this count explores more candidates and increases runtime; the other samples are discarded.
- **`use_msa=True` generates protein alignments.** Protein MSAs are obtained with MMseqs2 and require access to the configured search service or database. Supplied `msas` take precedence, including when `use_msa=False`; supply one bundle per complex. Set `use_msa=False` without supplied alignments for single-sequence prediction.
- **`num_cycles=20` and `num_steps=200` control refinement.** Cycles recycle the model trunk; steps control diffusion denoising. Both increase runtime when raised.
- **Set `seed` for reproducibility.** The proto-tools default `seed=None` draws a fresh random seed; the upstream CLI defaults to `101`. Pass an explicit value such as `seed=101` to repeat a seeded run in the same execution environment.
- **`chunk_size=1024` controls memory use.** Smaller chunks can reduce peak memory at additional runtime cost. `sample_diffusion_chunk_size=1` denoises candidates sequentially; raising it processes more samples together and can require more memory. `dtype="bf16"` is the default CUDA precision.
- **Confidence scales differ.** `avg_plddt` is on a 0–100 scale; `chain_plddt` is on a 0–1 scale. `ptm` and `iptm` are on a 0–1 scale, with `iptm=0` for a single chain. `gpde` is in angstroms; `has_clash` flags severe inter-polymer clashes. The primary metric, `ranking_score`, includes a clash penalty.
- **`include_pae_matrix=True` adds token-level uncertainty.** It attaches `pae` and `avg_pae` in angstroms for the selected structure. Ligands and modified residues can use multiple tokens, so this matrix need not have one row per residue; its storage grows quadratically with token count.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Mol-1 tool in this toolkit (`mol1-prediction`).

- **Managed environment and assets.** First inference builds an isolated environment and downloads the bundled `mol-1-base` checkpoint plus common reference data, about 2.5 GB compressed in total. Downloads use the managed model cache (`PROTO_MODEL_CACHE`, or `PROTO_MOL1_WEIGHTS_DIR` for this toolkit); extracted files and the environment require additional disk space. Pass a custom checkpoint path through `model_checkpoint` when running on a host that has that file. Checkpoint contents are not hashed into the result-cache key, and persistent workers keep loaded weights. Use a new checkpoint path after changing weights.
- **NVIDIA GPUs are preferred.** Memory depends strongly on token count and chunking. This wrapper accepts CPU and CUDA devices; CPU execution has not been validated here.
- **Prediction-only scope.** This wrapper disables templates and RNA MSA search, so nucleic-acid chains use their supplied sequences. Upstream constraint-guided modeling and fine-tuning are outside this toolkit's exposed API.
- **License scope.** The repository declares Apache-2.0 for its code; no separate checkpoint license is stated in the upstream materials. Reference data retain their publishers' terms, and third-party components retain their own notices. Consult the upstream [third-party notices](https://github.com/molence/Mol-1/blob/main/THIRD_PARTY_NOTICES.md) before redistributing those assets.
