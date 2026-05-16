<a href="https://bio-pro.mintlify.app/tools/binder-design/germinal"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Germinal

![Germinal](https://arcinstitute.org/news/germinal/germinal.jpg)

> *Image source: [Arc Institute](https://arcinstitute.org/news/building-germinal-for-ai-designed-antibody-molecules)*

> [!NOTE]
> **License:** Germinal uses Apache-2.0 for code and CC-BY-4.0 for model weights and has restrictions around commercial use and may require explicit attribution when utilized. Please refer to the [code license](https://github.com/SantiagoMille/germinal/blob/main/LICENSE) and [model weights license](https://github.com/google-deepmind/alphafold#model-parameters-license) for full terms. This tool also bundles dependencies with their own license terms: [PyRosetta](https://bio-pro.mintlify.app/tools/structure-scoring/pyrosetta), [IgLM](https://github.com/Graylab/IgLM); review each before commercial or redistribution use.

## Overview

Germinal is a complete pipeline for de novo, epitope-targeted antibody design (single-domain VHHs and scFvs), from [Mille-Fragoso et al., 2025](https://www.biorxiv.org/content/10.1101/2025.09.19.677421). This toolkit wraps it as one tool, `germinal-design`, that runs a full design campaign against one target per call: AF2-Multimer hallucination, AbMPNN sequence redesign, and structure validation with PyRosetta interface filtering.

## Background

Germinal produces epitope-targeted antibody binders computationally from a target structure and an epitope definition, an alternative to animal- or library-based discovery such as immunization, phage display, and hybridoma screening. The Germinal publication reports experimental binding success rates of 4 to 22 percent across the benchmarks it evaluates.

Germinal combines [ColabDesign](https://github.com/sokrypton/ColabDesign) with AlphaFold2-Multimer hallucination, antibody language-model gradients ([IgLM](https://github.com/Graylab/IgLM), [AbLang](https://github.com/oxpig/AbLang)), AbMPNN sequence redesign ([Dreyer et al., 2023](https://arxiv.org/abs/2310.19513)), and structure validation against [Chai-1](https://github.com/chaidiscovery/chai-lab), [AlphaFold3](https://github.com/google-deepmind/alphafold3), or [Protenix](https://github.com/bytedance/Protenix), followed by [PyRosetta](https://www.pyrosetta.org/) interface scoring and a multi-stage filter cascade. Relative to earlier antibody hallucination methods, Germinal additionally applies epitope-hotspot conditioning (the optimization is constrained so the binder contacts the user-specified residues), antibody language-model guidance (biasing designs toward sequences resembling natural antibodies), an early filtering stage that discards weak candidate designs before the computationally expensive structure-prediction step, and a structure-validation model that is independent of the model used during hallucination.

## Tools

### Germinal Antibody Design (`germinal-design`)

Runs one complete Germinal antibody-design campaign against a single target. Given a target PDB, a target chain, and the epitope hotspot residues, it runs a fixed (version-pinned) copy of the upstream `run_germinal.py` script, repeating hallucination, then AbMPNN redesign, then structure validation and filtering until either `max_trajectories` or `max_passing_designs` is reached, and returns ranked designs with predicted complex structures and per-design metrics (interface pTM, pAE, pDockQ2, pLDDT, and others). Upstream configuration defaults are preserved exactly; the only change is setting the structure-validation model (`structure_model`) to Chai-1, because Chai-1 installs automatically.

#### Applications

This tool performs de novo therapeutic antibody discovery: generating epitope-targeted VHH or scFv binders against a chosen target. It requires only a target structure and an epitope definition, and produces a ranked set of designs with predicted complex structures and per-design quality metrics, ready for selection and experimental testing.

#### Usage Tips

- **`design_type` selects the run preset.** `"vhh"` (single-domain nanobody, the default) or `"scfv"`; each loads the upstream preset with different filter thresholds, so leave the `*_threshold` fields set to `None` to let the correct preset apply.
- **Reduce `max_trajectories` when testing.** The upstream default of `10000` corresponds to a run that takes hours to days; use a small value while testing, and set `max_passing_designs` below `max_trajectories` to stop early once enough designs pass the filters.
- **`structure_model` selects the structure-validation model.** `"chai"` (the default) installs automatically; `"af3"` and `"protenix"` require model weights and environments that you must install yourself. The published filter thresholds are calibrated against AlphaFold3, so the thresholds may need adjusting when using Chai-1 to match the reported acceptance rates.
- **`germinal_overrides` passes additional settings to the underlying pipeline.** Any upstream configuration option that is not exposed as a dedicated field can be supplied here as a `<key>=<value>` pair.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Germinal tool in this toolkit (`germinal-design`).

- **Requires a GPU and can run for a long time.** An NVIDIA GPU with at least 40 GB of GPU memory is required (at least 80 GB for scFv mode or targets longer than 250 residues); running on CPU is not supported. A full run with default settings takes hours to days, so reduce `max_trajectories` when testing.
- **Structure-validation model setup varies.** `structure_model="chai"` needs no manual setup. `"af3"` requires AlphaFold3 weights that you must request and install yourself (access is restricted; request it through DeepMind's form) along with a container image. `"protenix"` requires a separate Protenix environment.
- **Bundled dependencies carry their own licenses.** The pipeline requires PyRosetta (free for academic, non-profit, and government use; commercial use requires a separate license from UW CoMotion; redistribution is not permitted) and IgLM (academic, non-commercial use only); AbLang2 and Chai-1 are Apache-2.0. See the License note above and the linked terms.
- **One campaign per call.** Each call is a single complete run against one target. To screen several targets, call the tool once per target (for example, in a loop) rather than expecting one call to process multiple targets at once.
