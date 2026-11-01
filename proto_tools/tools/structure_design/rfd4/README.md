<a href="https://bio-pro.mintlify.app/tools/structure-design/rfd4"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# RFD4-Proteina

![RFD4-Proteina](https://proto-bio.github.io/proto-assets/images/tool/rfd4/hero.png)

> [!NOTE]
> **License:** RFD4-Proteina is open source and free for academic and commercial use under a BSD-3-Clause license. Please refer to [the license](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/LICENSE) for full terms.

## Overview

[RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) is a generative model from the Institute for Protein Design and NVIDIA that designs protein sequences and all-atom structures jointly. It supports unconstrained generation, motif scaffolding, binder design, enzyme design, and fixed-backbone sequence design.

Unlike workflows that generate a backbone and then design its sequence, RFD4-Proteina generates both at once. Designs can be conditioned on existing structures, amino acid identities, interaction sites, and ligands. It also supports inverse folding, which retains a backbone while generating its sequence and sidechains.

## Background

RFD4-Proteina ([Corley et al., 2026](https://rfd4-proteina.org/assets/paper/RFD4-Proteina.pdf)) uses a flow-based generative framework to model biomolecular sequences and three-dimensional structures. Its central contribution is a shared representation of molecular geometry and chemical identity, allowing sequence and structure to be generated together rather than treated as separate prediction problems.

The model uses a partially latent representation. Polymer backbones are represented by explicit anchor coordinates, such as the Cα atoms of proteins, while additional residue-level features encode chemical identity and local conformation. An associated autoencoder converts these latent features into amino acid sequences and all-atom structures. Atomized components, including small-molecule ligands, retain information about their constituent atoms and chemical connectivity.

RFD4-Proteina builds on the Proteina line of models and incorporates several architectural and training improvements. Its training data extend beyond experimentally determined structures to include predicted structures and physically simulated conformations, broadening the range of molecular systems represented during training. The model also uses more extensive pairwise representations and recurrent updates during generation.

Two approaches allow additional computation to be applied during inference. The first, termed *intrinsic reasoning* by the authors, repeatedly updates intermediate representations during individual generative steps. Increasing these recurrent updates gives the model additional opportunities to refine the evolving structure. The second, *extrinsic reasoning*, uses search over candidate generation trajectories, selecting candidates according to external structural, geometric, or energetic scores. The latter is supported upstream but is not exposed through this Proto toolkit.

The authors report improved performance across several computational protein-design benchmarks, including protein binding, small-molecule binding, enzyme design, and DNA binding. For example, the reported clustered pass rates for small-molecule and DNA binding were 14.3% and 3.5%, respectively, compared with 2.4% and 0.5% for the corresponding RFdiffusion3 comparisons. These are computational success metrics based on structural and predictive criteria, rather than direct measurements of experimental binding or activity. The paper also presents separate experimental validation of selected designs.

For practical design workflows, the principal advantage is the joint generation of sequence and structure. RFD4-Proteina produces an all-atom model with its corresponding sequence, avoiding the need to run a separate inverse-folding model simply to assign amino acid identities to a generated backbone. The resulting designs should still be evaluated independently, for example through structure prediction, sequence analysis, or experimental characterization.

The model supports a range of conditioning options, including fixed and flexible structural motifs, sequence constraints, target hotspots, ligand interactions, solvent exposure, hydrogen-bond roles, secondary structure, CATH fold classifications, and specified amino acid probabilities. These are documented in the upstream [annotations reference](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/docs/reference/annotations.md). Symmetry-conditioned generation is not currently supported.

### Learning Resources

- [RFD4-Proteina documentation](https://rfd4-proteina.org/latest/) — Official documentation covering model inputs, conditioning, generation settings, and example workflows.
- [RFD4-Proteina model card](https://huggingface.co/nvidia/RFD4-Proteina) — Model weights, checkpoint information, training details, and licensing.
- [RFD4-Proteina on NVIDIA build](https://build.nvidia.com/ipd/rfd4-proteina) — Hosted inference endpoint for running the model without managing a local GPU environment.

## Tools

### RFD4-Proteina Design (`rfd4-design`)

Generates protein sequences and all-atom structures according to the constraints specified in each `RFD4DesignSpec`. Each specification represents an independent design task and produces `num_replicates` candidate designs. Results are returned as `RFD4Designs` collections, with one collection corresponding to each input specification.

#### Applications

Use `rfd4-design` when both the protein sequence and structure are design variables. Applications include de novo protein generation, binder design against specified target sites, scaffolding of structural or functional motifs, enzyme active-site design, and generation conditioned on fold class or solvent exposure.

The tool also supports partial diffusion, allowing an existing design to be modified rather than generated entirely from scratch. Fixed-coordinate motifs can be used when their positions must be preserved, while internal-coordinate constraints allow motifs to retain their internal geometry without fixing their global orientation.

Because the model generates sequences alongside structures, designs can proceed directly to independent evaluation without a separate inverse-folding stage. A useful initial assessment is to predict the structure from the generated sequence and compare the result with the designed structure.

#### Usage Tips

- **Input structures and file formats.** The Proto implementation accepts both PDB and mmCIF structures and prepares an mmCIF input for RFD4-Proteina. The upstream implementation requires mmCIF because PDB files do not reliably preserve all of the chemical connectivity and formal-charge information required by the model. During normal input preparation, the Proto implementation also standardizes chain identifiers so selections refer to the expected chains. Setting `input_as_annotated_cif=True` bypasses this conversion and preserves the supplied mmCIF, including existing conditioning annotations. Use this option for annotated structures produced by previous RFD4 runs.

- **Residue numbering.** RFD4-Proteina resolves residue selections using the mmCIF `label_seq_id` field rather than the author-provided `auth_seq_id` numbering. By default, Proto uses `residue_numbering="auth"` and translates selections into the numbering used by the prepared input. This is convenient when working from PDB files or structures whose displayed residue numbers differ from their internal indices. Set `residue_numbering="label"` when specifying `label_seq_id` values directly. Annotated mmCIF inputs are passed through without translation, and selections supplied through `raw_conditions` are likewise not modified. Check the numbering in the actual mmCIF whenever using either option.

- **Contig specification.** The `contig` field defines which parts of an input structure are retained and where new residues may be generated. Its comma-separated syntax resembles that of RFdiffusion3: `A42-48` retains an existing segment, `20-30` defines an expandable segment, and `/0` separates output chains. RFD4 additionally supports named expandable segments such as `E1:2-20`, single-residue selections such as `A52`, and polymer-type suffixes such as `10R` for an RNA segment. Preserved residues are selected using the original input's `label_seq_id` values, and output chain assignment follows the upstream contig rules. A contig must include at least one preserved segment; fully de novo generation instead requires an appropriately prepared input structure, such as an annotated expandable-chain template. Older specification fields, including `length`, `symmetry`, `unindex`, and `select_*`, are not supported. Omitting `contig` retains the full input structure.

- **Combining conditions.** Different conditioning types can interact. Fixing sidechain coordinates also constrains the corresponding sequence identity, while atomized components such as ligands and modified residues retain their chemical identities. Within a condition, overlapping selections are applied in order, with later entries taking precedence. Absolute and internal-coordinate conditioning cannot be applied simultaneously to the same atom. To move an atom from a fixed-coordinate region into a `rigid_groups` region, first release its absolute-coordinate constraint using `free_coordinates`. Ligands and modified residues should also have valid bond orders, including explicit Kekulized aromatic bonds rather than generic `AROMATIC` bonds.

- **Recycling and inference settings.** `recycle_strategy` controls the number and distribution of recurrent model evaluations during generation. The available schedules combine a computational budget (`low`, `medium`, or `high`) with a distribution (`flat`, `midpeak`, or `front_loaded`). The `midpeak` schedules use 120, 150, or 180 trunk evaluations, while the `flat` and `front_loaded` schedules use 120, 180, or 300. The `thinking_low`, `thinking_medium`, and `thinking_high` aliases correspond to the front-loaded schedules. By default, `thinking_medium` performs 180 evaluations across 60 integration steps. All recycling presets require `nsteps=60`, and the implementation validates this combination before execution. Set `recycle_strategy=None` to disable the preset schedule and specify a different number of integration steps.

- **Choosing an inference budget.** The released recycling presets are the recommended settings for design generation. Shorter runs without recycling can be useful for verifying that an input is valid and that the execution environment works, but they should not be assumed to produce comparable design quality. In local testing, a run with `recycle_strategy=None` and `nsteps=20` produced a degenerate poly-alanine sequence. This observation does not establish that every shortened run will fail, but it provides a reason to avoid treating minimal test configurations as production settings. For less expensive generation using the supported schedules, start with a `low` preset.

- **Partial diffusion.** Partial diffusion generates a modified design by beginning from an existing structure partway through the generative process. The implementation exposes two alternative controls. `partial_trajectory_fraction_steps` specifies the fraction of the trajectory to execute, while `partial_t_backbone` and `partial_t_latent` specify the starting levels of corruption for the corresponding modalities. The direction of these controls differs: a smaller trajectory fraction preserves more of the input, whereas larger values of the `partial_t_*` controls correspond to lower corruption. The two interfaces cannot be used together. When refining an earlier RFD4 design, `input_as_annotated_cif=True` preserves its existing conditioning information.

- **Sequence probabilities.** `sequence_probabilities` allows individual residue identities to be encouraged, constrained, or excluded. For example, setting the probability of `CYS` to `0.0` explicitly excludes cysteine at the selected positions. Amino acid identities without specified probabilities remain unconstrained and share the remaining probability mass. The probabilities assigned to a residue cannot sum to more than 1.

### RFD4-Proteina Inverse Folding (`rfd4-inverse-folding`)

Generates amino acid sequences and sidechain conformations for a specified protein backbone. The backbone is held fixed during design, while the model samples compatible sequence identities and sidechain geometries. Each replicate provides a candidate sequence together with its corresponding all-atom structure.

#### Applications

Use `rfd4-inverse-folding` to design alternative sequences for an existing protein structure. Inputs may include experimentally determined structures, predicted structures, or backbones generated by tools such as `rfd4-design` and `rfdiffusion3-design`.

The tool can also be used when sidechain packing is important, since it returns sidechain coordinates together with the designed sequence. It offers an alternative to `proteinmpnn-sample`, `ligandmpnn-sample`, and `fampnn-sample`, particularly for applications that benefit from joint modeling of sequence and all-atom geometry. As with other inverse-folding methods, the resulting sequences should be evaluated before making claims about stability, expression, or function.

#### Usage Tips

- **Sequence diversity.** The `sequence_sampling.tsr_k` parameter controls the sharpness of sequence sampling. Larger values produce more concentrated sampling, while smaller values generally allow greater sequence diversity. The upstream default is `4.0`; reducing this toward `2.0` can provide more varied candidates, although their predicted quality may differ. Generate multiple replicates when exploring alternative sequences for the same backbone.

- **Sequence and structural conditions.** Inverse folding supports conditions defined in `RFD4DesignSpec`, including fixed amino acid identities through `known_sequence`, solvent-exposure constraints through `burial`, and residue exclusions through `sequence_probabilities`. Settings intended specifically for unconstrained or partially constrained structural generation, such as `guidance_w` and partial-diffusion controls, are exposed through `rfd4-design` rather than this tool.

- **Input structure completeness.** Use a structure with well-resolved coordinates and appropriate chemical information. Missing atoms or incomplete sidechains may affect how the model interprets its fixed structural context. For example, local testing with a cropped PD-L1 target derived from PDB 5O45, containing 15 unresolved sidechain atoms across six residues, produced a Cα-only poly-alanine output. The same behavior was observed with the upstream command-line implementation. This particular structure is used as a binder-design target and should not be treated as a representative inverse-folding input. Inspect structural completeness and the resulting designs before interpreting their sequences.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

The general Proto guides for tool persistence, device management, parallel execution, and cloud inference apply to both `rfd4-design` and `rfd4-inverse-folding`.

- **Symmetry support.** RFD4-Proteina does not currently support symmetry-conditioned generation. For the design of symmetric protein assemblies, use [`rfdiffusion3-design`](https://bio-pro.mintlify.app/tools/structure-design/rfdiffusion3).

- **Hardware requirements.** This toolkit requires a CUDA-enabled GPU. The upstream `rfd4-generate` implementation initializes its Lightning trainer with GPU acceleration, so selecting `device="cpu"` does not provide a CPU inference path. The environment setup also requires Linux x86_64, a CUDA-capable system, and an NVIDIA driver with a major version of at least 525.

- **Model weights and checkpoints.** The default checkpoint is available from [`nvidia/RFD4-Proteina`](https://huggingface.co/nvidia/RFD4-Proteina) without an access request. Model files are downloaded automatically when needed. The denoising model and its corresponding autoencoder must be loaded as a compatible pair. `model_revision` can be used to select a specific Hugging Face revision for reproducibility. If `adapter_path` is provided, the adapter must be compatible with the selected base checkpoint and autoencoder.

- **Model persistence.** Generation runs in-process rather than through the `rfd4-generate` command-line entry point, so a persistent worker loads the checkpoint once and reuses it. The first call in a worker pays the load; later calls do not. In local testing on an H200, three sequential designs took 37s, 13s, and 13s. Batch work therefore benefits from running inside `ToolInstance.persist_tool("rfd4-design")`. Changing `model_checkpoint`, `model_revision`, `adapter_path`, or `use_torch_compile` restarts the worker, since each of those is fixed at load time. Each worker is also pinned to its assigned GPU and is restarted rather than moved if the device changes.

- **GPU memory.** Memory requirements depend on input size, sequence length, and `batch_size`. Test the longest intended design at `batch_size=1` before increasing parallelism. If a run exceeds available GPU memory, reducing the batch size is generally the first adjustment to make.

- **Model compilation.** `use_torch_compile` is disabled by default. When enabled, the implementation compiles the model for multiple sequence-length buckets, introducing substantial initial compilation overhead. Compilation may be worthwhile for repeated or large-scale inference, but offers less benefit for short runs.

- **Evaluating designs.** RFD4 outputs do not include a calibrated confidence score indicating whether a design will fold or function as intended. Independent structure prediction, for example with [`boltz2-prediction`](https://bio-pro.mintlify.app/tools/structure-prediction/boltz2) or [`esmfold-prediction`](https://bio-pro.mintlify.app/tools/structure-prediction/esmfold), can help assess consistency between the designed sequence and generated structure. Each `RFD4Design` includes a `complex` object suitable for downstream analysis. Use `chain_sequence_map()` or `chain_sequences` to access sequences across molecular entities. `extract_protein_chains()` returns only protein chains and therefore omits DNA and RNA chains.

- **Output organization.** Results are grouped by input specification. `designed_structures` contains one `RFD4Designs` collection per specification, in the original input order. Each collection contains up to `num_replicates` designs. If a specification produces no designs, its collection remains empty rather than changing the position of subsequent results.

- **Reproducibility.** Generation is stochastic, and `seed` can be used to control random sampling. Repeated inverse-folding runs with the same seed produced identical sequences and structures in local tests. Exact reproducibility is not guaranteed across different hardware or software environments. `deterministic_sampling=True` changes both modalities to ODE integration, avoiding additional noise injection during the trajectory, but does not guarantee bitwise reproducibility in every environment. For most design applications, evaluating several independent candidates is preferable to relying on a single output.

- **Features not exposed by this toolkit.** The current implementation does not expose upstream reward-guided beam search, `rfd4-evaluate`, antibody or VHH-specific workflows, or the Diffusers interface. Reward-guided search requires additional configuration and, for some scoring functions, external structure-prediction runtimes. The toolkit also does not expose a separate folding mode. In the upstream implementation, sequence-conditioned structure generation can be expressed through sequence conditioning, which is available through `known_sequence`.

- **Metrics and output filtering.** The implementation disables tagged upstream metrics through `metrics_tags=[]`, preventing the configured AlphaFold2, AlphaFold3, and OpenFold3 refolding oracles from running. Untagged metrics, including the upstream `unknown_residue_filter`, may still be evaluated and recorded in output metadata. The implementation also sets `delete_failing_pre_oracle=False` so designs are not silently removed by that preliminary filter. These metadata should not be interpreted as independent structural validation.

- **Third-party licensing.** The upstream RFD4-Proteina distribution includes optional integrations and container environments for external structure-prediction systems. These components may be subject to licenses and usage conditions that differ from those of RFD4-Proteina itself, including restrictions on particular model weights. The Proto toolkit described here does not invoke these external refolding oracles. Users who enable them independently should review the applicable licenses and distribution terms.
