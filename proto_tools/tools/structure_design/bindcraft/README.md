<a href="https://bio-pro.mintlify.app/tools/structure-design/bindcraft"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# BindCraft Binder Design

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

End-to-end de novo design of functional protein binders against a target by directly back-propagating through AlphaFold2. BindCraft hallucinates a binder backbone + sequence with a frozen target template, refines the sequence with ProteinMPNN, validates the redesigned complex with AlphaFold2, and filters the result through a battery of PyRosetta interface metrics — yielding accepted binders that have shown experimental hit rates of 10-100% in the original paper, with no MSA, no high-throughput screening, and no curated structural starting point required.

## Overview

BindCraft is a one-shot binder-design pipeline that treats AlphaFold2 as a differentiable structure-predictor and runs gradient descent against a binder objective (interface contacts + interface pTM + pLDDT + helicity bias + radius of gyration + …) on the binder logits while keeping the target frozen. The hallucinated binder is then sequence-refined with ProteinMPNN (which reuses the hallucinated backbone but produces a more designable sequence), revalidated with AlphaFold2 multimer, relaxed with PyRosetta, and finally scored against a ~30-metric filter set (interface dG, ΔSASA, shape complementarity, hotspot RMSD, hydrophobicity fractions, hydrogen-bond counts, secondary-structure percentages, …) before being accepted.

In the binder-design landscape this is the leading "AF2-hallucination" approach. Compared to RFdiffusion-based pipelines it does not need an inverse-folding step on a separately diffused backbone, and compared to single-pass ColabDesign hallucination it adds the MPNN+AF2+PyRosetta filtering loop that drives the published hit rates. Compared to Germinal-style differentiable design (also exposed in proto-tools as `alphafold2-binder`) it is a full end-to-end pipeline rather than a single gradient step. The trade-off is wall-clock time: a single accepted design takes ~10-30 min on an H100, and a production campaign is hours to days.

**Use cases:**

- **One-off sample (smoke test / notebook demo).** Set `number_of_final_designs=1` and `max_trajectories=1` with tiny iteration counts (`soft_iterations=10`, `temporary_iterations=5`, `hard_iterations=2`, `greedy_iterations=2`). Returns a single design (or none) in ~10-30 min on an H100. Good for verifying installation, demos, and sanity-checking a target before investing compute.
- **Quick scan / methods exploration.** A handful of designs (5-20) with a moderate trajectory cap (50-100) and medium iteration counts. Lets you compare hotspots, length ranges, helicity biases, or filter-override strategies without waiting two days.
- **Production binder-design campaign.** Upstream defaults: `number_of_final_designs=100`, `max_trajectories=False` (unlimited), full iteration counts (75/45/5/15). Produces enough diverse accepted designs to triage and order for experimental validation.
- **Targeting a defined epitope.** Provide `target_hotspot_residues` to focus the binder against specific functional residues (active site, binding pocket, catalytic loop, etc.).
- **Beta-rich targets.** Set `optimise_beta=True` (the default) to add extra hallucination iterations and AF2 recycles when a generated trajectory is beta-heavy.

**When NOT to use BindCraft:**
- You already have a PDB-bound binder that just needs sequence redesign — use `proteinmpnn-sample` instead.
- You need an antibody (CDR-only redesign on a fixed framework) — use `alphafold2-binder` with the Germinal backend, or a dedicated Ig-design pipeline.
- Your target is very large (>2000 residues) — AF2 multimer memory will become the bottleneck before BindCraft does anything useful; trim the target to its binder-accessible domain first.
- You just want a backbone, not a sequence — use `rfdiffusion3-design`.

## Background

For non-biologists, the four ideas BindCraft chains together are:

1. **Differentiable structure prediction.** AlphaFold2 maps an amino-acid sequence to a 3D structure. Because the network is differentiable, you can ask "if I nudge each amino-acid logit a little bit, how does the predicted structure (and any function of it) change?" — and you get exact gradients from the computation graph. BindCraft frames binder design as gradient descent against a sum of structural losses (good interface contacts, low PAE across the interface, high pLDDT, target hotspots within reach, …) computed on the AF2 forward pass.
2. **Hallucination.** The above process — optimising input logits against a structural objective — is called "hallucination". It produces a sequence and a backbone that, *in AF2's view*, satisfy the objective. The risk is that the hallucinated sequence may be designed *for AF2 specifically* and not actually fold the way AF2 thinks it does in the lab.
3. **ProteinMPNN refinement.** To mitigate that risk, BindCraft hands the hallucinated backbone to ProteinMPNN, an inverse-folding model trained to produce sequences that physically fold into a given backbone. The interface residues that contact the target are typically held fixed (`mpnn_fix_interface=True`); the rest of the binder is redesigned for foldability. Several MPNN sequences are sampled per trajectory.
4. **PyRosetta filtering.** Each MPNN-refined sequence is repredicted by AF2 multimer, the resulting complex is relaxed with PyRosetta, and a battery of physics-/geometry-based interface metrics (Rosetta dG, ΔSASA, shape complementarity, hotspot RMSD, hydrogen-bond satisfaction, …) is computed. A design is "accepted" only if it passes the upstream default filter thresholds (or your overrides).

## Tools

### BindCraft Binder Design (`bindcraft-design`)

Run the BindCraft binder-design pipeline against a target.

## Tool Catalog

| Tool key | Operation | Input | Output |
|---|---|---|---|
| `bindcraft-design` | Full pipeline | Target PDB + chain(s) + hotspots + binder length range | List of accepted binder designs (sequence + relaxed complex + per-design metrics) |

## Execution Modes

| Mode | `number_of_final_designs` | `max_trajectories` | Iteration counts (soft/temp/hard/greedy) | Runtime (H100) | Use case |
|---|---|---|---|---|---|
| One-off sample | 1 | 1 | 10/5/2/2 | ~10-30 min | Notebook demo, smoke test, target sanity check |
| Quick scan | 5 | 50 | 40/20/3/8 | ~2-4 h | Methods exploration, hotspot/length sweep |
| Production | 100 (default) | `False` (unlimited, default) | 75/45/5/15 (upstream defaults) | ~24-72 h | Real binder-design campaign |

GPU memory: AF2 multimer needs ~32-40 GB minimum and benefits from 80 GB for binders/targets in the 200-500 residue range. The pipeline is sequential per trajectory (`devices_per_instance=1`) — to parallelise across GPUs, run multiple `bindcraft-design` instances concurrently via `ToolPool`.

## How It Works

The pipeline is four stages per trajectory; trajectories are launched repeatedly until either `number_of_final_designs` accepted designs accumulate or `max_trajectories` is reached.

**Stage 1 — AF2 hallucination (4-stage default).** A binder of random length within `binder_lengths` is initialised with random logits, anchored next to the frozen target with optional hotspot bias, and optimised against a weighted sum of structural losses (`weights_plddt`, `weights_pae_intra`, `weights_pae_inter`, `weights_con_intra`, `weights_con_inter`, `weights_helicity`, `weights_iptm`, `weights_rg`, `weights_termini_loss`). The optimisation runs in four sub-stages — soft (`soft_iterations`), temporary (`temporary_iterations`), hard (`hard_iterations`), and greedy (`greedy_iterations`) — gradually committing the relaxed logits to discrete amino acids. `design_algorithm` selects which stages run (`"4stage"`, `"3stage"`, `"2stage"`, `"greedy"`, `"mcmc"`).

**Stage 2 — ProteinMPNN refinement.** The hallucinated binder backbone is handed to ProteinMPNN (`mpnn_weights="soluble"` by default) which samples `num_seqs` sequences at temperature `sampling_temp`. Interface residues are held fixed when `mpnn_fix_interface=True` so MPNN only redesigns the binder core/surface. The top `max_mpnn_sequences` are advanced.

**Stage 3 — AF2 validation.** Each MPNN-refined complex is repredicted by AF2 multimer with `num_recycles_validation` recycles (or the optimised-beta variants if the trajectory was beta-heavy). This is the honest test: AF2 was not trained on this sequence, so high pLDDT/iPTM/low interface PAE here is a real signal.

**Stage 4 — PyRosetta scoring & filter check.** The validated complex is relaxed with PyRosetta and scored on ~30 interface metrics (`BindCraftMetrics`). A design is *accepted* iff every metric passes the corresponding entry of the upstream `default_filters.json` (or your overrides via `filter_overrides`). Accepted designs are appended to `BindCraftOutput.designs`; trajectories that fail any filter are rejected and a new trajectory is launched.

## Input Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `target_pdb` | `str` | *required* | Target structure (file path or PDB-format string). |
| `target_chain` | `str` | `"A"` | Chain ID(s) of the frozen target (comma-separated for multi-chain). |
| `target_hotspot_residues` | `str \| None` | `None` | Comma-separated 1-indexed residue positions on the target the binder must contact. Supports ranges (e.g. `"1-10,56,78"`). `None` = unrestricted. |
| `binder_lengths` | `tuple[int, int]` | `(65, 150)` | `(min, max)` binder length range. |
| `binder_name` | `str` | `"binder"` | Project identifier; used as a prefix in output filenames. |
| `number_of_final_designs` | `int` | `100` | Target accepted-design count. Set to `1` for one-off sampling. |

## Configuration

`BindCraftConfig` exposes 51 of upstream's 61 advanced settings as typed fields. The other 10 — animations / loss-curve plots / MPNN FASTAs / trajectory pickles / zip archives / intermediate-PDB cleanup, plus the `sample_models` debug flag — are hardcoded in the dispatch payload (`_HARDCODED_INTERNAL_SETTINGS`) because proto-tools only returns accepted designs (sequence + relaxed structure + metrics) and never surfaces those scratch artifacts to the caller. User-facing defaults are byte-for-byte equal to upstream's `default_4stage_multimer.json` at the pinned commit (`7cd4ace1b7407adf66a50dfefa47de2270f5e4a9`); both invariants are guarded by tests. Many fields use `depends_on` so the UI hides knobs that the current `design_algorithm` / `enable_mpnn` / `optimise_beta` / loss-toggle / rejection-check setting renders inert. `device`, `seed`, `verbose`, and `timeout` are inherited from `BaseConfig`.

**Algorithm selection (4)**

| Field | Type | Default | Description |
|---|---|---|---|
| `design_algorithm` | `Literal["2stage", "3stage", "4stage", "greedy", "mcmc"]` | `"4stage"` | Hallucination algorithm. |
| `use_multimer_design` | `bool` | `True` | Use AF2 multimer params during hallucination. |
| `omit_AAs` | `str` | `"C"` | Amino acids to ban during design (no separator, e.g. `"C"` or `"CW"`). |
| `force_reject_AA` | `bool` | `False` | Drop any MPNN sequence containing residues from `omit_AAs` (hard reject). |

**Iteration counts (5)** — each `depends_on` `design_algorithm`

| Field | Type | Default | Used by |
|---|---|---|---|
| `soft_iterations` | `int` | `75` | 2stage / 3stage / 4stage |
| `temporary_iterations` | `int` | `45` | 3stage / 4stage |
| `hard_iterations` | `int` | `5` | 3stage / 4stage |
| `greedy_iterations` | `int` | `15` | 2stage / 4stage / greedy / mcmc |
| `greedy_percentage` | `float` (0-100] | `1.0` | 2stage / 4stage / greedy / mcmc — mutation rate as % of binder length |

**Loss weights (9)** — `weights_iptm` / `weights_rg` / `weights_termini_loss` `depends_on` their toggle

| Field | Type | Default | Description |
|---|---|---|---|
| `weights_plddt` | `float` | `0.1` | pLDDT loss weight (higher = push for more confident folding). |
| `weights_pae_intra` | `float` | `0.4` | Within-binder PAE loss weight. |
| `weights_pae_inter` | `float` | `0.1` | Binder↔target interface PAE loss weight. |
| `weights_con_intra` | `float` | `1.0` | Within-binder Cα contact loss weight (compactness). |
| `weights_con_inter` | `float` | `1.0` | Binder↔target interface contact loss weight (docking). |
| `weights_helicity` | `float` | `-0.3` | Helicity bias (negative discourages helices, positive encourages). |
| `weights_iptm` | `float` | `0.05` | Interface pTM loss weight. |
| `weights_rg` | `float` | `0.3` | Radius-of-gyration loss weight. |
| `weights_termini_loss` | `float` | `0.1` | N-/C-termini distance loss weight (cyclisable backbones). |

**Loss toggles (4)**

| Field | Type | Default | Description |
|---|---|---|---|
| `random_helicity` | `bool` | `False` | Randomise the sign of `weights_helicity` per trajectory. |
| `use_i_ptm_loss` | `bool` | `True` | Enable interface pTM loss. |
| `use_rg_loss` | `bool` | `True` | Enable radius-of-gyration loss. |
| `use_termini_distance_loss` | `bool` | `False` | Enable N-/C-termini distance loss. |

**Contact geometry (4)**

| Field | Type | Default | Description |
|---|---|---|---|
| `intra_contact_distance` | `float` (Å) | `14.0` | Within-binder contact distance cutoff. |
| `inter_contact_distance` | `float` (Å) | `20.0` | Interface contact distance cutoff. |
| `intra_contact_number` | `int` ≥1 | `2` | Number of intra-chain contacts per residue. |
| `inter_contact_number` | `int` ≥1 | `2` | Number of inter-chain contacts per residue. |

**Template masking / prediction modifiers (6)**

| Field | Type | Default | Description |
|---|---|---|---|
| `rm_template_seq_design` | `bool` | `False` | Hide target sequence from AF2 template during hallucination. |
| `rm_template_seq_predict` | `bool` | `False` | Hide target sequence from AF2 template during validation. |
| `rm_template_sc_design` | `bool` | `False` | Hide target side-chain coords from AF2 template during hallucination. |
| `rm_template_sc_predict` | `bool` | `False` | Hide target side-chain coords from AF2 template during validation. |
| `predict_initial_guess` | `bool` | `False` | Seed AF2 validation with the hallucinated complex coords. |
| `predict_bigbang` | `bool` | `False` | Seed AF2 validation with the hallucinated atom positions. |

**MPNN refinement (8)** — every field below `enable_mpnn` `depends_on` `enable_mpnn=True`

| Field | Type | Default | Description |
|---|---|---|---|
| `enable_mpnn` | `bool` | `True` | Refine each hallucinated binder with ProteinMPNN before AF2 re-validation. |
| `mpnn_fix_interface` | `bool` | `True` | Fix interface residues during MPNN redesign. |
| `num_seqs` | `int` ≥1 | `20` | MPNN sequences sampled per accepted trajectory. |
| `max_mpnn_sequences` | `int` ≥1 | `2` | Top-scoring MPNN sequences carried to AF2 validation. |
| `sampling_temp` | `float` >0 | `0.1` | MPNN sampling temperature (lower = more deterministic). |
| `backbone_noise` | `float` ≥0 | `0.0` | Std-dev of Gaussian noise on backbone before MPNN sampling. |
| `model_path` | `Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"]` | `"v_48_020"` | MPNN checkpoint (trailing digits = training noise). |
| `mpnn_weights` | `Literal["original", "soluble"]` | `"soluble"` | MPNN weight set. |

**AF2 validation / beta optimisation (7)** — every `optimise_beta_*` `depends_on` `optimise_beta=True`

| Field | Type | Default | Description |
|---|---|---|---|
| `num_recycles_design` | `int` ≥0 | `1` | AF2 recycles during hallucination. |
| `num_recycles_validation` | `int` ≥0 | `3` | AF2 recycles during validation. |
| `optimise_beta` | `bool` | `True` | Bump iterations + recycles when a trajectory looks β-heavy (>15% sheet). |
| `optimise_beta_extra_soft` | `int` ≥0 | `0` | Extra soft iterations for β-heavy designs (4stage only). |
| `optimise_beta_extra_temp` | `int` ≥0 | `0` | Extra temporary iterations for β-heavy designs (4stage only). |
| `optimise_beta_recycles_design` | `int` ≥0 | `3` | AF2 recycles during hallucination for β-heavy designs. |
| `optimise_beta_recycles_valid` | `int` ≥0 | `3` | AF2 recycles during validation for β-heavy designs. |

**Stopping / monitoring (4)** — `acceptance_rate` and `start_monitoring` `depends_on` `enable_rejection_check=True`

| Field | Type | Default | Description |
|---|---|---|---|
| `max_trajectories` | `int \| bool` | `False` | Max hallucination trajectories before stopping. `False` = unlimited; positive int = cap (`True` bool-coerces to 1). Validator rejects 0 and negatives. |
| `enable_rejection_check` | `bool` | `True` | Enable rolling acceptance-rate monitoring (stops a stalled run). |
| `acceptance_rate` | `float` (0-1] | `0.01` | Minimum design acceptance rate to keep running. |
| `start_monitoring` | `int` ≥0 | `600` | Trajectory count before acceptance-rate monitoring begins. |

**Hardcoded internal settings (10)** — not on `BindCraftConfig`; merged into the dispatch payload by `_HARDCODED_INTERNAL_SETTINGS`

| Upstream key | Hardcoded value | Why |
|---|---|---|
| `sample_models` | `True` | Standard upstream behavior (every preset uses it); flipping it is a debug knob. |
| `save_design_animations` | `False` *(override)* | proto-tools never returns animations — skip the I/O. |
| `save_design_trajectory_plots` | `False` *(override)* | proto-tools never returns loss-curve plots. |
| `save_trajectory_pickle` | `False` | Matches upstream default (debug-only artifact). |
| `save_mpnn_fasta` | `False` | Matches upstream default. |
| `zip_animations` | `False` *(override)* | No animations to zip. |
| `zip_plots` | `False` *(override)* | No plots to zip. |
| `remove_unrelaxed_trajectory` | `True` | Matches upstream default — keep scratch small. |
| `remove_unrelaxed_complex` | `True` | Matches upstream default. |
| `remove_binder_monomer` | `True` | Matches upstream default. |

**Filter overrides (1)**

| Field | Type | Default | Description |
|---|---|---|---|
| `filter_overrides` | `dict[str, Any]` | `{}` | Per-metric threshold overrides merged on top of upstream `default_filters.json`. Keys are upstream metric names (e.g. `"Average_pLDDT"`); values are filter dicts (e.g. `{"threshold": 0.85, "higher": True}`). The full upstream filter set is ~200 keys; only override what you care about. |

## Output Specification

```python
BindCraftOutput(
    designs: list[BindCraftDesign],   # one per accepted design
    n_trajectories_run: int,          # total trajectories attempted
    n_designs_accepted: int,          # equals len(designs)
    success: bool,
    execution_time: float,
    errors: list[str],
)
```

`BindCraftOutput` is also iterable, indexable, and supports `len()` over its accepted designs.

**`BindCraftDesign` fields:**

| Field | Type | Description |
|---|---|---|
| `design_name` | `str` | Unique design identifier emitted by upstream (e.g. `"binder_l60_s12345_mpnn3"`). |
| `binder_sequence` | `str` | Designed binder amino-acid sequence (1-letter codes). |
| `structure` | `Structure` | Relaxed target+binder complex; B-factors are pLDDT on the 0-100 PDB scale (`b_factor_type=PLDDT`). |
| `metrics` | `BindCraftMetrics` | Per-design averaged metrics evaluated by the filter check. |
| `seed` | `int` | Random seed of the trajectory that produced this design. |
| `interface_aas` | `dict[str, int]` | Amino-acid composition at the binder-target interface. |
| `interface_residues` | `list[int]` | 1-indexed binder residue positions at the interface. |

**`BindCraftMetrics` fields** (all per-design averages over the validation runs; primary metric: `avg_iptm`):

*AlphaFold2 confidence*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `avg_plddt` | float | 0.0-1.0 | fraction | Mean per-residue pLDDT across the complex. Higher is better. |
| `avg_ptm` | float | 0.0-1.0 | fraction | Predicted TM-score (global). |
| `avg_iptm` | float | 0.0-1.0 | fraction | Interface pTM. The headline binder-quality metric. |
| `avg_pae` | float | 0+ | Å | Mean predicted aligned error. Lower is better. |
| `avg_ipae` | float | 0+ | Å | Mean interface PAE (only between binder and target residues). |
| `avg_iplddt` | float | 0.0-1.0 | fraction | Interface pLDDT (binder residues at the interface). |
| `avg_ss_plddt` | float | 0.0-1.0 | fraction | pLDDT averaged over secondary-structure regions of the binder. |
| `avg_binder_plddt` | float | 0.0-1.0 | fraction | pLDDT of the binder alone (across whole binder). |
| `avg_binder_ptm` | float | 0.0-1.0 | fraction | pTM of the binder alone. |
| `avg_binder_pae` | float | 0+ | Å | PAE within the binder. |

*Rosetta interface energies (REU = Rosetta Energy Units)*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `binder_energy_score` | float | — | REU | Total Rosetta energy of the binder monomer. |
| `dG` | float | — | REU | Interface binding energy (more negative is better). |
| `dSASA` | float | 0+ | Å² | Buried surface area at the interface. |
| `dG_per_dSASA` | float | — | REU/Å² | Binding energy normalised by interface area. |
| `interface_sasa_pct` | float | 0-100 | percent | Fraction of binder surface buried at the interface. |
| `interface_hydrophobicity` | float | 0.0-1.0 | fraction | Hydrophobic fraction at the interface. |
| `surface_hydrophobicity` | float | 0.0-1.0 | fraction | Hydrophobic fraction over the whole binder surface. |
| `shape_complementarity` | float | 0.0-1.0 | fraction | Lawrence-Colman shape complementarity at the interface. |
| `packstat` | float | 0.0-1.0 | fraction | Rosetta interface packing statistic. |

*Hydrogen bonds*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `n_interface_hbonds` | float | 0+ | count | Number of interface hydrogen bonds. |
| `interface_hbonds_pct` | float | 0-100 | percent | Fraction of interface contacts forming H-bonds. |
| `n_interface_unsat_hbonds` | float | 0+ | count | Number of buried unsatisfied H-bond donors/acceptors. |
| `interface_unsat_hbonds_pct` | float | 0-100 | percent | Fraction of interface H-bond partners left unsatisfied. |

*Counts*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `n_interface_residues` | float | 0+ | count | Number of binder residues contacting the target. |

*Secondary structure (per-region percentages)*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `binder_helix_pct` | float | 0-100 | percent | Percent helix in the binder. |
| `binder_betasheet_pct` | float | 0-100 | percent | Percent beta-sheet in the binder. |
| `binder_loop_pct` | float | 0-100 | percent | Percent loop in the binder. |
| `interface_helix_pct` | float | 0-100 | percent | Percent of interface residues in helices. |
| `interface_betasheet_pct` | float | 0-100 | percent | Percent of interface residues in beta-sheets. |
| `interface_loop_pct` | float | 0-100 | percent | Percent of interface residues in loops. |

*RMSDs*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `hotspot_rmsd` | float | 0+ | Å | RMSD of the binder relative to the user-specified hotspots. |
| `target_rmsd` | float | 0+ | Å | RMSD of the predicted target chain to the input target. |
| `binder_rmsd` | float | 0+ | Å | RMSD of the validated binder to the hallucinated binder. |

*Clashes*

| Metric | Type | Range | Unit | Meaning |
|---|---|---|---|---|
| `unrelaxed_clashes` | float | 0+ | count | Number of steric clashes before PyRosetta relax. |
| `relaxed_clashes` | float | 0+ | count | Number of steric clashes after PyRosetta relax. |

**Supported export formats (`BindCraftOutput._export_output`):** `pdb` (directory of one PDB per design plus `stats.json`) or `json` (single bundled JSON with PDB strings inline).

## Interpreting Results

The upstream `default_filters.json` thresholds reflect what the BindCraft authors found correlates with experimental hits. Useful rules of thumb derived from the paper:

| Metric | "Good" | "Excellent" | Notes |
|---|---|---|---|
| `avg_plddt` | > 0.80 | > 0.85 | Whole-complex confidence; designs accepted at default filters typically clear 0.80. |
| `avg_iptm` | > 0.50 | > 0.70 | Headline metric. The paper shows hit rate climbs steeply above ~0.55. |
| `avg_iplddt` | > 0.70 | > 0.80 | Binder-side interface confidence. |
| `dG` | < -25 REU | < -40 REU | More-negative is better; depends on interface size. |
| `dG_per_dSASA` | < -0.010 | < -0.015 | Energy per unit area; size-independent. |
| `shape_complementarity` | > 0.55 | > 0.65 | Lawrence-Colman; >0.65 is rare but correlates with strong binders. |
| `hotspot_rmsd` | < 5 Å | < 3 Å | Did the binder actually find the hotspot you specified? |
| `n_interface_unsat_hbonds` | < 4 | < 2 | Buried unsatisfied H-bonds penalise stability. |

A *single accepted design* already passed every default filter — the metrics above describe how to rank/triage among accepted designs when ordering for experimental validation. If you're seeing zero accepted designs across many trajectories, relax the filters (see `filter_overrides`) before increasing iteration counts; the design loop is working but the gating is too strict for your target.

## Quick Start Examples

The same import block is used in all three examples:

```python
from proto_tools.tools.structure_design.bindcraft import (
    BindCraftConfig,
    BindCraftInput,
    run_bindcraft_design,
)
```

**Example 1 — One-off sample (~10-30 min on H100)**

Use this to verify the install and sanity-check a target. Tiny iteration counts and a single trajectory keep wall-clock down; expect either one design or none.

```python
inputs = BindCraftInput(
    target_pdb="path/to/target.pdb",        # e.g. PD-L1 from PDB 4ZQK, chain A
    target_chain="A",
    target_hotspot_residues="56",            # PD-L1 Y56 sits in the PD-1 binding face
    binder_lengths=(60, 70),
    binder_name="quick_test",
    number_of_final_designs=1,                # binder is always assigned chain B (upstream invariant)
)
config = BindCraftConfig(
    max_trajectories=1,
    soft_iterations=10,
    temporary_iterations=5,
    hard_iterations=2,
    greedy_iterations=2,
    num_seqs=2,
    max_mpnn_sequences=1,
    seed=42,
)

result = run_bindcraft_design(inputs, config)
print(result.designs[0].binder_sequence if result.designs else "no design accepted")
```

**Example 2 — Quick scan with a custom filter override**

A 5-design exploration with the iPTM threshold relaxed (handy if your target is hard and the default 0.55 floor never fires). `filter_overrides` only lists the keys you want to change; everything else inherits from upstream `default_filters.json`.

```python
inputs = BindCraftInput(
    target_pdb="path/to/4zqk.pdb",
    target_chain="A",
    target_hotspot_residues="54-67",         # PD-L1 residues forming the PD-1 contact patch
    binder_lengths=(70, 110),
    binder_name="pdl1_scan",
    number_of_final_designs=5,
)
config = BindCraftConfig(
    max_trajectories=50,
    soft_iterations=40,
    temporary_iterations=20,
    hard_iterations=3,
    greedy_iterations=8,
    num_seqs=8,
    max_mpnn_sequences=2,
    filter_overrides={
        "Average_i_pTM": {"threshold": 0.45, "higher": True},
        "Average_ShapeComplementarity": {"threshold": 0.55, "higher": True},
    },
    seed=0,
)

result = run_bindcraft_design(inputs, config)
print(f"{result.n_designs_accepted}/{result.n_trajectories_run} trajectories accepted")
for d in result.designs:
    print(f"  {d.design_name}  iPTM={d.metrics.avg_iptm:.3f}  dG={d.metrics.dG:+.1f}")
```

**Example 3 — Production run with custom hotspots and beta-optimisation**

The default 100-design / unlimited-trajectory production setup, with explicit beta-optimisation for a beta-rich target (e.g. an Ig fold). Expect ~24-72h on a single H100.

```python
inputs = BindCraftInput(
    target_pdb="path/to/4zqk.pdb",
    target_chain="A",
    target_hotspot_residues="54-67,113-123", # multiple hotspot patches
    binder_lengths=(80, 150),
    binder_name="pdl1_production",
    number_of_final_designs=100,             # upstream default
)
config = BindCraftConfig(
    max_trajectories=False,                  # unlimited — let it run until 100 accepted
    optimise_beta=True,                      # bump recycles & iterations on beta-heavy trajectories
    optimise_beta_extra_soft=25,
    optimise_beta_extra_temp=15,
    weights_helicity=-0.3,                   # mild anti-helix bias (default)
    seed=12345,
)

result = run_bindcraft_design(inputs, config)
result.export("pdl1_production", export_path="out", file_format="pdb")  # → out/pdl1_production/{design}.pdb + stats.json
```

## Best Practices & Gotchas

- **Length range guidance.** Upstream's default is `[65, 150]`; that range is the typical sweet spot. Below ~50 you're effectively designing a peptide and the AF2-multimer signal weakens; above ~200 GPU memory and per-trajectory time blow up.
- **Hotspot picking.** Pick *functional* residues (active site, paratope contact, catalytic loop, allosteric pocket), not arbitrary surface residues. A poor hotspot list either gets ignored (binder lands elsewhere) or kills acceptance rate (binder can't satisfy the geometry).
- **Helicity bias.** The default `weights_helicity=-0.3` is a mild *anti*-helix bias (the upstream authors found AF2 over-produces α-helical bundles). Set it positive (e.g. `+0.3`) to *encourage* helices for helix-friendly targets; set it to `0` to let AF2 do whatever it wants. `random_helicity=True` randomises the sign per trajectory for diversity.
- **Beta-rich targets.** Leave `optimise_beta=True` (default) on; consider also bumping `optimise_beta_extra_soft` and `optimise_beta_extra_temp` to give beta-heavy trajectories more time to converge.
- **GPU memory.** AF2 multimer needs ~32-40 GB minimum and benefits from 80 GB for binders/targets in the 200-500 residue range. If you OOM on long binders, shrink `binder_lengths` first; PyTorch fragmentation on AF2 multimer is hard to recover without restart.
- **PyRosetta licensing.** PyRosetta is free for academic non-profit use only. Commercial users must obtain a separate license — see the upstream BindCraft repo's setup notice and the PyRosetta licensing page (https://els2.comotion.uw.edu/product/pyrosetta).
- **First-time setup.** Downloads ~5.5 GB of AF2 weights (shared with the `alphafold2` toolkit's weights cache) plus the ColabDesign + ProteinMPNN + BindCraft repos.
- **Filter overrides escape hatch.** The full upstream `default_filters.json` is ~200 keys covering every metric in `BindCraftMetrics` plus secondary-structure/composition gates. Only override the metrics you actually care about; values you don't list keep their upstream default. Use `{"threshold": value, "higher": True/False}` for scalar thresholds, or `null` to disable the filter entirely.
- **Acceptance-rate monitoring.** With `enable_rejection_check=True` (default), the run aborts early if the rolling acceptance rate drops below `acceptance_rate=0.01` after `start_monitoring=600` trajectories. Disable it (`enable_rejection_check=False`) for stubborn targets where you want to keep grinding.
- **`max_mpnn_sequences=2` is a CPU-time floor, not a quality cap.** Each MPNN-validated sequence triggers a full AF2 multimer prediction; bumping to 5-10 helps for hard targets but multiplies validation time per trajectory.
- **Caching.** The tool is registered with `cacheable=False` because per-trajectory randomness (and the unbounded `max_trajectories`) makes naive caching unsafe; rely on `seed` + `max_trajectories` for reproducibility instead.

## References

- **Pacesa et al. (2025).** "One-shot design of functional protein binders with BindCraft." *Nature*, 646(8084), 483-492. [DOI: 10.1038/s41586-025-09429-6](https://doi.org/10.1038/s41586-025-09429-6)
- **Pacesa et al. (2024).** "BindCraft: one-shot design of functional protein binders." *bioRxiv*, 2024.09.30.615802. [DOI: 10.1101/2024.09.30.615802](https://doi.org/10.1101/2024.09.30.615802)
- **Upstream implementation (pinned commit):** [martinpacesa/BindCraft @ `7cd4ace1b7407adf66a50dfefa47de2270f5e4a9`](https://github.com/martinpacesa/BindCraft/tree/7cd4ace1b7407adf66a50dfefa47de2270f5e4a9)
- **ColabDesign (Sergey Ovchinnikov, pinned commit `e31a56fe1d9b4de25c8697f3a28b75892941cc72`):** [https://github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- **ProteinMPNN.** Dauparas, J. et al. (2022). "Robust deep learning-based protein sequence design using ProteinMPNN." *Science*, 378(6615), 49-56. [DOI: 10.1126/science.add2187](https://doi.org/10.1126/science.add2187)
- **PyRosetta.** Chaudhury, S., Lyskov, S., Gray, J. J. (2010). "PyRosetta: a script-based interface for implementing molecular modeling algorithms using Rosetta." *Bioinformatics*, 26(5), 689-691. [DOI: 10.1093/bioinformatics/btq007](https://doi.org/10.1093/bioinformatics/btq007)
- **AlphaFold2.** Jumper, J. et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596(7873), 583-589. [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

## Related Tools

- **`alphafold2-binder`** — single-pass differentiable binder design (one ColabDesign forward+backward pass returning logit gradients). Use this when you want to drive an outer optimisation loop yourself; use BindCraft when you want the full hallucination + MPNN + AF2 + PyRosetta pipeline end-to-end.
- **`proteinmpnn-sample`** / **`proteinmpnn-score`** — ProteinMPNN as a standalone tool when you already have a backbone and want sequences (or want to score an existing sequence against a backbone).
- **`pyrosetta-interface-analyzer`** / **`pyrosetta-relax`** — PyRosetta interface scoring and FastRelax exposed as standalone tools, for re-scoring or relaxing complexes outside the BindCraft loop.
- **`rfdiffusion3-design`** — alternative diffusion-based binder backbone generation. Pair with `proteinmpnn-sample` and `alphafold2-prediction` for an RFD3-style pipeline; choose between this and BindCraft based on whether you trust diffusion priors or AF2 hallucination more for your target class.
