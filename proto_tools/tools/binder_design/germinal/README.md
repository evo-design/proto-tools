<a href="https://bio-pro.mintlify.app/tools/binder-design/germinal"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Germinal

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

Germinal is an end-to-end pipeline for de novo, epitope-targeted antibody design — both single-domain VHHs (nanobodies) and scFvs — introduced by [Mille-Fragoso et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.09.19.677421). It combines [ColabDesign](https://github.com/sokrypton/ColabDesign) + AlphaFold2-Multimer hallucination with antibody language model gradients ([IgLM](https://github.com/Graylab/IgLM), [AbLang](https://github.com/oxpig/AbLang)), [AbMPNN](https://arxiv.org/abs/2310.19513) sequence redesign, and downstream structure validation against [Chai-1](https://github.com/chaidiscovery/chai-lab), [AlphaFold3](https://github.com/google-deepmind/alphafold3), or [Protenix](https://github.com/bytedance/Protenix) — followed by [PyRosetta](https://www.pyrosetta.org/)-based interface scoring and a multi-stage filter cascade.

This wrapper exposes Germinal as a single tool (`germinal-design`) that runs one end-to-end design campaign per call against one target. The upstream [`SantiagoMille/germinal`](https://github.com/SantiagoMille/germinal) repository is installed at a pinned commit and invoked via subprocess; all upstream config defaults from `configs/run/{vhh,scfv}.yaml`, `configs/config.yaml`, and `configs/filter/*` are preserved verbatim.

## Background

**What does this tool do?**
Germinal designs novel antibody sequences (VHH or scFv) that bind a user-specified epitope on a target protein. Given a target PDB and a list of hotspot residues defining the epitope, it produces a ranked set of antibodies with predicted complex structures and per-design quality metrics (interface pTM, pAE, pDockQ2, pLDDT, etc.).

**Why is this important?**
Traditional antibody discovery campaigns (immunization, phage display, hybridoma screening) take many months and require live animals or large naive libraries. De novo computational design can compress this loop to days, with experimental success rates of 4-22% reported across the published Germinal benchmarks — competitive with conventional pipelines while requiring only a target structure and an epitope definition. This is a step-change for therapeutic antibody discovery, especially against difficult targets where wet-lab campaigns repeatedly fail.

**Scientific foundation:**
Germinal builds on prior antibody hallucination work and adds four key advances:
1. **Epitope hotspot conditioning** — ColabDesign loss terms force the hallucinated binder to contact the user-specified hotspot residues, making the campaign genuinely epitope-targeted rather than opportunistic.
2. **Antibody language model gradient merging** — IgLM/AbLang likelihoods are merged into the hallucination gradient to bias designs toward natural antibody-like sequences (germline-like CDR distributions, framework conservation).
3. **Multi-stage filter cascade** — designs flow through cheap structural filters (clashes, sc_rmsd, hotspot contacts, CDR interface fraction) before the expensive external structure-validation step, so most failed trajectories are killed early.
4. **Independent structure-validation backend** — final designs are re-folded with a backend distinct from the AF2-Multimer used during hallucination (Chai-1, AF3, or Protenix), so the filter is not just measuring the hallucinator's self-consistency.

## Tools

### Germinal Antibody Design (`germinal-design`)

Run a Germinal antibody-design campaign end-to-end.

Spawns the upstream `run_germinal.py` (pinned commit) inside the tool's
standalone env, parses the resulting `runs/<exp>/{accepted,redesign_candidates,trajectories}/`
output trees, and returns a typed :class:`GerminalOutput`. Each call is one
end-to-end campaign against a single target — Germinal's pipeline is
stateful within a run so we do not fan out across targets.

## Tool Catalog

| Tool key | Operation | Input | Output | Use case |
|---|---|---|---|---|
| `germinal-design` | De novo VHH or scFv design | Target PDB + chain + hotspots | List of `GerminalDesign` (sequence + structure + metrics) | Epitope-targeted antibody design campaign |

## Execution Modes

- **GPU required** — NVIDIA GPU with ≥40 GB VRAM. H100 80 GB is recommended for scFv mode or large targets (>250 residues); CPU is not supported by the upstream pipeline.
- **Runtime** — 2-8 minutes per design on H100 80 GB. The default `max_trajectories=10000` means a full campaign takes hours to days; reduce for development.
- **Tests** — a small smoke test (`max_trajectories=2`) completes in roughly 5-15 minutes on an H100.

## How It Works

The Germinal pipeline runs three stages per trajectory, looping until either `max_trajectories` is exhausted or `max_passing_designs` is reached:

1. **Hallucination** — ColabDesign + AlphaFold2-Multimer optimize a randomly initialized binder backbone against the target. Antibody-specific losses (CDR length / framework constraints, IgLM and AbLang language-model gradients) bias the trajectory toward natural antibody geometries; epitope hotspot loss terms force the binder to contact the specified residues.
2. **Sequence redesign** — AbMPNN samples multiple new sequences for the hallucinated backbone, holding structure fixed. This decouples sequence-likelihood from the AF2-Multimer-driven hallucination and produces a diverse sequence pool per trajectory.
3. **Structure validation + filtering** — Each (backbone, sequence) candidate is re-folded with the chosen `structure_model` backend (Chai-1 by default; AlphaFold3 or Protenix optional). PyRosetta computes interface metrics (clashes, sc_rmsd, interface shape complementarity, hbonds, hydrophobicity, pDockQ2). A multi-stage filter cascade (initial → final, with `design_type`-specific thresholds) accepts, marks-for-redesign, or rejects each design.

> **Source-fidelity note**: This wrapper installs the upstream `SantiagoMille/germinal` repo at pinned commit `1e1c1a5` and spawns its `python run_germinal.py` via subprocess. All Germinal config defaults (`configs/run/{vhh,scfv}.yaml`, `configs/config.yaml`, `configs/filter/*`) are preserved verbatim. The only divergence from upstream defaults is `structure_model="chai"` — we default to Chai-1 because it auto-installs via `pip`, whereas upstream defaults to `protenix` for VHH and `af3` for scFv (both of which require user-provisioned weights and environments).

## Input Parameters

`GerminalInput` fields (defined in [`germinal_design.py`](germinal_design.py)):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_pdb` | `str` | *required* | Target structure as a PDB file path or PDB-format string. Must include a chain matching `target_chain`. |
| `target_chain` | `str` | `"A"` | Chain ID(s) of the target. Single letter (e.g. `"A"`) or comma-separated for multi-chain targets (e.g. `"A,B"`). |
| `binder_chain` | `str` | `"B"` | Chain ID assigned to the designed binder. Must differ from `target_chain`. Default matches Germinal's `configs/target/pdl1.yaml`. |
| `hotspots` | `list[str]` | `[]` | Target hotspot residues in `"<chain_letter><resnum>"` format (e.g. `["A37", "A39", "A41"]`). The binder is forced to contact these residues. |
| `target_name` | `str \| None` | `None` | Short identifier; used as the Hydra `target=<name>` selector and as a prefix in output filenames. If `None`, derived from a hash of the PDB content. |
| `hotspot_residue` | `str \| None` | `None` | Optional single residue (e.g. `"W40"`) used as the Chai-1 contact-restraint anchor. Mirrors Germinal's `hotspot_residue` field in `configs/target/*.yaml`. |

**Hotspot format**: each entry must match `<chain_letter><resnum>` (e.g. `"A37"`). Hotspot chains must exist in `target_chain`. Validated at construction time.

## Configuration

### Main parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `design_type` | `Literal["vhh", "scfv"]` | `"vhh"` | Run preset: `"vhh"` (single-domain nanobody) or `"scfv"` (scFv). Selects the upstream `configs/run/{vhh,scfv}.yaml` preset. |
| `max_trajectories` | `int` | `10000` | Hard cap on total trajectories before stopping. Reduce for dev/testing. |
| `max_passing_designs` | `int` | `100` | Stop early once this many designs pass all final filters. Set below `max_trajectories` to enable early stopping. |
| `structure_model` | `Literal["chai", "af3", "protenix"]` | `"chai"` | Cofolding backend. Default `"chai"` (auto-installed). `"af3"` and `"protenix"` require user-provisioned weights — see [Backend Configuration](#backend-configuration). |
| `plddt_threshold` | `float \| None` | `None` | Override final `external_plddt` filter. **Preset wins if `None`** — VHH preset: `> 0.87`, scFv preset: `> 0.85`. |
| `iptm_threshold` | `float \| None` | `None` | Override final `external_iptm` filter. **Preset wins if `None`** — both presets: `> 0.74`. |
| `ipae_threshold` | `float \| None` | `None` | Override final `external_pae` filter (Å). **Preset wins if `None`** — VHH preset: `< 7.5`, scFv preset: `< 8`. |
| `ptm_threshold` | `float \| None` | `None` | Override final `external_ptm` filter. **Preset wins if `None`** — both presets: `> 0.84`. |
| `pdockq2_threshold` | `float \| None` | `None` | Override final `pdockq2` filter. **Preset wins if `None`** — both presets: `> 0.23`. |

### Advanced parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_hallucinated_trajectories` | `int` | `1000` | Stop after this many trajectories complete the hallucination stage (independent of how many pass downstream filters). Source default. |
| `germinal_overrides` | `dict[str, Any]` | `{}` | Arbitrary Hydra overrides for `run_germinal.py`. Applied verbatim as `<key>=<value>` CLI args. Use to set keys not exposed as typed fields (e.g. `{"logits_steps": 100, "weights_iptm": 1.0, "omit_AAs": "CM"}`). See `configs/run/{vhh,scfv}.yaml` for the full key set. |
| `filter_overrides` | `dict[str, dict[str, dict[str, Any]]]` | `{}` | Override filter YAML values. Schema: `{"initial" \| "final": {<filter_name>: {"value": <v>, "operator": <op>}}}`. Merged on top of the design_type preset. |

## Output Specification

### `GerminalOutput`

| Field | Type | Description |
|---|---|---|
| `designs` | `list[GerminalDesign]` | All produced designs across the `accepted`, `redesign_candidate`, and `trajectory` stages. |
| `pipeline_stats` | `dict[str, int]` | Per-stage counts from Germinal's `failure_counts.csv` (trajectories attempted, designs accepted, per-filter failure counts). |
| `num_accepted` | `int` (computed) | Number of designs in the `accepted` stage. |
| `num_designs` | `int` (computed) | Total number of returned designs across all stages. |

`GerminalOutput` is iterable and indexable — `for design in result:` and `result[0]` work directly. Inherited fields from `BaseToolOutput`: `success`, `execution_time`, `errors`, `tool_id`.

### `GerminalDesign`

| Field | Type | Description |
|---|---|---|
| `sequence_heavy` | `str` | Heavy chain (or VHH) amino-acid sequence. |
| `sequence_light` | `str \| None` | Light chain sequence (scFv only; `None` for VHH). |
| `structure` | `Structure` | Predicted binder + target complex (proto-tools `Structure` object). |
| `metrics` | `GerminalDesignMetrics` | Per-design quality metrics (see below). |
| `stage_passed` | `Literal["accepted", "redesign_candidate", "trajectory"]` | Highest pipeline stage reached. |
| `design_id` | `str` | Germinal's internal design identifier (e.g. `"<target>_<traj_idx>_<mpnn_idx>"`). |
| `trajectory_index` | `int` | Parent hallucination trajectory index. |
| `mpnn_index` | `int` | AbMPNN sample index within the trajectory. |

### `GerminalDesignMetrics`

The `metric_spec` exposes the following keys (primary metric: `i_ptm`):

**Trajectory metrics** (always available, from Germinal's `TRAJECTORY_METRICS_TO_SAVE`):
- `plddt`, `ptm`, `i_ptm`, `i_pae`, `pae` — AF2-Multimer confidence/error metrics
- `loss` — total hallucination loss
- `lm_ll` — antibody language model log-likelihood (IgLM/AbLang)
- `helix`, `beta_strand` — secondary structure fractions

**Filter metrics** (available after the filter stage):
- `clashes` (count), `sc_rmsd` (Å)
- `binder_near_hotspot` (bool), `cdr3_hotspot_contacts` (count), `percent_interface_cdr` (fraction)
- `interface_shape_comp`, `interface_hbonds` (count), `interface_hydrophobicity`
- `surface_hydrophobicity`
- `pdockq2`

**External structure-validation metrics** (available after structure validation; produced by Chai-1 / AF3 / Protenix):
- `external_plddt`, `external_iptm`, `external_ptm`, `external_pae`

Access via attribute (`d.metrics.i_ptm`) or mapping (`d.metrics["i_ptm"]`).

### Export formats

`GerminalOutput.export(name=..., export_path=..., file_format=...)` supports:
- `"pdb"` — one PDB file per design at `<export_path>/<name>/<design_id>.pdb`
- `"csv"` — single CSV at `<export_path>/<name>.csv` (one row per design; sequences + flattened metrics)
- `"json"` — single JSON file at `<export_path>/<name>.json` (all designs excluding structures)

## Model Variants

The `structure_model` config field selects the structure-validation backend. Each has different setup requirements:

### Chai-1 (default)

No manual setup. The Germinal standalone env auto-installs `chai_lab` via `pip`, and Chai-1 fetches its weights on first use. This is the recommended starting point.

### AlphaFold3

Requires manual installation of AF3 weights (gated; request access at [DeepMind's AF3 form](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters)) and a Singularity container. Once installed, point Germinal at them via `germinal_overrides`:

```python
config = GerminalConfig(
    structure_model="af3",
    germinal_overrides={
        "af3_repo_path": "/path/to/alphafold3",
        "af3_sif_path": "/path/to/alphafold3.sif",
        "af3_model_dir": "/path/to/af3_weights",
        "af3_db_dir": "/path/to/af3_dbs",
    },
)
```

See the [Germinal upstream docs](https://github.com/SantiagoMille/germinal#alphafold3) for the full install procedure. AF3 is the backend used in the published filter calibration and is recommended for production campaigns.

### Protenix

Requires a separate conda env with [Protenix](https://github.com/bytedance/Protenix) installed (the Germinal env does not bundle it). Once installed:

```python
config = GerminalConfig(
    structure_model="protenix",
    germinal_overrides={
        "protenix_conda_env": "protenix",
        "protenix_model_name": "protenix_base_default_v1.0.0",
    },
)
```

## Quick Start Examples

**Example 1: Minimal VHH smoke test (~2 trajectories)**

Fast end-to-end check that the pipeline runs. Probably won't produce any accepted designs at this trajectory count — used to verify the env is healthy.

```python
from proto_tools.tools.binder_design.germinal import (
    run_germinal_design,
    GerminalInput,
    GerminalConfig,
)

inputs = GerminalInput(
    target_pdb="tests/dummy_data/pdl1.pdb",
    target_chain="A",
    binder_chain="B",
    hotspots=["A56", "A66", "A115"],
    target_name="pdl1_smoke",
)
config = GerminalConfig(
    design_type="vhh",
    max_trajectories=2,
    max_passing_designs=1,
)
result = run_germinal_design(inputs, config)
print(f"Returned {result.num_designs} designs ({result.num_accepted} accepted)")
```

**Example 2: Production VHH against PD-L1 with published filter thresholds**

Mirrors the Germinal paper's PD-L1 campaign. Filter thresholds left as `None` so the VHH preset values apply.

```python
from proto_tools.tools.binder_design.germinal import (
    run_germinal_design,
    GerminalInput,
    GerminalConfig,
)

inputs = GerminalInput(
    target_pdb="pdbs/pdl1.pdb",
    target_chain="A",
    binder_chain="B",
    hotspots=["A56", "A66", "A115"],
    target_name="pdl1",
)
config = GerminalConfig(
    design_type="vhh",
    max_trajectories=10000,
    max_passing_designs=100,
    structure_model="chai",
    # All filter thresholds left as None → VHH presets apply:
    #   external_plddt > 0.87, external_iptm > 0.74, external_pae < 7.5,
    #   external_ptm > 0.84, pdockq2 > 0.23
)
result = run_germinal_design(inputs, config)
for d in result:
    if d.stage_passed == "accepted":
        print(f"{d.design_id}: i_ptm={d.metrics.i_ptm:.3f} sequence={d.sequence_heavy}")
```

**Example 3: scFv mode with custom Hydra overrides**

scFv against a custom target, tuning the hallucination loss weights and step counts via `germinal_overrides`.

```python
from proto_tools.tools.binder_design.germinal import (
    run_germinal_design,
    GerminalInput,
    GerminalConfig,
)

inputs = GerminalInput(
    target_pdb="pdbs/my_target.pdb",
    target_chain="A",
    binder_chain="B",
    hotspots=["A101", "A104", "A108"],
    target_name="my_target_scfv",
)
config = GerminalConfig(
    design_type="scfv",
    max_trajectories=5000,
    max_passing_designs=50,
    structure_model="chai",
    germinal_overrides={
        "logits_steps": 100,      # more sequence-optimization steps (default: 60 for scFv)
        "weights_iptm": 1.0,      # stronger interface pTM loss
        "omit_AAs": "CM",         # exclude cysteine and methionine from the design
    },
)
result = run_germinal_design(inputs, config)
result.export(name="scfv_designs", export_path="./outputs", file_format="csv")
```

## Best Practices & Gotchas

- **One target per call.** `run_germinal_design()` is one end-to-end campaign against one target. To screen multiple targets, loop in Python.
- **Reduce `max_trajectories` for dev.** The upstream default (10000) is hours-to-days; use `max_trajectories=2..50` while iterating.
- **Set `max_passing_designs` below `max_trajectories`** to enable early stopping once enough designs accept.
- **Prefer `None` for filter thresholds** — VHH and scFv presets use *different* values (e.g. VHH `external_plddt > 0.87` vs scFv `> 0.85`). Leaving `*_threshold=None` lets the right preset apply.
- **AF3 is the published gold standard.** Filter thresholds are calibrated for AlphaFold3; Chai-1 is the convenience default but you may need to retune to match published acceptance rates.
- **GPU memory:** VHH fits in 40 GB; scFv and targets >250 residues typically need 80 GB.
- **Pick hotspots carefully.** Buried residues or non-contiguous patches drop acceptance rates regardless of other tuning.
- **PyRosetta is academic-only** — see the [PyRosetta license](https://www.pyrosetta.org/home/licensing-pyrosetta).
- **`germinal_overrides` is the escape hatch** for any upstream Hydra knob not exposed as a typed `GerminalConfig` field; applied verbatim.

## References

**Primary publication:**
- Mille-Fragoso, L. S., Driscoll, C. L., Wang, J. N., Dai, H., Widatalla, T., Zhang, J. L., Zhang, X., Rao, B., Feng, L., Hie, B. L., & Gao, X. J. (2025). *Efficient generation of epitope-targeted de novo antibodies with Germinal.* bioRxiv. [DOI: 10.1101/2025.09.19.677421](https://doi.org/10.1101/2025.09.19.677421)

**Implementation:**
- Upstream repository: [SantiagoMille/germinal](https://github.com/SantiagoMille/germinal)
- PMC mirror: [PMC12485712](https://pmc.ncbi.nlm.nih.gov/articles/PMC12485712/)

**Upstream dependencies:**
- [ColabDesign](https://github.com/sokrypton/ColabDesign) — AF2-Multimer hallucination scaffold
- AbMPNN — antibody-specific ProteinMPNN variant ([Dreyer et al., 2023](https://arxiv.org/abs/2310.19513)); weights bundled inside the Germinal repo at `colabdesign/colabdesign/mpnn/weights_abmpnn/`
- [IgLM](https://github.com/Graylab/IgLM) / [AbLang](https://github.com/oxpig/AbLang) — antibody language models for hallucination gradient
- [Chai-1](https://github.com/chaidiscovery/chai-lab) — default structure-validation backend
- [AlphaFold3](https://github.com/google-deepmind/alphafold3) / [Protenix](https://github.com/bytedance/Protenix) — alternative structure-validation backends
- [PyRosetta](https://www.pyrosetta.org/) — interface scoring (clashes, sc_rmsd, shape complementarity, hbonds, pDockQ2)

## Related Tools

**Tools often used together:**
- [`chai1-prediction`](../../structure_prediction/chai1/) — the default structure-validation backend; can be invoked standalone to re-score Germinal outputs
- [`alphafold3-prediction`](../../structure_prediction/alphafold3/) — alternative validation backend
- [`protenix-prediction`](../../structure_prediction/protenix/) — alternative validation backend
- [`proteinmpnn-sample`](../../inverse_folding/proteinmpnn/) — exposes the AbMPNN sequence-design step independently if you want to run inverse folding outside the Germinal pipeline (set `model_choice="abmpnn"`)

**Alternative tools:**
- [`rfdiffusion3-design`](../../structure_design/rfdiffusion3/) — general-purpose de novo binder design (non-antibody scaffolds)
