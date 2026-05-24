<a href="https://bio-pro.mintlify.app/tools/structure-design/rfdiffusion3"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# RFdiffusion3

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview
RFdiffusion3 is a [diffusion](https://en.wikipedia.org/wiki/Diffusion_model)-based generative model for de novo protein structure design. Unlike structure prediction tools that predict what a given sequence will fold into, RFdiffusion3 generates novel protein structures (and sequences) that satisfy specified constraints.

## Background

**What does this tool do?**
RFdiffusion3 generates novel protein structures at atomic resolution using a diffusion process. Given constraints like target binding sites, structural motifs, or symmetry requirements, it produces designed proteins with both 3D coordinates and amino acid sequences optimized to fold into those structures.

**Why is this important?**
De novo protein design enables creation of proteins with novel functions not found in nature:
- Designing binders for therapeutic targets (drugs, vaccines)
- Creating enzymes with novel catalytic activities
- Engineering protein materials with desired properties
- Scaffolding around active sites for functional enzyme design
- Building symmetric assemblies for nanotechnology applications

**Scientific foundation:**
RFdiffusion3 uses a denoising diffusion process operating on all atoms (backbone and side-chains) simultaneously. Starting from random noise, it iteratively refines coordinates through learned denoising steps while conditioning on specified constraints. Key advances include:
1. **All-atom diffusion**: Models both backbone and side-chain atoms for realistic structure generation
2. **Flexible conditioning**: Supports diverse constraints (motifs, binders, symmetry, ligands)
3. **Sequence co-design**: Generates compatible sequences alongside structures
4. **Contig language**: Powerful specification system for complex design tasks

## Tools

### RFdiffusion3 Structure Design (`rfdiffusion3-design`)

Design protein structures using RFdiffusion3.

Uses RFdiffusion3, a diffusion-based generative model, to design novel
protein structures under specified constraints. Unlike structure prediction,
RFdiffusion3 generates both structure AND sequence, making it suitable for:

- De novo protein design (unconditional generation)
- Motif scaffolding (design around fixed structural motifs)
- Protein binder design (design proteins that bind to targets)
- Enzyme design (scaffold around catalytic sites)
- Symmetric protein design (design homo-oligomers)

Runs via local GPU execution in isolated Python environments.

## Execution Modes

- **Local GPU**: Requires GPU with >=24GB VRAM (A100/H100 recommended for production)
- **Runtime**: ~1-10 minutes per design batch depending on size and complexity

## How It Works

RFdiffusion3 uses a diffusion process that operates on atomic coordinates:
1. **Forward process**: Training data (real proteins) is progressively noised
2. **Reverse process**: A neural network learns to denoise, recovering structure
3. **Conditioning**: Constraints are applied during denoising to guide generation
4. **Sampling**: Multiple independent designs are generated and can be ranked

The model is trained on protein structures from the [PDB](https://www.rcsb.org/), learning the distribution of valid protein geometries. During inference, it generates novel structures by sampling from this learned distribution while respecting specified constraints.

**Key assumptions:**
- Generated structures should be physically realistic (proper bond lengths, angles)
- Constraints accurately specify the design objective
- Generated sequences should be designable (can actually fold into the structure)
- Larger designs may require more diffusion steps for quality

**Limitations:**
- Computational cost: Large designs (>500 residues) require significant GPU memory and time
- No explicit energy function: Relies on learned patterns, not physics-based scoring
- Designability not guaranteed: Generated structures may need validation with structure prediction
- Limited to proteins: Cannot design small molecules, nucleic acids, or carbohydrates

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `design_specs` | `List[RFdiffusion3DesignSpec]` | *required* | Design specifications with constraints |

Each `RFdiffusion3DesignSpec` contains:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_structure` | `Optional[str]` | Path to PDB/CIF or PDB string (motif scaffolding, binder design) |
| `length` | `Optional[str]` | Target length (int or "min-max" range as string) |
| `contig` | `Optional[str]` | Contig string specifying design topology |
| `ligand` | `Optional[str]` | Ligand selection by 3-letter codes (e.g. `"HAX,OAA"`) |
| `unindex` | `Optional[str \| dict]` | Unindexed motif components (flexible position) |
| `select_fixed_atoms` | `Optional[bool \| str \| dict]` | Atoms held fixed in 3D space during diffusion |
| `select_unfixed_sequence` | `Optional[bool \| str \| dict]` | Residues whose sequence can change |
| `select_hotspots` | `Optional[bool \| str \| dict]` | Hotspots for binder/PPI design |
| `symmetry` | `Optional[str \| dict]` | Symmetry for homo-oligomer: group-id str (e.g. `"C3"`) or `SymmetryConfig` dict; pair with `sampler_kind="symmetry"` |
| `select_buried` / `select_partially_buried` / `select_exposed` | `Optional[bool \| str \| dict]` | RASA conditioning |
| `select_hbond_donor` / `select_hbond_acceptor` | `Optional[dict[str, list[str]]]` | Atom-wise H-bond flags |
| `redesign_motif_sidechains` | `Optional[bool]` | Fix motif backbone, redesign side-chains |
| `plddt_enhanced` | `Optional[bool]` | pLDDT-based denoising enhancement (upstream default: True) |
| `infer_ori_strategy` | `Optional["com" \| "hotspots"]` | Origin placement strategy |
| `ori_token` | `Optional[List[float]]` | `[x, y, z]` origin override (Angstroms) |
| `partial_t` | `Optional[float]` | Noise (Angstroms) for partial diffusion (5.0-15.0 recommended) |
| `is_non_loopy` | `Optional[bool]` | True/False produces fewer/more loops |

Additional InputSpecification fields can be passed via `**kwargs` (model uses `extra="allow"`). See [upstream input.md](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md).

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_batches` | `int` | `1` | Independent batches per spec (total designs per spec = `n_batches * diffusion_batch_size`) |
| `diffusion_batch_size` | `int` | `8` | Designs sampled in parallel per batch (memory scales with this) |
| `num_timesteps` | `int` | `200` | Diffusion steps (higher = better quality, slower) |
| `step_scale` | `float` | `1.5` | Step size scale; higher = less diverse, more designable (typical 1.0-2.0) |
| `sampler_kind` | `"default" \| "symmetry"` | `"default"` | Sampler kind; `"symmetry"` for homo-oligomer design |
| `center_option` | `"all" \| "motif" \| "diffuse"` | `"all"` | Coordinate-frame centering mode |
| `use_classifier_free_guidance` | `bool` | `False` | Enable classifier-free guidance (CFG) |
| `cfg_scale` | `float` | `1.5` | CFG scale (typical 1.0-3.0); requires CFG enabled |
| `cfg_features` | `Optional[List[str]]` | `None` | CFG feature names; None uses upstream default |
| `cfg_t_max` | `Optional[float]` | `None` | Max diffusion timestep at which CFG is applied (0.0-1.0) |
| `gamma_0` | `float` | `0.6` | Sampler stochasticity; 0.0 = deterministic ODE |
| `low_memory_mode` | `bool` | `False` | Memory-efficient tokenization (slower); set if GPU RAM is tight |
| `dump_trajectories` | `bool` | `False` | Save diffusion trajectory frames to output dir |
| `align_trajectory_structures` | `bool` | `False` | Align trajectory frames (only when dumping) |
| `prevalidate_inputs` | `bool` | `False` | Fail-fast input JSON validation before launching diffusion |

### Sweep Priorities

1. **`num_timesteps`**: Most impactful for structure quality. Use 50-100 for rapid prototyping, 200-500 for production.
2. **`step_scale`**: Controls diversity vs designability tradeoff. Lower (1.0-1.3) for diverse exploration, higher (1.5-2.0) for safer designs.
3. **`diffusion_batch_size`**: Generate multiple designs per run; rank by downstream metrics.
4. **`use_classifier_free_guidance`** + **`cfg_scale`**: Enable when active-site / H-bond constraints matter; otherwise leave off (no-op when off).

## Output Specification

```python
RFdiffusion3Output(
    designed_structures: List[RFdiffusion3Designs],   # one bundle per input spec
)
```

Each `RFdiffusion3Designs` bundle holds the designs produced for one input spec
(length = `n_batches * diffusion_batch_size`):

| Field | Type | Description |
|-------|------|-------------|
| `spec_key` | `str` | Identifier of the originating input spec (e.g. `"spec-0"`) |
| `structures` | `list[RFdiffusion3Structure]` | The designs generated for this spec |

Each `RFdiffusion3Structure` contains:

| Field | Type | Description |
|-------|------|-------------|
| `structure` | `Structure` | Full atomic coordinates of designed protein |
| `sequence` | `str` | Amino acid sequence (multi-chain: chains separated by `/`) |
| `metadata` | `dict` | Sampled contig, chain info, and logged metrics |

## Interpreting Results

RFdiffusion3 does not provide built-in confidence scores. Assess design quality using downstream tools:

| Validation method | Tool | What it measures |
|-------------------|------|------------------|
| Structure prediction | ESMFold / Boltz2 | Whether the designed sequence folds into the intended structure |
| Inverse folding score | ProteinMPNN | Sequence-structure compatibility (perplexity) |
| Physics-based scoring | Rosetta energy | Physical plausibility of the structure |
| Fold confidence | AlphaFold pLDDT | Confidence that the sequence will fold as designed |

**Validation workflow:**
1. Generate designs with RFdiffusion3
2. Score with ProteinMPNN (perplexity < 4.0 is a good sign)
3. Predict structure from designed sequence with ESMFold/Boltz2
4. Compare predicted structure to designed structure (RMSD < 2A is excellent)

## Quick Start Examples

**Example 1: Unconditional protein design**
```python
from proto_tools.tools.structure_design.rfdiffusion3 import (
    run_rfdiffusion3, RFdiffusion3Input, RFdiffusion3Config, RFdiffusion3DesignSpec
)

# Design a 100-residue protein
spec = RFdiffusion3DesignSpec(length="100")
inputs = RFdiffusion3Input(design_specs=[spec])
config = RFdiffusion3Config(
    diffusion_batch_size=8,
    num_timesteps=200,
    step_scale=1.5
)

result = run_rfdiffusion3(inputs, config)

# Result is one bundle per input spec; each bundle holds the N designs for that spec.
for bundle in result.designed_structures:
    for design_index, structure in enumerate(bundle.structures):
        print(f"{bundle.spec_key} design {design_index}: {structure.sequence[:50]}...")
```

**Example 2: Variable-length design**
```python
from proto_tools.tools.structure_design.rfdiffusion3 import (
    run_rfdiffusion3, RFdiffusion3Input, RFdiffusion3Config, RFdiffusion3DesignSpec
)

# Design proteins between 80-120 residues
spec = RFdiffusion3DesignSpec(length="80-120")
inputs = RFdiffusion3Input(design_specs=[spec])
config = RFdiffusion3Config(
    diffusion_batch_size=4,
    num_timesteps=300  # Higher quality
)

result = run_rfdiffusion3(inputs, config)
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`num_timesteps`**:
  - Low values (50-100): Fast but potentially lower quality; good for initial exploration
  - High values (200-500): Higher quality; use for final production designs
- **`step_scale`**:
  - Low values (1.0-1.3): More diverse outputs; may include unusual structures
  - High values (1.5-2.0): More conservative; structures closer to training data

**Common mistakes:**
1. **Not validating designs**: Always run structure prediction on designed sequences to verify designability
2. **Ignoring motif specification**: For scaffolding, carefully specify which atoms to fix
3. **Insufficient sampling**: Generate multiple designs (8-32) and rank; don't rely on single designs
4. **Wrong contig syntax**: Chain breaks use `\0`, not `/`; ranges use `-` (e.g., `A1-50`)

**Tips for optimal results:**
- **Start simple**: Test with unconditional design before complex constraints
- **Validate with ProteinMPNN**: Score designs to check sequence-structure compatibility
- **Use step_scale ~1.5**: Good balance of diversity and quality for most applications
- **Generate batches**: Easier to select good designs from a batch than to get one perfect design

**Edge cases to watch for:**
- **Very long proteins (>300 AA)**: May need more timesteps; memory usage increases
- **Complex multi-chain designs**: Ensure contig string correctly specifies chain breaks
- **Tight geometric constraints**: May require more samples to find satisfying designs
- **Symmetric designs**: Use symmetry mode for homo-oligomers

## Seed Reproducibility

The `seed` parameter is passed to the rfd3 subprocess but the diffusion
pipeline uses non-deterministic CUDA operations that cannot be controlled
from outside the process, so different runs with the same seed produce
different designs. Upstream issue:
[RosettaCommons/foundry#170](https://github.com/RosettaCommons/foundry/issues/170).

## References

**Primary publication:**
- Butcher, Krishna, Mitra et al. (2025). "De novo Design of All-atom Biomolecular Interactions with RFdiffusion3". *bioRxiv*. [DOI: 10.1101/2025.09.18.676967](https://doi.org/10.1101/2025.09.18.676967)

**Implementation:**
- GitHub: [https://github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry)
- Documentation: [https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/README.md](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/README.md)

**Additional resources:**
- RFdiffusion (original): [https://github.com/RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion)
- Foundry model collection: [https://github.com/RosettaCommons/foundry](https://github.com/RosettaCommons/foundry)

## Related Tools

**Tools often used together:**
- `esmfold` / `boltz2-prediction`: Validate designs by predicting structure from designed sequence
- `proteinmpnn-score`: Score sequence-structure compatibility of designs
- `proteinmpnn-sample`: Generate alternative sequences for a designed structure

**Alternative tools:**
- RFdiffusion (original): Backbone-only diffusion; faster but less detailed
- Chroma: Alternative diffusion-based protein design method
- ESM3: Sequence-focused generative model (no explicit structure generation)
