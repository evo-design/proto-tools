<a href="https://bio-pro.mintlify.app/tools/structure-dynamics/bioemu"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# BioEmu

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

BioEmu generates protein conformational ensembles using a diffusion generative model trained on [molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) (MD) simulation data. Given a protein sequence, it produces an ensemble of 3D backbone structures representing the [equilibrium distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution) of conformations the protein adopts -- capturing folded states, alternative conformations, and conformational heterogeneity without running explicit MD simulations.

- **Tool key**: `bioemu-sample`
- **Input**: Single-chain protein sequences (monomers only, recommended <= 500 residues)
- **Output**: `StructureEnsemble` objects containing sampled backbone conformations
- **Execution**: GPU required

## Background

Proteins are not static objects. In solution, a protein constantly fluctuates between conformational states, and this dynamics is essential for function: enzyme catalysis, [allosteric regulation](https://en.wikipedia.org/wiki/Allosteric_regulation), molecular recognition, and signal transduction all depend on conformational changes.

Traditional approaches to studying protein dynamics include:
- **Molecular dynamics (MD)**: Physically rigorous but computationally expensive (microseconds of simulation can take weeks of GPU time)
- **[NMR spectroscopy](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance_spectroscopy_of_proteins)**: Experimental ensemble methods, but limited to small proteins
- **[Normal mode analysis](https://en.wikipedia.org/wiki/Normal_mode)**: Fast but limited to harmonic fluctuations around a single structure

BioEmu takes a fundamentally different approach: it learns the equilibrium conformational distribution directly from MD training data using a score-based diffusion model. Given only a protein sequence, it generates an ensemble of backbone conformations that approximate the [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution) -- the thermodynamically correct distribution of states the protein visits at equilibrium.

This makes BioEmu orders of magnitude faster than MD while capturing the same large-scale conformational heterogeneity (though it does not model explicit time-dependent dynamics or rare events).

## Tools

### BioEmu Conformational Ensemble Sampling (`bioemu-sample`)

Generate protein conformational ensembles using BioEmu.

## Model Variants

| Model | Description | Recommended |
|-------|-------------|-------------|
| `bioemu-v1.0` | Initial preprint release | No |
| `bioemu-v1.1` | Weights from the published Science paper | Yes (default) |
| `bioemu-v1.2` | Trained on extended MD + folding free-energy data | Use when folding-state thermodynamics matter |

All three variants accept the same inputs and produce the same output format. v1.1 generally produces higher-quality ensembles than v1.0; v1.2 is recommended when comparing predictions against folding-free-energy measurements.

## Execution Modes

| Mode | Backend | Device |
|------|---------|--------|
| Local venv | `ToolInstance("bioemu")` running `standalone/inference.py` | Local GPU (`cuda`) |


## How It Works

1. **Sequence encoding**: The input amino acid sequence is encoded into a representation that conditions the generative model
2. **Diffusion sampling**: Starting from random noise, the model iteratively denoises 3D backbone coordinates through a learned reverse diffusion process, guided by the sequence embedding
3. **Ensemble generation**: The process is repeated `num_samples` times (default 500), each producing an independent backbone conformation drawn from the learned equilibrium distribution
4. **Quality filtering**: If `filter_samples=True` (default), BioEmu applies internal quality checks to remove poorly generated samples (e.g., structures with steric clashes or broken chain geometry)
5. **Structure packaging**: Filtered conformations are returned as `Structure` objects within a `StructureEnsemble`

## Input Parameters

`BioEmuInput` extends `StructurePredictionInput`:

| Field | Type | Description |
|-------|------|-------------|
| `complexes` | `List[Complex]` | Protein complexes to sample. Each must be a single-chain monomer with standard amino acids only. |

**Constraints:**
- Each complex must contain exactly one protein chain (monomers only)
- Only standard amino acid characters are allowed
- Sequences > 500 residues will produce a warning (quality may degrade)
- Chain modifications are not supported

## Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_samples` | `int` | `500` | >= 1 | Number of conformations to sample per sequence |
| `model_name` | `"bioemu-v1.0"` \| `"bioemu-v1.1"` \| `"bioemu-v1.2"` | `"bioemu-v1.1"` | -- | BioEmu checkpoint variant |
| `filter_samples` | `bool` | `True` | -- | Drop unphysical samples (steric clashes, chain discontinuities) |
| `batch_size` | `int` | `10` | >= 1 | Batch size at L=100; effective batch scales as `batch_size * (100/L)^2` |
| `denoiser_type` | `"dpm"` \| `"heun"` | `"dpm"` | -- | Diffusion sampler algorithm (dpm = 50 deterministic steps; heun = stochastic) |
| `denoiser_config` | `Optional[str]` | `None` | -- | Path to a custom denoiser/steering YAML (e.g. `physical_steering.yaml`); overrides `denoiser_type` when set |
| `msa_host_url` | `Optional[str]` | `None` | -- | Override the ColabFold MMseqs2 MSA server URL |
| `cache_embeds_dir` | `Optional[str]` | `None` | -- | Directory to cache MSA embeddings across runs |
| `cache_so3_dir` | `Optional[str]` | `None` | -- | Directory to cache SO3 precomputations across runs |
| `output_dir` | `Optional[str]` | `None` | -- | Optional directory for raw BioEmu output files |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cpu"` | Inference device (inherited from StructurePredictionConfig) |
| `verbose` | `bool` | `False` | -- | Verbose logging (inherited from StructurePredictionConfig) |

### Parameter Guides

**Number of samples:**

| num_samples | Use Case | Compute Time |
|-------------|----------|--------------|
| 10-50 | Quick check of conformational diversity | Seconds |
| 100-200 | Moderate ensemble for visualization | Minutes |
| 500 (default) | Publication-quality ensemble | Minutes |
| 1000-5000 | Detailed free-energy landscape analysis | Tens of minutes |

**Batch size:**

The `batch_size` parameter controls how many samples are generated in parallel on the GPU. Larger values use more GPU memory but are faster. If you encounter out-of-memory errors, reduce `batch_size`.

| batch_size | GPU Memory | Speed |
|------------|------------|-------|
| 1 | Minimal | Slow |
| 10 (default) | Moderate | Good balance |
| 50 | High | Fast (if memory allows) |

**Steering (`denoiser_config`):**

BioEmu ships a Sequential Monte Carlo (SMC) steering system that biases sampling toward more physically plausible structures (fewer steric clashes, fewer chain breaks). To enable it, set `denoiser_config` to the path of a steering YAML (e.g. `src/bioemu/config/steering/physical_steering.yaml` from the upstream package). The YAML — not the wrapper — controls the particle count (`num_particles`, typically 3–10), the potential set, and start/end timesteps; the wrapper just forwards the path to upstream.

## Output Specification

`BioEmuOutput` contains:

| Field | Type | Description |
|-------|------|-------------|
| `ensembles` | `List[StructureEnsemble]` | One ensemble per input complex |

Each `StructureEnsemble` contains:

| Field | Type | Description |
|-------|------|-------------|
| `structures` | `List[Structure]` | Individual backbone conformations |
| `sequence` | `str` | The input protein sequence |

Each `Structure` in the ensemble has metadata:

| Key | Description |
|-----|-------------|
| `ensemble_idx` | Index of the input complex |
| `frame_idx` | Index of this conformation within the ensemble |

Output metadata:

| Key | Description |
|-----|-------------|
| `num_complexes` | Number of input complexes processed |
| `total_structures` | Total number of conformations generated across all complexes |
| `model_name` | BioEmu model variant used |

Export formats: `pdb`, `json`

## Interpreting Results

- **Ensemble size**: The number of structures in the ensemble may be less than `num_samples` if `filter_samples=True`, since poor-quality samples are removed.
- **Conformational clusters**: Superimpose the ensemble structures to identify clusters of similar conformations, which correspond to metastable states.
- **RMSD distribution**: Calculate pairwise RMSD across the ensemble to quantify conformational diversity. A narrow distribution suggests a rigid protein; a broad distribution suggests flexibility.
- **Per-residue flexibility**: Calculate per-residue [RMSF](https://en.wikipedia.org/wiki/Root-mean-square_deviation_of_atomic_positions) (root-mean-square fluctuation) across the ensemble to identify flexible loops and rigid core regions.
- **Backbone only**: BioEmu outputs backbone atoms (N, CA, C, O). Side-chain coordinates are not included. Use a side-chain packing tool (e.g., SCWRL or Rosetta) if side-chain conformations are needed.

## Quick Start Examples

**Basic ensemble sampling:**
```python
from proto_tools.tools.structure_prediction.shared_data_models import (
    Complex,
)
from proto_tools.tools.structure_dynamics.bioemu import (
    BioEmuInput, BioEmuConfig, run_bioemu,
)

complex_ = Complex(
    chains=[{"sequence": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK", "entity_type": "protein"}],
)

inputs = BioEmuInput(complexes=[complex_])
config = BioEmuConfig(num_samples=100, model_name="bioemu-v1.1")
result = run_bioemu(inputs, config)

ensemble = result.ensembles[0]
print(f"Generated {len(ensemble.structures)} conformations")
print(f"Sequence: {ensemble.sequence}")
```

**Multiple proteins in one call:**
```python
complexes = [
    Complex(chains=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"]),
    Complex(chains=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVG"]),
]

inputs = BioEmuInput(complexes=complexes)
config = BioEmuConfig(num_samples=200)
result = run_bioemu(inputs, config)

for i, ensemble in enumerate(result.ensembles):
    print(f"Protein {i}: {len(ensemble.structures)} conformations")
```

**Export ensemble to PDB files:**
```python
result.export("/path/to/output_dir", file_format="pdb")
# Creates: output_dir/ensemble_0/conformation_0.pdb, conformation_1.pdb, ...
```

**Access individual conformations:**
```python
ensemble = result.ensembles[0]

# Get PDB string for a specific conformation
pdb_string = ensemble.structures[0].structure_pdb

# Write a specific conformation to file
with open("conformation_0.pdb", "w") as f:
    f.write(pdb_string)
```

**Quick sampling for visualization:**
```python
config = BioEmuConfig(
    num_samples=20,
    batch_size=20,
    filter_samples=False,
)
result = run_bioemu(inputs, config)
```

## Best Practices & Gotchas

- **Monomers only.** BioEmu does not support multi-chain complexes. Each input complex must contain exactly one protein chain. Attempting to pass multi-chain inputs will raise a validation error.
- **500 residues is the practical limit.** Sequences longer than 500 residues will produce a warning. Quality and computational cost both degrade for longer sequences.
- **Backbone only.** BioEmu generates backbone coordinates (N, CA, C, O) without side chains. This is sufficient for assessing conformational diversity but not for detailed binding analysis.
- **Default 500 samples is well-calibrated.** The default provides good coverage of the conformational landscape for most proteins. Reduce to 50-100 for quick checks; increase to 1000+ only if you need detailed free-energy analysis.
- **Filter samples by default.** Keep `filter_samples=True` unless you specifically want to analyze the raw model output. Unfiltered ensembles may contain structures with steric clashes.
- **Standard amino acids only.** Non-standard residues, modified amino acids, and non-protein chains are not supported and will raise validation errors.
- **GPU is required.** Diffusion sampling is computationally intensive. CPU execution is technically possible but impractically slow.
- **Batch size affects memory, not output.** The `batch_size` parameter controls GPU parallelism during sampling. Reduce it if you encounter CUDA out-of-memory errors; increasing it beyond 50 rarely helps.

## References

- Zheng, S., He, J., Liu, C., et al. (2024). Predicting equilibrium distributions for molecular systems with deep learning. *Nature Machine Intelligence*. DOI: [10.1038/s42256-024-00837-3](https://doi.org/10.1038/s42256-024-00837-3)
- Also published in *Science*: https://www.science.org/doi/10.1126/science.adv9817
- GitHub: https://github.com/microsoft/bioemu

## Related Tools

**Often used together:**
- **AlphaFold** (`alphafold3-prediction`) -- Predict a single high-confidence structure; compare against BioEmu ensemble
- **ESMFold** (`esmfold-prediction`) -- Fast single-structure prediction for initial screening before ensemble sampling
- **Segmasker** (`segmasker-score`) -- Screen sequences for low-complexity before ensemble generation

**Alternatives for dynamics:**
- **Molecular dynamics** (GROMACS, OpenMM) -- Explicit time-resolved dynamics; more accurate but orders of magnitude slower
- **Normal mode analysis** -- Fast harmonic approximation of dynamics around a single structure; less accurate than BioEmu for large conformational changes
