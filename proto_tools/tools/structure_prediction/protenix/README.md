<a href="https://bio-pro.mintlify.app/tools/structure-prediction/protenix"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Protenix

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

Protenix is ByteDance's open-source reimplementation of AlphaFold3 that predicts 3D structures of biomolecular complexes using a [diffusion](https://en.wikipedia.org/wiki/Diffusion_model)-based architecture. It supports proteins, DNA, RNA, ligands, and their multi-chain complexes with optional post-translational modifications. Protenix is the first fully open-source model to match or exceed AlphaFold3 accuracy across diverse benchmarks.

- **Tool key**: `protenix-prediction`
- **Input**: Multi-chain complexes (protein, DNA, RNA, ligand) with optional modifications
- **Output**: 3D structures in mmCIF format with confidence metrics (pTM, ipTM, pLDDT, GPDE)
- **MSA support**: Automatic via ColabFold search (optional)
- **GPU required**: Yes

## Background

Predicting the 3D structure of biomolecular complexes is fundamental to understanding biological function. Protein-protein interactions govern signaling pathways, protein-DNA interactions control gene regulation, and protein-ligand binding underlies drug action.

Protenix uses a diffusion-based architecture (following the AlphaFold3 approach) that:
- **Generates full-atom coordinates** rather than inter-residue distances, enabling direct prediction of ligand binding poses and nucleic acid conformations
- **Handles multi-modal inputs** natively: proteins, DNA, RNA, and small molecule ligands in a single prediction
- **Supports chemical modifications** via [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (CCD) codes, enabling prediction of structures with phosphorylation, methylation, and other PTMs
- **Uses MSA information** (optional) from ColabFold search to capture evolutionary conservation signals

The model architecture consists of a Pairformer module (iterative refinement of pair representations), followed by a diffusion module that denoises random 3D coordinates into the predicted structure. Multiple samples are generated and ranked by a confidence head.

## Tools

### Protenix Structure Prediction (`protenix-prediction`)

Predict 3D structures using Protenix.

Uses Protenix, an open-source reimplementation of AlphaFold3 by ByteDance
Research, to predict 3D structures of proteins, DNA, RNA, ligands, and their
complexes. Supports local GPU execution via isolated Python environments.

All input complexes are batched into a single Protenix CLI call for efficiency,
avoiding repeated model loading.

## Model Variants

| Model Name | Size | Recycling | Diffusion Steps | Notes |
|------------|------|-----------|-----------------|-------|
| `protenix_base_default_v1.0.0` | Base | 10 cycles | 200 steps | **Recommended.** Best overall accuracy. |
| `protenix_base_20250630_v1.0.0` | Base | 10 cycles | 200 steps | More recent training data cutoff (June 2025). |
| `protenix_base_default_v0.5.0` | Base | 10 cycles | 200 steps | Earlier base model version. |
| `protenix_base_constraint_v0.5.0` | Base | 10 cycles | 200 steps | Supports contact/pocket constraints for incorporating experimental priors. |
| `protenix_mini_default_v0.5.0` | Mini | 4 cycles | 5 steps | Compact model, faster predictions. |
| `protenix_mini_esm_v0.5.0` | Mini | 4 cycles | 5 steps | ESM2 embeddings. Good when MSAs are unavailable. |
| `protenix_mini_ism_v0.5.0` | Mini | 4 cycles | 5 steps | ISM embeddings. Alternative MSA-free mode. |
| `protenix_tiny_default_v0.5.0` | Tiny | 4 cycles | 5 steps | Smallest and fastest. High-throughput screening. |

**Choosing a model:**
- For publication-quality predictions, use `protenix_base_default_v1.0.0`
- For rapid screening or resource-constrained environments, use `protenix_mini_default_v0.5.0` or `protenix_tiny_default_v0.5.0`
- For MSA-free prediction (orphan proteins, designed sequences), use `protenix_mini_esm_v0.5.0`

## How It Works

1. **Input preparation**: Chains are converted to Protenix JSON format with entity types and modifications
2. **MSA generation** (optional): ColabFold search generates multiple sequence alignments for protein chains
3. **Feature embedding**: Sequences and MSAs are processed into pair and single representations
4. **Pairformer recycling**: Iterative refinement of pair representations (default: 10 cycles)
5. **Diffusion sampling**: Random 3D coordinates are denoised over multiple steps (default: 200 steps) to produce structure samples
6. **Ranking**: Multiple samples are generated per seed, and the best is selected by a confidence ranking score
7. **Output**: mmCIF structure files with per-residue and global confidence metrics

## Input Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | Yes | Complexes to predict. Each complex contains one or more chains. |

Each `StructurePredictionComplex` contains chains that can be specified as:
- Plain strings (entity type auto-inferred): `["MVLSPADKTN", "ATCGATCG"]`
- `Chain` objects with explicit types: `Chain(sequence="MVLSPADKTN", entity_type="protein")`
- `Chain` objects with modifications: `Chain(sequence="MVLSPADKTN", modifications=[(5, "SEP")])`

Supported entity types: `protein`, `dna`, `rna`, `ligand`

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `ProtenixModelName` | `"protenix_base_default_v1.0.0"` | Model checkpoint to use (see Model Variants) |
| `seeds` | `List[int]` | `[0]` | Random seeds for sampling. Each seed generates `num_diffusion_samples` structures. |
| `use_msa` | `bool` | `True` | Generate and use MSAs for protein chains via ColabFold |
| `colabfold_search_config` | `ColabfoldSearchConfig` | `None` (uses defaults) | Configuration for ColabFold MSA search |
| `num_diffusion_samples` | `int` | `5` | Structure samples per seed (best is returned by ranking score) |
| `num_diffusion_steps` | `int` | `200` | Denoising steps in the diffusion process |
| `num_pairformer_cycles` | `int` | `10` | Pairformer recycling iterations |
| `device` | `str` | `"cuda"` | Inference device |
| `verbose` | `bool` | `False` | Log progress during MSA generation, model loading, and prediction |

### Parameter Guides

**`num_diffusion_samples`** controls conformational exploration:

| Value | Use Case | Runtime Impact |
|-------|----------|---------------|
| 1-3 | Rapid screening, initial exploration | Fastest |
| 5 | Default, good balance of speed and quality | Moderate |
| 7-10 | High-confidence predictions, flexible complexes | Slowest |

**`num_diffusion_steps`** controls structure refinement quality:

| Value | Use Case | Runtime Impact |
|-------|----------|---------------|
| 100-150 | Fast screening, acceptable quality | ~50% of default |
| 200 | Default, recommended for production | Baseline |
| 300-500 | Maximum quality, critical predictions | 1.5-2.5x default |

**`num_pairformer_cycles`** controls iterative refinement:

| Value | Use Case | Runtime Impact |
|-------|----------|---------------|
| 3-5 | Fast screening | ~50% of default |
| 10 | Default, good accuracy | Baseline |
| 15-20 | Maximum refinement, publication quality | 1.5-2x default |

### Sweep Priorities

When using Protenix in optimization loops, prioritize sweeping:

1. **`num_diffusion_samples`**: Most impactful for finding optimal conformations and binding poses
2. **`num_pairformer_cycles`**: Affects refinement quality of the pair representations
3. **`num_diffusion_steps`**: For critical predictions, increase for finer structure quality
4. **`use_msa`**: Critical for accuracy but adds runtime. Try `False` for designed/orphan proteins

## Output Specification

| Field | Type | Description |
|-------|------|-------------|
| `structures` | `List[Structure]` | One predicted structure per input complex |

Each `Structure` contains 3D coordinates (mmCIF format) and a `metrics` dictionary:

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `confidence_score` | `float` | 0.0-1.0 | Primary ranking score (weighted combination) |
| `ptm` | `float` | 0.0-1.0 | Predicted TM-score (global fold quality) |
| `iptm` | `float` | 0.0-1.0 | Interface pTM (inter-chain interface quality) |
| `avg_plddt` | `float` | 0.0-1.0 | Average per-residue confidence (normalized from 0-100) |
| `gpde` | `float` | >0 (Angstroms) | Global Predicted Distance Error |
| `chain_ptm` | `List[float]` | 0.0-1.0 each | Per-chain PTM scores |
| `chain_plddt` | `List[float]` | 0.0-1.0 each | Per-chain average pLDDT |
| `chain_pair_iptm` | `List[List[float]]` | 0.0-1.0 each | Pairwise interface PTM between all chain pairs |
| `has_clash` | `bool` |: | Whether the structure has steric clashes |

## Interpreting Results

**Confidence score thresholds:**

| Range | Interpretation | Action |
|-------|---------------|--------|
| > 0.8 | Excellent: high confidence | Structure suitable for detailed analysis |
| 0.6-0.8 | Good: moderate confidence | Verify key interactions manually |
| < 0.6 | Poor: low confidence | Consider more samples, MSA, or alternative approaches |

**Metric-specific guidance:**
- **ptm > 0.8**: High confidence in the overall fold. Suitable for structural analysis.
- **iptm > 0.8**: High confidence in inter-chain interfaces. Binding mode is reliable.
- **iptm 0.6-0.8**: Moderate interface confidence. The binding orientation may be approximate.
- **avg_plddt > 0.9**: Very high per-residue confidence. Side-chain conformations are meaningful.
- **gpde < 10 A**: Good spatial accuracy. Relative chain positions are reliable.
- **has_clash = True**: Steric clashes present. May need energy minimization or more diffusion samples.

**Per-chain metrics** (`chain_ptm`, `chain_plddt`) help identify which chains in a complex are well-predicted versus uncertain. **Pairwise interface metrics** (`chain_pair_iptm`) reveal which specific chain-chain interfaces are confident.

## Quick Start Examples

**Single protein structure:**
```python
from proto_tools.tools.structure_prediction.protenix import (
    ProtenixInput,
    ProtenixConfig,
    run_protenix,
)

inputs = ProtenixInput(complexes=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"])
config = ProtenixConfig(
    model_name="protenix_base_default_v1.0.0",
    num_diffusion_samples=5,
    verbose=True,
)

result = run_protenix(inputs, config)
structure = result.structures[0]
print(f"Confidence: {structure.metrics['confidence_score']:.2f}")
print(f"pTM: {structure.metrics['ptm']:.2f}")
print(f"pLDDT: {structure.metrics['avg_plddt']:.2f}")
```

**Protein-protein complex:**
```python
inputs = ProtenixInput(
    complexes=[["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
                "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVGTSKEAKQDGIDLYKHMFE"]]
)
config = ProtenixConfig(num_diffusion_samples=5, verbose=True)

result = run_protenix(inputs, config)
metrics = result.structures[0].metrics
print(f"Interface confidence (ipTM): {metrics['iptm']:.2f}")
print(f"Chain-pair ipTM: {metrics['chain_pair_iptm']}")
```

**Protein-DNA complex with modifications:**
```python
from proto_tools.tools.structure_prediction.protenix import ProtenixInput, ProtenixConfig, run_protenix
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    Chain,
)

complex = StructurePredictionComplex(
    chains=[
        Chain(
            sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            entity_type="protein",
            modifications=[(4, "SEP")],  # Phosphoserine at position 4
        ),
        Chain(sequence="ATCGATCGATCGATCG", entity_type="dna"),
    ]
)

inputs = ProtenixInput(complexes=[complex])
config = ProtenixConfig(num_diffusion_samples=5, use_msa=True, verbose=True)
result = run_protenix(inputs, config)
```

**Fast screening with mini model (no MSA):**
```python
inputs = ProtenixInput(complexes=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"])
config = ProtenixConfig(
    model_name="protenix_mini_esm_v0.5.0",
    use_msa=False,
    num_diffusion_samples=3,
    verbose=True,
)

result = run_protenix(inputs, config)
```

**Export structure to PDB file:**
```python
result = run_protenix(inputs, config)
result.export("protenix_output/", file_format="pdb")
# Writes protenix_output/structure_0.pdb
```

## Best Practices & Gotchas

- **MSA improves accuracy significantly**: Keep `use_msa=True` for natural proteins. Only disable for designed sequences, orphan proteins, or when speed is critical.
- Ensure you have a CUDA-capable GPU with sufficient VRAM (~16-24 GB for base models, ~8 GB for mini/tiny).
- **CUDA isolation**: Protenix uses a fully isolated CUDA environment managed via micromamba. If you encounter CUDA compilation errors, check the `$VENV_PATH/cuda_env` directory.
- **Batch complexes for efficiency**: Pass multiple complexes in a single `run_protenix` call rather than calling the function repeatedly. This avoids reloading the model for each complex.
- **Increase samples for flexible complexes**: Protein-ligand docking and antibody-antigen complexes benefit from higher `num_diffusion_samples` (7-10) to explore binding poses.
- **Check `has_clash`**: If the output structure has steric clashes, consider increasing `num_diffusion_samples` or `num_diffusion_steps`.
- **Modifications use 1-based positions**: Position 1 is the first residue/base, following standard biological convention.
- **Ligand input format**: Ligands are specified as SMILES strings or CCD codes, not as sequences.
- **Seeds for diversity**: Use multiple seeds (`seeds=[0, 1, 2]`) to explore different sampling trajectories. Each seed produces `num_diffusion_samples` independent samples.

## Seed Reproducibility

Protenix honours `--seed`, but the `cuequivariance` triangle
multiplication/attention kernels we enable for speed accumulate float
ops non-deterministically, causing ~1-2 mÅ coordinate drift between
runs with the same seed. Forcing the `torch` fallback kernel would
restore determinism at a significant speed cost. Upstream confirmation
and deterministic-mode escape hatch:

- [bytedance/Protenix#116 — Different results for the same seed](https://github.com/bytedance/Protenix/issues/116)
- [bytedance/Protenix#119 — unstable predictions](https://github.com/bytedance/Protenix/issues/119)

The Protenix maintainers' recommended fix is to set
`USE_DEEPSPEED_EVO_ATTENTION=false` and pass `--deterministic true` to
the CLI — proto-tools currently does not enable this path because it
trades off significant inference speed.

## References

- ByteDance Protenix Team. "Protenix: Toward High-Accuracy Open-Source Biomolecular Structure Prediction." [Technical Report](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf)
- Abramson, J., Adler, J., Dunger, J. et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* 630, 493-500 (2024). DOI: [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)
- GitHub: [bytedance/Protenix](https://github.com/bytedance/Protenix)
- Benchmarks: [Protenix v1.0.0 model benchmark](https://github.com/bytedance/Protenix/blob/main/docs/model_1.0.0_benchmark.md)

## Related Tools

**Used together:**
- `colabfold-search`: MSA generation used internally by Protenix when `use_msa=True`
- `proteinmpnn`: Design sequences for predicted structures, then validate with Protenix

**Alternatives:**
- `alphafold3-prediction`: Original AlphaFold3 (requires API access)
- `esmfold-prediction`: Fast single-chain protein folding without MSA (no DNA/RNA/ligand support)
- `chai-prediction`: Another open-source multi-modal structure prediction model
