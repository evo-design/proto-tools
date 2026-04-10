<a href="https://bio-pro.mintlify.app/tools/structure-prediction/chai1"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Chai1

## Overview

Chai1 is a multi-modal structure prediction model from Chai Discovery that predicts 3D structures of proteins, ligands, and glycans using a [diffusion](https://en.wikipedia.org/wiki/Diffusion_model)-based architecture. It excels at modeling protein-ligand complexes and can incorporate ESM embeddings for improved accuracy.

## Background

**What does this tool measure/predict?**
Chai1 predicts the 3D atomic coordinates of biomolecular complexes from sequences. It outputs full-atom structures for proteins, ligands, and glycans with confidence scores including per-residue pLDDT, interface pTM (ipTM), and overall structure confidence.

**Why is this important?**
Understanding how proteins interact with small molecules is critical for:
- Drug design and virtual screening (predicting binding poses)
- Validating designed protein-ligand interactions
- Modeling glycoprotein structures with attached carbohydrates
- Predicting binding site conformations
- Screening compound libraries against protein targets

**Scientific foundation:**
Chai1 uses a modern deep learning architecture combining:
1. **ESM-2 embeddings** (optional): Pre-trained protein language model representations that capture evolutionary and structural patterns from millions of protein sequences.
2. **Diffusion-based structure generation**: A generative modeling approach that iteratively denoises random coordinates into realistic 3D structures, naturally handling the multi-modal nature of biomolecular complexes.
3. **Trunk network**: An attention-based architecture that processes sequence features through multiple recycling passes for iterative refinement.

Confidence metrics include:
- **pLDDT** (predicted Local Distance Difference Test): Per-residue confidence score (0-100), where >90 indicates high confidence, 70-90 is moderate, and <70 suggests low confidence.
- **pTM** (predicted Template Modeling score): Overall structure accuracy (0-1), where >0.8 indicates high confidence in the global fold.
- **ipTM** (interface pTM): Confidence in inter-chain interfaces (0-1), critical for multi-chain complexes.

## Execution Modes

Chai1 requires GPU with >=40GB VRAM (H100 recommended; can run on A100-80GB).

- **Local execution**: Runs on local GPU. Runtime ~30-120 seconds per complex on H100.

## How It Works

**Method overview:**
Chai1 employs a multi-stage architecture:
1. **MSA generation (optional):** When `use_msa=True`, protein sequences are processed using ColabFold search to retrieve multiple sequence alignments, providing evolutionary context. MSA search can use remote ColabFold API or local databases.
2. **Input encoding:** Protein sequences are encoded using ESM-2 embeddings (optional but recommended) to capture evolutionary context. MSA features are integrated when available. Ligands are represented as SMILES strings and encoded with chemical featurizers. Glycans use specialized carbohydrate representations.
3. **Trunk network processing:** The encoded features pass through a transformer-based trunk network with multiple recycling passes. Each recycle refines the intermediate representation by incorporating inter-chain and inter-molecular attention.
4. **Diffusion-based structure prediction:** Starting from random coordinates, a diffusion model iteratively denoises positions over many timesteps to generate physically realistic 3D structures. Multiple samples can be generated and ranked by confidence.

**Key assumptions:**
- The complex forms a stable, defined 3D structure
- Ligand binding is thermodynamically favorable (no covalent modifications)
- Glycan attachment points are correctly specified in the input
- Entity types (protein/ligand/glycan) are correctly identified

**Limitations:**
- **Maximum length:** 2,048 residues/atoms total across all chains
- **No DNA/RNA:** Cannot model nucleic acid complexes (use Boltz2)
- **No covalent ligands:** Cannot model covalent inhibitors or covalent modifications
- **Single conformation:** Predicts one structure, not conformational ensembles
- **No explicit water:** Does not model structural water molecules
- **Glycan complexity:** Limited to standard glycan building blocks

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | *required* | List of complexes; each can have protein chains, ligands, and/or glycans |
| `use_esm_embeddings` | `bool` | `True` | Use ESM-2 embeddings for improved protein representations |
| `use_msa` | `bool` | `True` | Whether to generate and use MSAs for protein chains using ColabFold search |
| `colabfold_search_config.search_mode` | `str` | `"remote"` | MSA search mode; `"remote"` uses ColabFold API, `"local"` uses local databases |
| `colabfold_search_config.use_metagenomic` | `bool` | `False` | Include metagenomic sequences in MSA search for improved coverage |
| `num_trunk_recycles` | `int` | `3` | Iterative refinement passes; more = better quality but slower |
| `num_diffn_timesteps` | `int` | `200` | Denoising steps; more = finer structures but slower |
| `num_diffn_samples` | `int` | `1` | Independent structure samples; best is returned |
| `num_trunk_samples` | `int` | `1` | Trunk forward passes per diffusion sample |
| `seed` | `int` | `42` | Random seed for reproducibility |

## Configuration

### Parameter Guides

| Parameter | Sweep Range | Notes |
|-----------|-------------|-------|
| `use_esm_embeddings` | `True, False` | Always use `True` unless debugging |
| `num_trunk_recycles` | `1 - 10` | Higher = more refined but slower |
| `num_diffn_timesteps` | `100 - 500` | Higher = finer structures; 200 is good default |
| `num_diffn_samples` | `1 - 5` | More samples explore binding pose diversity |
| `num_trunk_samples` | `1 - 3` | Multiple trunk passes per diffusion sample |

### Sweep Priorities

1. **`use_esm_embeddings`**: Most impactful for protein structure quality. Always use `True` unless debugging.
2. **`num_trunk_recycles`**: Affects structure refinement quality. Try 3, 5, 10 to balance speed vs accuracy.
3. **`num_diffn_timesteps`**: For high-stakes predictions, increase to 400-500. Use 100-200 for rapid screening.

## Output Specification

```python
# Return type: Chai1Output
Chai1Output(
    structures: List[Structure],  # One per input complex
)

# Each Structure contains metrics:
metrics = {
    "avg_plddt": float,        # Average pLDDT across all residues (0-100)
    "ptm": float,              # Predicted TM-score (0-1), may be None
    "iptm": float,             # Interface pTM (0-1), may be None
    "confidence_score": float, # Aggregate confidence, may be None
}
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `avg_plddt` | `float` | `0 - 100` | Mean per-residue confidence; >90 = excellent, 70-90 = good, <70 = uncertain |
| `ptm` | `float` | `0.0 - 1.0` | Global fold confidence; >0.8 = reliable topology |
| `iptm` | `float` | `0.0 - 1.0` | Interface confidence; >0.8 = reliable interactions between chains |
| `confidence_score` | `float` | `0.0 - 1.0` | Aggregate ranking score combining multiple metrics |

## Interpreting Results

**Thresholds & decision boundaries:**
- **Excellent:** `avg_plddt > 90` and `iptm > 0.8`: High confidence; structure suitable for detailed analysis
- **Acceptable:** `70 < avg_plddt <= 90` and `iptm > 0.6`: Moderate confidence; verify key interactions manually
- **Poor:** `avg_plddt <= 70` or `iptm <= 0.6`: Low confidence; binding pose may be incorrect; consider redesigning

**Tips for interpreting output:**
- For protein-ligand binding, `iptm` is more informative than `avg_plddt` about binding confidence
- Filter by `iptm > 0.6` as a first-pass quality filter for binding predictions
- Visualize binding poses (PyMOL, ChimeraX) colored by pLDDT to identify uncertain regions
- Very small ligands (<10 atoms) may have high uncertainty; binding pose might be less reliable
- Flexible ligands: predictions represent one conformation; actual binding may involve conformational sampling

## Quick Start Examples

```python
from proto_tools.tools.structure_prediction.chai1 import (
    run_chai1_prediction,
    Chai1PredictionInput,
    Chai1PredictionConfig,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    ProteinChain,
    LigandChain,
)

# Protein-ligand complex prediction
inputs = Chai1PredictionInput(
    complexes=[
        StructurePredictionComplex(
            protein_chains=[ProteinChain(sequence="MKTVRQ...")],
            ligand_chains=[LigandChain(sequence="CC(=O)Oc1ccccc1C(O)=O")],  # Aspirin SMILES
        )
    ]
)
config = Chai1PredictionConfig(
    use_esm_embeddings=True,
    num_trunk_recycles=3,
    num_diffn_timesteps=200,
)
result = run_chai1_prediction(inputs, config)

# Check results
for structure in result.structures:
    print(f"avg_pLDDT: {structure.metrics['avg_plddt']:.1f}")
    print(f"ipTM: {structure.metrics.get('iptm', 'N/A')}")
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`num_trunk_recycles`**:
  - Low values (1-2): Fast but less refined; use for initial screening
  - High values (5-10): More refined structures; use for final predictions
- **`num_diffn_timesteps`**:
  - Low values (100-150): Fast; acceptable for screening large libraries
  - High values (300-500): More accurate; use for publication-quality structures
- **`num_diffn_samples`**:
  - Single sample (1): Fastest; often sufficient with high recycling
  - Multiple samples (3-5): Better for exploring binding pose diversity

**Common mistakes:**
1. **Including DNA/RNA chains:** Chai1 only supports protein, ligand, glycan. For nucleic acids, use Boltz2.
2. **Exceeding 2,048 residue limit:** Always verify total length including ligand atoms. For large proteins, consider trimming to binding site region.
3. **Ignoring iptm for complexes:** For protein-ligand binding, `iptm` is more informative than `avg_plddt` about binding confidence.
4. **Using default samples for critical predictions:** Increase `num_diffn_samples` to 3-5 for important binding pose predictions.
5. **Wrong SMILES format for ligands:** Ensure ligand sequences are valid SMILES strings, not amino acid sequences.

**Edge cases to watch for:**
- **Very small ligands (<10 atoms):** May have high uncertainty; binding pose might be less reliable
- **Flexible ligands:** Predictions represent one conformation; actual binding may involve conformational sampling
- **Allosteric sites:** May require longer proteins to capture allosteric communication
- **Metal-binding sites:** Metals are not directly modeled; predictions may be less accurate near metal sites
- **Highly charged ligands:** Electrostatic interactions may be less accurate without explicit solvent

## Seed Reproducibility

Fresh subprocesses with the same seed produce identical structures, but
consecutive calls inside the same persistent worker drift by ~3 mÅ due
to hidden CUDA/JIT state in `chai_lab` that torch does not expose a
reset API for. Use one-shot execution (not `persist_tool`) for bit-exact
reproducibility across repeat calls. Upstream confirmation that some
chai_lab operations are inherently non-deterministic:

- [chaidiscovery/chai-lab#228 — guarantee result reproducibility with seed](https://github.com/chaidiscovery/chai-lab/issues/228)
- [chaidiscovery/chai-lab#246 — How does the seed number influence the results?](https://github.com/chaidiscovery/chai-lab/issues/246)

## References

**Primary publication:**
- Chai Discovery Team (2024). "Chai-1: Decoding the molecular interactions of life". *bioRxiv*. [DOI: 10.1101/2024.10.10.615955](https://www.biorxiv.org/content/10.1101/2024.10.10.615955)
- Summary: Introduces Chai1 as an open-source multi-modal structure prediction model achieving competitive accuracy with AlphaFold3 on protein-ligand, protein-protein, and protein-glycan complexes.

**Implementation:**
- GitHub: [https://github.com/chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab)
- Documentation: [https://chaidiscovery.com/blog/introducing-chai-1](https://chaidiscovery.com/blog/introducing-chai-1)

**Additional resources:**
- Chai Discovery: [https://www.chaidiscovery.com](https://www.chaidiscovery.com)
- Model weights: Available via Hugging Face through chai-lab package

## Related Tools

**Tools often used together:**
- **`esm2-embedding`**: Generate embeddings for sequence analysis; Chai1 uses ESM-2 internally for protein representation
- **`esmfold`**: Quick protein-only structure validation before running expensive protein-ligand predictions

**Alternative tools:**
- **`boltz2-prediction`**: Supports DNA/RNA complexes that Chai1 cannot handle; use for nucleic acid interactions
- **`esmfold`**: Much faster for protein-only structures; use when no ligands/glycans needed
- **`alphafold2`**: Higher accuracy for protein-only; use for final validation of protein folds
