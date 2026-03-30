<a href="https://bio-pro.mintlify.app/tools/structure-prediction/alphafold2"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AlphaFold2

## Overview

AlphaFold2 predicts 3D protein structures from amino acid sequences using the original DeepMind model via the ColabDesign JAX wrapper (`alphafold2-prediction`). It supports optional [multiple sequence alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSA) generation via ColabFold search for improved accuracy, and can predict both monomeric and multimeric protein structures.

## When to Use This Tool

**Primary use cases:**
- High-accuracy protein structure prediction when quality matters more than speed
- Final validation of designed protein sequences after initial ESMFold screening
- Monomeric structure prediction with MSA-enhanced accuracy
- Benchmarking designed sequences against state-of-the-art structure prediction

**When NOT to use this tool:**
- Rapid high-throughput screening of many sequences: use ESMFold (`esmfold-prediction`) instead (60x faster)
- Heteromeric protein-protein complexes with distinct chains: use Boltz2 (`boltz2-prediction`) or AlphaFold3 (`alphafold3-prediction`)
- Protein-ligand or protein-nucleic acid complexes: use Boltz2 or AlphaFold3 (AlphaFold2 supports proteins only)
- Conformational dynamics or ensemble generation: use BioEmu (`bioemu-simulation`)
- Backbone design (inverse problem): use RFDiffusion3 (`rfdiffusion3-generate`)

**Comparison with alternatives:**
- **AlphaFold2 vs ESMFold:** AlphaFold2 is more accurate (especially with MSAs) but ~60x slower. Use ESMFold for screening, AlphaFold2 for final validation.
- **AlphaFold2 vs AlphaFold3:** AlphaFold3 handles proteins, nucleic acids, ligands, and modifications. AlphaFold2 is protein-only but uses the well-validated original architecture.
- **AlphaFold2 vs Boltz2/Chai-1:** Boltz2 and Chai-1 are next-generation structure predictors supporting all biomolecular types. AlphaFold2 remains a strong baseline for protein-only prediction.

## Biological Background

**What does this tool measure/predict?**
AlphaFold2 predicts the 3D atomic coordinates of protein structures from amino acid sequences. It outputs full-atom protein structures with per-residue confidence scores (pLDDT), global structure quality metrics (pTM), and inter-chain confidence for multimers (ipTM).

**Why is this important?**
- Protein engineering: validate that designed sequences fold into intended structures
- Drug discovery: predict target protein structures for virtual screening
- Functional annotation: infer function from predicted 3D structure
- Protein design pipelines: final quality gate before experimental validation
- Structural biology: generate models for proteins without experimental structures

**Scientific foundation:**
AlphaFold2 uses a two-track architecture combining evolutionary and structural information:

1. **Multiple Sequence Alignment (MSA) processing:** Evolutionary information from homologous sequences is processed through the Evoformer module, which uses axial attention to capture coevolutionary patterns that constrain 3D structure.
2. **Structure module:** Iteratively refines 3D coordinates using invariant point attention (IPA), which operates directly in 3D space respecting rotational and translational symmetry.
3. **Recycling:** The prediction is iteratively refined by feeding outputs back through the network (configurable via `num_recycles`), progressively improving accuracy.
4. **Ensemble averaging:** Multiple independently trained model parameter sets (1-5) can be averaged for higher confidence predictions.

The model was trained on experimentally determined structures from the [Protein Data Bank](https://www.rcsb.org/) and achieves near-experimental accuracy for many protein families.

## Execution Modes

- **Local GPU (recommended):** Requires GPU with >=16GB VRAM. Runtime varies with sequence length: ~30-120 seconds per monomer (100-500 residues) on A100 GPU, depending on MSA usage and recycling.
- **CPU:** Possible but extremely slow (minutes to hours per prediction). Use only for testing with very short sequences.

## How It Works

**Method overview:**
1. **MSA generation (optional):** If `use_msa=True`, ColabFold search queries sequence databases (UniRef, environmental sequences) to build a multiple sequence alignment. MSAs capture evolutionary covariation that strongly constrains structure prediction. Currently supported for single-chain predictions only.
2. **Model inference:** The ColabDesign JAX wrapper runs the AlphaFold2 neural network with the specified number of recycling iterations and model parameters.
3. **Output processing:** Raw predictions are converted to Structure objects with PDB coordinates, confidence metrics (pLDDT, pTM, ipTM, PAE), and metadata.

**Key assumptions:**
- Input sequences are valid protein sequences (standard amino acids plus 'X' for unknown)
- The protein folds into a stable 3D structure (not intrinsically disordered)
- For MSA mode: homologous sequences exist in public databases
- For multimers: MSA generation is not yet supported (runs in single-sequence mode)

**Limitations:**
- Proteins only: no DNA, RNA, ligands, glycans, or post-translational modifications
- MSA generation currently limited to single-chain predictions
- Maximum sequence length constrained by GPU memory (typically ~2,000 residues on 40GB GPU)
- Single static structure prediction (no conformational ensembles)
- Accuracy depends on availability of evolutionary information for MSA-based predictions

**Computational requirements:**
- **Hardware:** GPU with >=16GB VRAM (A100 40GB recommended for longer sequences)
- **Runtime:** ~30-120s per monomer on A100 GPU with MSA; ~10-30s without MSA
- **Scalability:** Processes complexes sequentially; use `ToolInstance.persist()` for batch workloads

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | *required* | Protein complexes to predict. Each complex contains one or more protein chains. Accepts shorthand: `["MKTV..."]` for a single chain, or full `StructurePredictionComplex` objects for multi-chain. |

**StructurePredictionComplex fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chains` | `List[Chain]` | Chains in the complex. Each chain has `sequence` (str) and optional `entity_type` (auto-detected as "protein"). |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_msa` | `bool` | `True` | Generate MSAs via ColabFold search for improved accuracy. Set `False` for faster single-sequence mode. Only works for single-chain predictions. |
| `num_recycles` | `int` | `3` | Recycling iterations (0-48). Higher = more refined but slower. 3 is standard; 10-20 for difficult targets. |
| `model_num` | `int` | `1` | Which AF2 parameter set to use (1-5). Different sets can produce different predictions. Mutually exclusive with `num_ensemble_models > 1`. |
| `num_ensemble_models` | `int` | `1` | Number of model sets to run and average (1-5). Higher = better quality, linearly more compute. Mutually exclusive with custom `model_num`. |
| `seed` | `Optional[int]` | `None` | Random seed for reproducibility. |
| `device` | `str` | `"cuda"` | Device for inference (`"cuda"` or `"cpu"`). |
| `colabfold_search_config` | `Optional[ColabfoldSearchConfig]` | `None` | Advanced ColabFold MSA search settings. Uses defaults if None. |

### Parameter Guides

**Recycling iterations:**

| num_recycles | Quality | Speed | Use Case |
|-------------|---------|-------|----------|
| 1 | Lower | Fastest | Quick screening, prototyping |
| 3 | Good (default) | Moderate | Standard predictions |
| 10-20 | Higher | Slow | Difficult targets, final validation |
| 48 | Marginal gains | Very slow | Diminishing returns; rarely needed |

**Ensemble models:**

| num_ensemble_models | Quality | Speed | Use Case |
|--------------------|---------|-------|----------|
| 1 | Standard (default) | 1x | Most predictions |
| 3 | Improved | 3x | Important targets |
| 5 | Best | 5x | Publication-quality predictions |

### Sweep Priorities

1. **`use_msa`**: Largest impact on accuracy. MSA-based predictions are significantly better for proteins with evolutionary relatives. Sweep `[True, False]` to compare.
2. **`num_recycles`**: Second most impactful. Sweep `[3, 10, 20]` for important targets.
3. **`num_ensemble_models`**: Sweep `[1, 3, 5]` for final validation runs.

## Output Specification

```python
# Return type: AlphaFold2Output (extends StructurePredictionOutput)
AlphaFold2Output(
    structures: List[Structure],  # One per input complex
    metadata: dict,               # {"num_complexes": int, "total_chains": int}
    success: bool,
    errors: List[str],
)

# Each Structure contains:
Structure(
    avg_plddt: float,       # Average per-residue confidence (0.0-1.0)
    ptm: float,             # Predicted TM-score (0.0-1.0)
    iptm: Optional[float],  # Interface pTM for multimers (0.0-1.0)
    avg_pae: Optional[float], # Average predicted aligned error (lower = better)
    # Plus PDB coordinates, per-residue pLDDT in B-factor column, etc.
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `avg_plddt` | `float` | `0.0 - 1.0` | Mean per-residue confidence. >0.9 = well-folded, 0.7-0.9 = moderate, <0.7 = disordered/uncertain |
| `ptm` | `float` | `0.0 - 1.0` | Global fold confidence. >0.8 = reliable topology, <0.5 = fold likely incorrect |
| `iptm` | `Optional[float]` | `0.0 - 1.0` | Interface confidence for multimers. >0.8 = confident interface, <0.5 = unreliable. None for monomers. |
| `avg_pae` | `Optional[float]` | `0.0 - 30+` | Average predicted aligned error in Angstroms. Lower = more confident relative positioning. |

**Supported export formats:** `pdb`, `cif`, `json`

## Interpreting Results

**Thresholds & decision boundaries:**
- **Excellent:** `avg_plddt > 0.9` and `ptm > 0.8` — High-confidence prediction. Structure is reliable for downstream analysis (docking, design, function annotation). Proceed to experimental validation.
- **Good:** `0.8 < avg_plddt <= 0.9` and `ptm > 0.7` — Moderate confidence. Core domains likely correct; flexible regions may be uncertain. Check per-residue pLDDT for problem areas.
- **Marginal:** `0.7 < avg_plddt <= 0.8` — Mixed confidence. Some well-folded regions but significant uncertainty. Consider redesigning low-confidence regions or trying MSA-enhanced prediction.
- **Poor:** `avg_plddt <= 0.7` or `ptm < 0.5` — Low confidence. Structure likely unreliable. Sequence may be intrinsically disordered, or the model lacks sufficient evolutionary information. Redesign or use alternative methods.

**For multimeric complexes:**
- `iptm > 0.8`: Confident interface prediction
- `iptm < 0.5`: Interface likely unreliable; consider Boltz2 or AlphaFold3 for complex modeling

**Interpreting edge cases:**
- High avg_plddt but low pTM can indicate well-folded domains with incorrect relative orientation
- Low pLDDT regions may be biologically relevant (flexible linkers, disordered regions) rather than prediction failures
- MSA-free predictions (`use_msa=False`) may have systematically lower confidence for well-characterized protein families — this reflects reduced information, not necessarily poor structure

## Quick Start Examples

**Example 1: Basic single-chain prediction (MSA-enhanced)**
```python
from bio_programming_tools.tools.structure_prediction.alphafold2 import (
    run_alphafold2, AlphaFold2Input, AlphaFold2Config,
)

# Predict structure of a single protein
inputs = AlphaFold2Input(complexes=["MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"])
config = AlphaFold2Config(use_msa=True, num_recycles=3)

result = run_alphafold2(inputs, config)

structure = result.structures[0]
print(f"avg_pLDDT: {structure.avg_plddt:.3f}")
print(f"pTM: {structure.metrics['ptm']:.3f}")
```

**Example 2: Fast single-sequence mode (no MSA)**
```python
from bio_programming_tools.tools.structure_prediction.alphafold2 import (
    run_alphafold2, AlphaFold2Input, AlphaFold2Config,
)

# Faster prediction without MSA — useful for screening designed sequences
# that may not have evolutionary relatives
inputs = AlphaFold2Input(complexes=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDI"])
config = AlphaFold2Config(use_msa=False, num_recycles=3)

result = run_alphafold2(inputs, config)
print(f"avg_pLDDT: {result.structures[0].avg_plddt:.3f}")
```

**Example 3: High-quality ensemble prediction**
```python
from bio_programming_tools.tools.structure_prediction.alphafold2 import (
    run_alphafold2, AlphaFold2Input, AlphaFold2Config,
)

# Average 5 model parameter sets for highest quality
inputs = AlphaFold2Input(complexes=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDI"])
config = AlphaFold2Config(
    use_msa=True,
    num_recycles=10,           # More recycling for difficult targets
    num_ensemble_models=5,     # Average all 5 AF2 parameter sets
)

result = run_alphafold2(inputs, config)
s = result.structures[0]
print(f"avg_pLDDT: {s.avg_plddt:.3f}, pTM: {s.metrics['ptm']:.3f}")
```

**Example 4: Batch prediction with persistence**
```python
from bio_programming_tools.tools.structure_prediction.alphafold2 import (
    run_alphafold2, AlphaFold2Input, AlphaFold2Config,
)
from bio_programming_tools.utils.tool_instance import ToolInstance

sequences = [
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK",
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDI",
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTV",
]

config = AlphaFold2Config(use_msa=False, num_recycles=3)

# Keep model loaded across predictions for batch efficiency
with ToolInstance.persist():
    for seq in sequences:
        result = run_alphafold2(AlphaFold2Input(complexes=[seq]), config)
        s = result.structures[0]
        status = "PASS" if s.avg_plddt > 0.8 else "REVIEW"
        print(f"pLDDT={s.avg_plddt:.2f} [{status}]: {seq[:30]}...")
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`use_msa`**:
  - `True` (default): Best accuracy for natural proteins with evolutionary relatives. Adds ~30-60s for MSA search.
  - `False`: Use for designed/de novo sequences that lack evolutionary relatives, or for faster screening.
- **`num_recycles`**:
  - Low (1-3): Fast predictions, sufficient for most well-behaved proteins
  - High (10-20): For difficult targets where initial predictions have low confidence
  - Very high (>20): Diminishing returns; rarely justified
- **`num_ensemble_models`**:
  - 1 (default): Standard quality, fastest
  - 3-5: For publication-quality predictions or when confidence is borderline

**Common mistakes:**
1. **Using MSA mode for de novo designed sequences:** Designed sequences often have no evolutionary relatives, so MSA adds compute time without improving accuracy. Set `use_msa=False` for designed sequences.
2. **Setting both `model_num` and `num_ensemble_models`:** These are mutually exclusive. Use `model_num` to test specific parameter sets, or `num_ensemble_models` to average multiple sets.
3. **Expecting multi-chain MSA support:** MSA generation currently works for single-chain predictions only. Multi-chain predictions automatically run without MSA.
4. **Not using persistence for batch workloads:** Model loading takes significant time. Use `ToolInstance.persist()` when predicting multiple structures.

**Tips for optimal results:**
- Screen with ESMFold first (fast), then validate top candidates with AlphaFold2 (accurate)
- For designed proteins, compare `use_msa=True` vs `use_msa=False` — if scores are similar, MSA isn't helping
- Use `num_ensemble_models=3` or higher for important final predictions
- Check per-residue pLDDT (in B-factor column of PDB output) to identify problematic regions

**Edge cases to watch for:**
- Very short sequences (<30 residues): Low pLDDT may reflect genuine flexibility (peptides), not prediction failure
- Sequences with many 'X' (unknown) residues: Predictions will have lower confidence in those regions
- Multi-chain complexes: MSA is skipped; predictions may be less accurate than single-chain
- GPU memory limits: Very long sequences (>1500 residues) may require GPUs with >24GB VRAM

## References

**Primary publication:**
- Jumper, J. et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*, 596(7873), 583-589. [DOI: 10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- Summary: Introduces AlphaFold2, which achieves near-experimental accuracy in protein structure prediction by combining evolutionary information (MSAs) with a novel neural network architecture using attention mechanisms and equivariant transformations.

**Implementation:**
- ColabDesign (JAX wrapper): [https://github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- Original AlphaFold2: [https://github.com/google-deepmind/alphafold](https://github.com/google-deepmind/alphafold)

**Additional resources:**
- AlphaFold Protein Structure Database: [https://alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk) - precomputed structures for 200M+ proteins
- ColabFold: [https://github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) - fast MSA generation and structure prediction

## Related Tools

**Tools often used together:**
- **`colabfold-search`**: MSA generation used internally by AlphaFold2 when `use_msa=True`. Can also be called independently for custom MSA workflows.
- **`esmfold-prediction`**: Fast structure prediction for initial screening before AlphaFold2 validation.
- **`structure-metrics`**: Compute structural quality metrics (helix length, radius of gyration) on predicted structures.
- **`tmalign-alignment`** / **`usalign-alignment`**: Compare predicted structures to reference structures using TM-score.

**Alternative tools (similar function):**
- **`esmfold-prediction`**: 60x faster but slightly less accurate. Best for high-throughput screening.
- **`alphafold3-prediction`**: Handles proteins, nucleic acids, ligands, and modifications. Use for non-protein or complex biomolecular predictions.
- **`boltz2-prediction`**: Next-generation predictor supporting all biomolecular types including protein-ligand complexes.
- **`chai1-prediction`**: Alternative multi-modal structure predictor with similar capabilities to Boltz2.
- **`protenix-prediction`**: Open-source AlphaFold3 implementation.
