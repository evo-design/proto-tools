<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESMFold

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

ESMFold is a fast protein structure prediction model from Meta AI that predicts 3D structures directly from amino acid sequences using a [language model](https://en.wikipedia.org/wiki/Language_model) approach, without requiring [multiple sequence alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment).

This package also includes `esmfold-gradient`, a differentiable confidence tool that runs ESMFold over a relaxed `(L, 20)` logits distribution for one designated target chain in a complex and returns the gradient of a weighted confidence loss (pLDDT, pTM, pAE) with respect to those input logits. It can be used as a structure-aware loss inside MCMC, gradient descent, or any optimization loop over relaxed protein sequences.

## Tool Catalog

| Tool | Description | Output |
|------|-------------|--------|
| `esmfold-prediction` | Predict 3D structure from sequence | Structure(s), pLDDT, pTM, optional PAE |
| `esmfold-gradient` | Differentiable confidence loss + gradient over relaxed logits | Gradient, weighted loss, per-term metrics, predicted structure |

## Background

**What does this tool measure/predict?**
ESMFold predicts the 3D atomic coordinates of protein structures from amino acid sequences. It outputs full-atom protein structures with confidence scores for each residue (pLDDT) and overall structure quality metrics (pTM score).

**Why is this important?**
Protein structure determines function. Knowing whether a designed protein will fold into a stable, well-defined 3D structure is critical for:
- Validating that designed proteins will actually fold (not be disordered)
- Predicting whether domains will adopt intended conformations
- Identifying flexible vs rigid regions
- Evaluating oligomeric assembly states (dimers, trimers, etc.)
- Screening out poorly-folded or aggregation-prone designs

**Scientific foundation:**
ESMFold uses the ESM-2 protein language model to generate structure-aware embeddings, which are then processed through a structure prediction head based on AlphaFold2's architecture. The model learns protein structure from sequence patterns alone, without needing evolutionary information (MSAs). Confidence metrics include:
- **pLDDT** (predicted Local Distance Difference Test): Per-residue confidence score (0-100), where >90 indicates high confidence, 70-90 is moderate, and <70 suggests disorder or low confidence.
- **pTM** (predicted Template Modeling score): Overall structure accuracy (0-1), where >0.8 indicates high confidence in the global fold.

## Tools

### ESMFold Gradient (`esmfold-gradient`)

Run one differentiable ESMFold confidence pass.

This is the gradient counterpart to :func:`run_esmfold`: one target-chain
logit matrix is relaxed into ESMFold's sequence pathway, all requested
confidence terms are summed into a single weighted loss, and one backward
pass returns `d(loss) / d(logits)`.

### ESMFold Structure Prediction (`esmfold-prediction`)

Predict protein 3D structures using ESMFold.

Uses ESMFold, a fast transformer-based protein structure prediction model from
Meta AI, to predict 3D structures without requiring multiple sequence alignments.
Supports local GPU execution with automatic batching for memory efficiency.

## Execution Modes

ESMFold requires GPU with >=16GB VRAM (24GB recommended for longer sequences).

- **Local execution**: Runs on local GPU. Runtime ~5-30 seconds per monomer (100-400 residues) on A100 GPU; scales with sequence length squared.

## How It Works

**Method overview:**
ESMFold uses a two-stage approach:
1. **Sequence encoding:** The ESM-2 language model (650M parameters) processes the amino acid sequence and generates contextual embeddings for each residue, capturing evolutionary and structural patterns learned from millions of protein sequences.
2. **Structure decoding:** These embeddings are fed into a structure module (based on AlphaFold2's Evoformer and structure prediction heads) that predicts 3D coordinates, confidence scores, and inter-residue distances.

Unlike AlphaFold2, ESMFold does not require multiple sequence alignments (MSAs), making it much faster but potentially less accurate for proteins with rich evolutionary information.

**Key assumptions:**
- The protein sequence folds into a single stable structure (not intrinsically disordered)
- The structure can be predicted from sequence patterns alone (no cofactors, post-translational modifications, or context-dependent folding)
- For oligomers: All chains are identical (homomeric) and fold symmetrically

**Limitations:**
- **Maximum length:** 2,400 residues total across all chains in a complex
- **Homomers only:** Cannot model heteromeric complexes (different chains with different sequences)
- **No ligands/cofactors:** Cannot include small molecules, metal ions, or post-translational modifications
- **Single conformation:** Predicts one structure, not conformational ensembles or dynamic regions
- **MSA-free tradeoff:** Slightly lower accuracy than AlphaFold2 for well-characterized protein families

**Differentiable confidence (gradient tool):**

The gradient tool replaces the discrete embedding lookup for one designated *target chain* with a soft mixture of amino-acid embeddings drawn from a relaxed `(L, 20)` distribution. The relaxed mixture is injected at two stages — into ESM-2's word embeddings and into ESMFold's own AA embedding layer — and the rest of the pipeline (ESM-2 stack, structure module trunk, confidence heads) runs as usual under autograd. A weighted combination of `1 − pLDDT`, `1 − pTM`, and `pAE / 31.75` produces a single scalar loss; one backward pass returns ∂loss / ∂logits while ESMFold's parameters stay frozen. Optional soft/hard mixing knobs (and a Straight-Through Estimator) let callers trade smoothness for guidance toward discrete sequences.

## Input Parameters

### Prediction Tool

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | *required* | Complexes to predict. Accepts `StructurePredictionComplex` objects, sequence strings for single-chain complexes, or lists of sequence strings for multi-chain complexes. Total residues per complex must be <=2,400. |
| `msas` | `Dict[str, MSA] \| None` | `None` | Hidden advanced field inherited from shared structure inputs. ESMFold does not require MSAs and normally leaves this unset. |

### Gradient Tool

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logits` | `List[List[float]]` | *required* | Relaxed target-chain logits with shape `(L, 20)` in canonical amino-acid order `ACDEFGHIKLMNPQRSTVWY` |
| `temperature` | `float` | `1.0` | Softmax temperature applied to logits before they are mixed into the relaxed embedding |
| `chains` | `List[str]` | *required* | Complete protein-chain sequences for the complex. Each sequence listed in `target_chain_indices` is replaced by the hard decode of `logits` before folding, but its length must equal `len(logits)` |
| `target_chain_indices` | `List[int]` | `[0]` | Zero-based chain indices that receive the relaxed input logits. Must reference distinct, in-bounds chains; gradients are summed through the shared logits tensor when more than one target chain is selected |

## Configuration

### Prediction Tool (`ESMFoldConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residue_idx_offset` | `int` | `512` | Residue numbering gap between chains in multi-chain structures; rarely needs adjustment. |
| `chain_linker` | `str` | `"G" * 25` | Internal glycine linker sequence used to connect chains before prediction; removed/relabelled in the output structure. |
| `max_batch_residues` | `int` | `1200` | Maximum total residues per inference batch; lower this if GPU memory is tight. |
| `device` | `str` | `"cuda"` | Execution device, inherited from `StructurePredictionConfig`. |

### Gradient Tool (`ESMFoldGradientConfig`)

Inherits every field above from `ESMFoldConfig` and adds:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_weights` | `dict[str, float]` | `{"plddt": 1.0}` | Non-negative weights for the confidence loss terms. Valid keys: `plddt` (uses `1 − avg_pLDDT`), `ptm` (uses `1 − pTM`), `pae` (uses `avg_PAE / 31.75`). Terms with weight `0.0` are skipped; if all weights are zero, the gradient short-circuits to a zero-gradient + `loss=0.0` forward pass. |
| `soft` | `float` | `1.0` | Blend between hard argmax one-hot (`0.0`) and full softmax probabilities (`1.0`) for the relaxed target sequence |
| `hard` | `float` | `0.0` | Straight-Through Estimator coefficient — `1.0` runs the forward pass on hard one-hot tokens while gradients still flow through soft probabilities |
| `compute_gradient` | `bool` | `True` | When `True`, runs the backward pass and returns the gradient. Set `False` for forward-only confidence scoring; `gradient` is `None` in the output |

### Parameter Guides

| Parameter | Sweep Range | Notes |
|-----------|-------------|-------|
| `complexes` chain count | `1 - 6` repeated chains | Determines the modeled assembly state. For a homodimer, pass a complex with two identical chains. GPU memory increases with total residues. |
| `chain_linker` | `10 - 100` glycines | Affects chain separation during multi-chain prediction. The default is 25 glycines. |
| `max_batch_residues` | `300 - 2400` | Controls batching, not model quality. Lower values reduce memory pressure. |

### Sweep Priorities

1. **Complex chain count**: Most impactful for oligomer design. Use `complexes=[["SEQ"]]` for monomer, `complexes=[["SEQ", "SEQ"]]` for homodimer, and so on.
2. **`chain_linker`**: Affects packing geometry for multi-chain predictions; try 15, 25, 50 glycines if default gives poor inter-chain contacts.
3. **`max_batch_residues`**: Tune for available GPU memory when batching many candidates.

## Output Specification

### ESMFoldOutput (Prediction)

```python
# Return type: ESMFoldOutput
ESMFoldOutput(
    structures: List[Structure],  # One per input complex
    success: bool,
    errors: List[str]
)

# Accessors on each returned Structure:
structure = result.structures[0]
structure.structure                 # Stored structure content
structure.structure_pdb             # PDB format property
structure.structure_cif             # mmCIF format property
structure.per_residue_plddt         # Per-residue pLDDT property, or None
structure.metrics                   # ESMFoldMetrics

# Metrics live under structure.metrics:
metrics = structure.metrics

ESMFoldMetrics(
    avg_plddt: float,             # Average pLDDT across all residues (0-1)
    ptm: float | None,            # Predicted TM-score (0-1)
    avg_pae: float | None,        # Average predicted aligned error
    pae: list[list[float]] | None # Full PAE matrix when requested
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `structure.metrics.avg_plddt` | `float` | `0.0 - 1.0` | Mean per-residue confidence; >0.9 = well-folded, <0.7 = disordered/uncertain |
| `structure.metrics.ptm` | `float \| None` | `0.0 - 1.0` | Global fold confidence; >0.8 = reliable topology, <0.5 = fold likely incorrect |
| `structure.per_residue_plddt` | `list[float] \| None` | `0.0 - 1.0` each | Identifies flexible/disordered regions vs well-folded domains |
| `structure.structure_pdb` | `str` | n/a | Predicted coordinates as PDB text |
| `structure.structure_cif` | `str` | n/a | Predicted coordinates as mmCIF text |

### ESMFoldGradientOutput

| Field | Type | Description |
|-------|------|-------------|
| `gradient` | `List[List[float]] \| None` | Gradient matrix with the same `(L, 20)` shape and amino-acid column order as the input logits. `None` when `compute_gradient=False` |
| `loss` | `float` | Scalar weighted confidence loss (sum of `loss_weights[k] * loss_terms[k]` over enabled terms) |
| `metrics` | `dict[str, Any]` | `avg_plddt`, `ptm`, `avg_pae`, optional `pae_matrix`, plus per-term unweighted losses (`loss_plddt`, `loss_ptm`, `loss_pae`) for whichever terms had non-zero weight |
| `vocab` | `List[str]` | Amino-acid column ordering for the input logits and returned gradient — always canonical protein order `ACDEFGHIKLMNPQRSTVWY` |
| `structure` | `Structure` | Predicted ESMFold complex structure for the hard-decoded sequence (same `Structure` type as `esmfold-prediction` returns) |

## Interpreting Results

**Thresholds & decision boundaries:**
- **Excellent:** `structure.metrics.avg_plddt > 0.9`: High confidence, well-folded structure suitable for most applications
- **Acceptable:** `0.7 < structure.metrics.avg_plddt <= 0.9`: Moderate confidence; some flexible or uncertain regions; review per-residue pLDDT
- **Poor:** `structure.metrics.avg_plddt <= 0.7`: Low confidence; likely disordered or poorly modeled; consider redesigning sequence

**Tips for interpreting output:**
- Average pLDDT can hide poorly-folded regions. Always check `structure.per_residue_plddt` to identify problem areas.
- Low pLDDT regions may be biologically relevant (e.g., flexible linkers, disordered regions). Context matters.
- Filter by `structure.metrics.avg_plddt > 0.8` as a first-pass quality filter during optimization
- Visualize structures (PyMOL, ChimeraX) colored by pLDDT to identify flexible regions
- For oligomers, check whether inter-chain contacts look reasonable; ESMFold may not accurately predict interfaces

## Quick Start Examples

```python
from proto_tools.tools.structure_prediction.esmfold import (
    run_esmfold,
    ESMFoldInput,
    ESMFoldConfig,
)
from proto_tools.tools.structure_prediction.shared_data_models import StructurePredictionComplex

# Single protein prediction
inputs = ESMFoldInput(
    complexes=[
        StructurePredictionComplex(chains=["MKTVRQERLKSIVRI..."])
    ]
)
config = ESMFoldConfig()
result = run_esmfold(inputs, config)

# Check results
for structure in result.structures:
    print(f"avg_pLDDT: {structure.metrics.avg_plddt:.3f}")
    if structure.metrics.ptm is not None:
        print(f"pTM: {structure.metrics.ptm:.3f}")

# Homodimer prediction: one complex with two identical chains
dimer_inputs = ESMFoldInput(
    complexes=[
        StructurePredictionComplex(chains=["MKTVRQERLKSIVRI...", "MKTVRQERLKSIVRI..."])
    ]
)
result_dimer = run_esmfold(dimer_inputs, config)
```

**Differentiable confidence gradient over a relaxed target chain:**
```python
from proto_tools.tools.structure_prediction.esmfold import (
    ESMFoldGradientInput, ESMFoldGradientConfig, run_esmfold_gradient,
)
from proto_tools.utils import one_hot_protein_logits

# Seed a relaxed target-chain distribution from a discrete sequence
target_seq = "MKTAYIAKQR"
logits = one_hot_protein_logits(target_seq, sharpness=2.0)

inputs = ESMFoldGradientInput(
    logits=logits,
    chains=[target_seq],          # length must equal len(logits)
    target_chain_indices=[0],     # which chains receive the relaxed distribution
)
config = ESMFoldGradientConfig(
    loss_weights={"plddt": 1.0, "ptm": 0.5},
    num_recycles=1,
)

result = run_esmfold_gradient(inputs, config)
print(f"weighted loss: {result.loss:.3f}")
print(f"avg pLDDT:    {result.metrics['avg_plddt']:.3f}")
print(f"loss_plddt:   {result.metrics['loss_plddt']:.3f}")
print(f"grad shape:   ({len(result.gradient)}, {len(result.gradient[0])})")
# Step the relaxed sequence: logits ← logits − lr · gradient
```

## Best Practices & Gotchas

**Parameter tuning:**
- **Complex chain count**:
  - One chain: Fast, use for most single-chain proteins
  - Repeated chains: Model homomeric assemblies; GPU memory increases with total residues; very high chain counts are rarely biologically relevant
- **`chain_linker`**:
  - Low values (10-15): For tightly packed or continuous chains
  - High values (50-100): For loosely associated or distant chains
- **`max_batch_residues`**:
  - Lower values: Safer on smaller GPUs
  - Higher values: Better throughput if memory is available

**Common mistakes:**
1. **Exceeding 2,400 residue limit:** Always check the total residues across all chains in each complex. For a 600-residue homomer, at most four chains fit the hard model limit.
2. **Interpreting low pLDDT as "bad" sequences:** Low pLDDT regions may be biologically relevant (e.g., flexible linkers, disordered regions). Context matters.
3. **Using ESMFold for heteromeric complexes:** ESMFold is best for single-chain proteins and simple homomeric assemblies. For A-B dimers, protein-ligand complexes, or modified chains, use Boltz2, AlphaFold2/3, Chai1, or Protenix.
4. **Ignoring per-residue pLDDT:** Average pLDDT can hide poorly-folded regions. Always check `per_residue_plddt` to identify problem areas.

**Edge cases to watch for:**
- **Very short sequences (<30 aa):** May have low pLDDT due to lack of structural constraints; this is often biologically realistic (e.g., peptides are flexible)
- **Highly repetitive sequences:** May produce extended or disordered structures with low confidence
- **Non-standard amino acids:** Replace with 'X' (unknown) or closest standard amino acid; ESMFold will predict but confidence may be lower
- **Large complexes approaching 2,400 residues:** May run out of GPU memory; reduce chain count, shorten sequences, or lower `max_batch_residues`

**Gradient tool:**

1. **Vocab order**: Input logits and the returned gradient share the canonical protein order `ACDEFGHIKLMNPQRSTVWY`. The tool maps to ESM-2 and OpenFold token indices internally.

2. **Target chain length**: `len(logits)` must equal the length of every chain referenced in `target_chain_indices`. Non-target chains can have any length (they fold normally with their fixed sequences).

3. **Repeated target chains**: For homomers where multiple chain occurrences share the same designed sequence, list each occurrence in `target_chain_indices` once — gradients sum through the shared logits tensor.

4. **`soft` / `hard` mixing**: Default (`soft=1.0`, `hard=0.0`) uses pure soft probabilities — best for smooth optimization over the relaxed simplex. Set `hard=1.0` for the Straight-Through Estimator (forward sees argmax tokens, backward flows through soft probs) when the relaxed forward diverges too far from the discrete fold.

5. **Loss weights**: Provide a `dict[str, float]` over `{"plddt", "ptm", "pae"}`. Setting all weights to `0.0` short-circuits to a forward-only discrete pass with `loss=0.0` and a zero gradient — a useful sanity check.

6. **Forward-only mode**: Set `compute_gradient=False` to skip the backward pass; `gradient` will be `None` but `loss`, `metrics`, and the predicted `structure` are still returned.

7. **Memory**: A single ESMFold gradient pass is dominated by the structure-module trunk activations. Long target chains (>500 residues) on a single 24 GB GPU may need `num_recycles=1` and a single `target_chain_index`.

## References

**Primary publication:**
- Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model". *Science*, 379(6637), 1123-1130. [DOI: 10.1126/science.ade2574](https://www.science.org/doi/10.1126/science.ade2574)
- Summary: Introduces ESMFold as an MSA-free structure prediction method that achieves AlphaFold2-like accuracy at 60x lower computational cost by using ESM-2 language model embeddings.

**Implementation:**
- GitHub: [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)
- Documentation: [https://github.com/facebookresearch/esm/tree/main/esm/esmfold](https://github.com/facebookresearch/esm/tree/main/esm/esmfold)

**Additional resources:**
- ESM Metagenomic Atlas: [https://esmatlas.com](https://esmatlas.com) - precomputed structures for 600M+ proteins
- Tutorial: [https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/ESMFold.ipynb)

## Related Tools

**Tools often used together:**
- **`esm3`**: Generate protein sequences likely to fold well, then validate with ESMFold
- **`esm2-embedding`**: Get sequence embeddings for similarity analysis; ESMFold uses ESM-2 internally

**Alternative tools:**
- **`alphafold2`**: Slower but more accurate; use for final validation after ESMFold screening
- **`boltz2`**: Handles heteromeric complexes and small molecules; use when you need protein-protein or protein-ligand interactions
