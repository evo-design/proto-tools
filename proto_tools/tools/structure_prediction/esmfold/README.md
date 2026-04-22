<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESMFold

## Overview

ESMFold is a fast protein structure prediction model from Meta AI that predicts 3D structures directly from amino acid sequences using a [language model](https://en.wikipedia.org/wiki/Language_model) approach, without requiring [multiple sequence alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment).

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

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | *required* | Complexes to predict. Accepts `StructurePredictionComplex` objects, sequence strings for single-chain complexes, or lists of sequence strings for multi-chain complexes. Total residues per complex must be <=2,400. |
| `msas` | `Dict[str, MSA] \| None` | `None` | Hidden advanced field inherited from shared structure inputs. ESMFold does not require MSAs and normally leaves this unset. |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `residue_idx_offset` | `int` | `512` | Residue numbering gap between chains in multi-chain structures; rarely needs adjustment. |
| `chain_linker` | `str` | `"G" * 25` | Internal glycine linker sequence used to connect chains before prediction; removed/relabelled in the output structure. |
| `max_batch_residues` | `int` | `1200` | Maximum total residues per inference batch; lower this if GPU memory is tight. |
| `device` | `str` | `"cuda"` | Execution device, inherited from `StructurePredictionConfig`. |

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
