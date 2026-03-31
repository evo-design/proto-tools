<a href="https://bio-pro.mintlify.app/tools/structure-prediction/esmfold"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESMFold

## Overview

ESMFold is a fast protein structure prediction model from Meta AI that predicts 3D structures directly from amino acid sequences using a [language model](https://en.wikipedia.org/wiki/Language_model) approach, without requiring [multiple sequence alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment).

## When to Use This Tool

**Primary use cases:**
- **Rapid structure validation:** Quickly check if designed proteins are predicted to fold well
- **Filtering poorly-folded designs:** Screen out sequences with low pLDDT before more expensive experimental validation
- **Oligomer design:** Predict symmetric oligomers (dimers, trimers, etc.) by replicating sequences
- **Structure-based optimization:** Use as a constraint during sequence optimization to ensure foldability
- **Domain architecture validation:** Check if multi-domain proteins fold with expected domain arrangements

**When NOT to use this tool:**
- **High-accuracy structure modeling for publication:** Use AlphaFold2 or experimental methods (X-ray, [cryo-EM](https://en.wikipedia.org/wiki/Cryogenic_electron_microscopy)) for final structures
- **Protein-protein binding interfaces:** Use AlphaFold-Multimer or Boltz2 for heteromeric complex modeling with distinct chains
- **Conformational dynamics:** ESMFold predicts single static structures, not conformational ensembles
- **Very long proteins (>2,400 residues):** ESMFold has a hard limit; use AlphaFold2 or split into domains
- **Non-protein molecules:** ESMFold only handles amino acids (no DNA, RNA, ligands, glycans)

**Comparison with alternatives:**
- **ESMFold vs AlphaFold2:** ESMFold is 60x faster but slightly less accurate. Use ESMFold for high-throughput screening, AlphaFold2 for final validation.
- **ESMFold vs Boltz2:** Boltz2 handles heteromeric complexes and small molecules; ESMFold is better for homomeric oligomers and pure protein sequences.
- **ESMFold vs RosettaFold:** Similar speed/accuracy tradeoff; ESMFold is more widely adopted and easier to deploy.

## Biological Background

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
| `complexes` | `List[StructurePredictionComplex]` | *required* | List of protein complexes to predict; each complex can have multiple chains for homomers |
| `n_replications` | `int` | `1` | Number of times to replicate each chain (2=dimer, 3=trimer, etc.); total residues must be <=2,400 |
| `residue_idx_offset` | `int` | `512` | Offset for residue indexing; controls chain break tokens in model; rarely needs adjustment |
| `chain_linker` | `int` | `25` | Number of residues between chains; affects inter-chain geometry |

## Configuration

### Parameter Guides

| Parameter | Sweep Range | Notes |
|-----------|-------------|-------|
| `n_replications` | `1 - 6` | Determines assembly state; GPU memory increases linearly |
| `chain_linker` | `10 - 100` | Affects packing geometry for multi-chain predictions |

### Sweep Priorities

1. **`n_replications`**: Most impactful for oligomer design; determines assembly state. Sweep 1-4 to find optimal oligomeric state.
2. **`chain_linker`**: Affects packing geometry for multi-chain predictions; try 15, 25, 50 if default gives poor inter-chain contacts.

## Output Specification

```python
# Return type: ESMFoldOutput
ESMFoldOutput(
    structures: List[PredictedStructure],  # One per input complex
    success: bool,
    errors: List[str]
)

PredictedStructure(
    avg_plddt: float,              # Average pLDDT across all residues (0-1)
    ptm: float,                    # Predicted TM-score (0-1)
    structure_pdb_output: str,     # PDB format structure
    structure_cif_output: str,     # mmCIF format structure
    per_residue_plddt: List[float] # pLDDT for each residue (0-1)
)
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `avg_plddt` | `float` | `0.0 - 1.0` | Mean per-residue confidence; >0.9 = well-folded, <0.7 = disordered/uncertain |
| `ptm` | `float` | `0.0 - 1.0` | Global fold confidence; >0.8 = reliable topology, <0.5 = fold likely incorrect |
| `per_residue_plddt` | `List[float]` | `0.0 - 1.0` each | Identifies flexible/disordered regions vs well-folded domains |

## Interpreting Results

**Thresholds & decision boundaries:**
- **Excellent:** `avg_plddt > 0.9` — High confidence, well-folded structure suitable for most applications
- **Acceptable:** `0.7 < avg_plddt <= 0.9` — Moderate confidence; some flexible or uncertain regions; review per-residue pLDDT
- **Poor:** `avg_plddt <= 0.7` — Low confidence; likely disordered or poorly modeled; consider redesigning sequence

**Tips for interpreting output:**
- Average pLDDT can hide poorly-folded regions. Always check `per_residue_plddt` to identify problem areas.
- Low pLDDT regions may be biologically relevant (e.g., flexible linkers, disordered regions). Context matters.
- Filter by `avg_plddt > 0.8` as a first-pass quality filter during optimization
- Visualize structures (PyMOL, ChimeraX) colored by pLDDT to identify flexible regions
- For oligomers, check whether inter-chain contacts look reasonable; ESMFold may not accurately predict interfaces

## Quick Start Examples

```python
from proto_tools.tools.structure_prediction.esmfold import (
    run_esmfold_prediction,
    ESMFoldPredictionInput,
    ESMFoldPredictionConfig,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    ProteinChain,
)

# Single protein prediction
inputs = ESMFoldPredictionInput(
    complexes=[
        StructurePredictionComplex(
            protein_chains=[ProteinChain(sequence="MKTVRQERLKSIVRI...")],
        )
    ]
)
config = ESMFoldPredictionConfig()
result = run_esmfold_prediction(inputs, config)

# Check results
for structure in result.structures:
    print(f"avg_pLDDT: {structure.avg_plddt:.3f}")
    print(f"pTM: {structure.ptm:.3f}")

# Dimer prediction
config_dimer = ESMFoldPredictionConfig(n_replications=2)
result_dimer = run_esmfold_prediction(inputs, config_dimer)
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`n_replications`**:
  - Low values (1): Fast, use for most single-chain proteins
  - High values (2-6): Predict symmetric assemblies; GPU memory increases linearly; >6 rarely biologically relevant
- **`chain_linker`**:
  - Low values (10-15): For tightly packed or continuous chains
  - High values (50-100): For loosely associated or distant chains

**Common mistakes:**
1. **Exceeding 2,400 residue limit:** Always check `n_replications * sequence_length <= 2400`. For a 600-residue protein, max n_replications = 4.
2. **Interpreting low pLDDT as "bad" sequences:** Low pLDDT regions may be biologically relevant (e.g., flexible linkers, disordered regions). Context matters.
3. **Using ESMFold for heteromeric complexes:** ESMFold replicates the same sequence. For A-B dimers (different sequences), use Boltz2 or AlphaFold-Multimer.
4. **Ignoring per-residue pLDDT:** Average pLDDT can hide poorly-folded regions. Always check `per_residue_plddt` to identify problem areas.

**Edge cases to watch for:**
- **Very short sequences (<30 aa):** May have low pLDDT due to lack of structural constraints; this is often biologically realistic (e.g., peptides are flexible)
- **Highly repetitive sequences:** May produce extended or disordered structures with low confidence
- **Non-standard amino acids:** Replace with 'X' (unknown) or closest standard amino acid; ESMFold will predict but confidence may be lower
- **Large oligomers approaching 2,400 limit:** May run out of GPU memory; reduce n_replications or use smaller GPU-optimized builds

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
