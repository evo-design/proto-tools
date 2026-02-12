# Boltz2

## Overview

Boltz2 is a state-of-the-art multi-modal structure prediction model that predicts 3D structures of proteins, DNA, RNA, and ligands using a diffusion-based architecture with MSA integration. It provides the broadest molecular type support among current structure prediction tools.

## When to Use This Tool

**Primary use cases:**
- **Protein-DNA complexes:** Model transcription factors bound to DNA regulatory elements
- **Protein-RNA interactions:** Predict RNA-binding protein structures with bound RNA
- **Multi-molecular complexes:** Model systems with proteins, nucleic acids, AND ligands together
- **Nucleic acid structures:** Predict DNA/RNA secondary and tertiary structures
- **Heteromeric protein complexes:** Model protein-protein interactions with different chains (A-B dimers)
- **Drug-nucleic acid interactions:** Screen small molecules targeting RNA structures

**When NOT to use this tool:**
- **High-throughput protein-only screening:** ESMFold is 10-100x faster for pure protein predictions
- **Glycoprotein modeling:** Use Chai1 for glycan-containing structures (Boltz2 doesn't support glycans)
- **Very rapid predictions:** Boltz2's thorough sampling (25 samples default) takes longer than single-shot methods
- **Systems without any nucleic acids or ligands:** ESMFold or Chai1 may be faster for protein-only
- **Real-time applications:** Runtime is too long for interactive use cases

**Comparison with alternatives:**
- **Boltz2 vs Chai1:** Boltz2 supports DNA/RNA; Chai1 supports glycans. Choose based on molecular types needed.
- **Boltz2 vs ESMFold:** ESMFold is much faster but protein-only with no MSA. Use Boltz2 for complexes or when MSA improves accuracy.
- **Boltz2 vs AlphaFold-Multimer:** Similar accuracy; Boltz2 adds ligand and nucleic acid support.
- **Boltz2 vs RoseTTAFold2NA:** Both handle protein-nucleic acid; Boltz2 is newer with diffusion architecture.

## Biological Background

**What does this tool measure/predict?**
Boltz2 predicts the 3D atomic coordinates of biomolecular complexes involving proteins, nucleic acids (DNA/RNA), and small molecule ligands. It outputs full-atom structures with comprehensive confidence metrics including pTM, ipTM, pLDDT, and specialized interface scores for different molecular type combinations.

**Why is this important?**
The ability to model protein-nucleic acid and multi-molecular complexes is essential for:
- Understanding gene regulation (transcription factor-DNA complexes)
- Modeling RNA-protein interactions (ribonucleoproteins, RNA-binding proteins)
- Drug design targeting nucleic acids (RNA therapeutics, antisense oligonucleotides)
- CRISPR and gene editing tool development (Cas protein-guide RNA-DNA complexes)
- Studying ribosomes, spliceosomes, and other RNA-protein machines

**Scientific foundation:**
Boltz2 combines several modern deep learning advances:
1. **MSA-based evolutionary features**: Multiple sequence alignments provide critical evolutionary co-variation signals that indicate residue contacts and structural constraints.
2. **Diffusion-based generative modeling**: Starting from noise, the model iteratively refines coordinates through learned denoising steps, naturally handling the flexibility of biomolecular complexes.
3. **Multi-modal architecture**: Specialized encoders for proteins (amino acid sequences), DNA/RNA (nucleotide sequences), and ligands (SMILES) enable unified prediction of heterogeneous complexes.
4. **Extensive sampling**: By default generates 25 independent structure samples and returns the best by confidence, exploring conformational diversity.

Confidence metrics include:
- **pTM** (predicted TM-score): Global structure accuracy (0-1), primary metric for single chains.
- **ipTM** (interface pTM): Confidence in inter-chain interfaces (0-1), primary metric for complexes.
- **pLDDT** (predicted LDDT): Per-residue confidence scores.
- **Specialized interface scores**: `ligand_iptm`, `protein_iptm` for specific interaction types.

## Execution Modes

Boltz2 requires GPU with >=40GB VRAM (H100 or A100-80GB recommended).

- **Local execution**: Runs on local GPU. Runtime ~2-10 minutes per complex on H100/A100. Requires internet for remote MSA search mode.

## How It Works

**Method overview:**
Boltz2 uses a sophisticated pipeline:
1. **MSA generation (optional):** When `use_msa=True`, protein sequences are processed using ColabFold search to retrieve multiple sequence alignments, providing evolutionary context and co-variation signals. MSA search can use remote ColabFold API or local databases.
2. **Feature encoding:** Proteins, DNA, RNA, and ligands are encoded with type-specific featurizers. MSA features are integrated via attention mechanisms when available.
3. **Iterative recycling:** The encoded features pass through the model architecture multiple times (recycling_steps), progressively refining the structural representation.
4. **Diffusion sampling:** A diffusion process generates 3D coordinates by iteratively denoising over many steps (sampling_steps). Multiple independent samples (diffusion_samples) explore different conformations.
5. **Ranking and selection:** All samples are scored by confidence metrics, and the best structure is returned.

**Key assumptions:**
- The complex forms a stable, well-defined 3D structure
- MSA signals are informative (works best for proteins with homologs)
- Entity types are correctly specified or inferred
- Nucleic acid sequences use standard bases (A, T/U, C, G)
- Ligands are specified as valid SMILES strings

**Limitations:**
- **No glycans:** Cannot model carbohydrate modifications (use Chai1)
- **Runtime:** Thorough sampling (25 samples x 200 steps) takes 2-10 minutes per complex
- **Network dependency:** Requires internet access when using remote MSA search mode
- **Memory intensive:** Large complexes require significant GPU memory

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | *required* | List of complexes; each can have protein, DNA, RNA, and/or ligand chains |
| `use_msa` | `bool` | `True` | Whether to generate and use MSAs for protein chains using ColabFold search |
| `colabfold_search_config.search_mode` | `str` | `"remote"` | MSA search mode; `"remote"` uses ColabFold API, `"local"` uses local databases |
| `colabfold_search_config.use_metagenomic` | `bool` | `False` | Include metagenomic sequences in MSA search for improved coverage |
| `recycling_steps` | `int` | `10` | Iterative refinement passes; more = better quality but slower |
| `sampling_steps` | `int` | `200` | Denoising steps; more = finer structures but slower |
| `diffusion_samples` | `int` | `25` | Independent structure samples; best is returned by confidence |
| `num_workers` | `int` | `min(cpu_count, 4)` | CPU workers for parallel processing |
| `devices` | `str` | `"0"` | GPU device IDs (e.g., "0" or "0,1" for multi-GPU) |

## Configuration

### Parameter Guides

| Parameter | Sweep Range | Notes |
|-----------|-------------|-------|
| `recycling_steps` | `3 - 20` | Higher = more refined but slower |
| `sampling_steps` | `100 - 500` | Higher = finer structures; 200 is good default |
| `diffusion_samples` | `5 - 50` | More samples explore conformational diversity |
| `use_msa` | `True, False` | Disabling reduces accuracy significantly |

### Sweep Priorities

1. **`diffusion_samples`**: Most impactful for conformational diversity and finding optimal binding poses. Use 5-10 for screening, 25-50 for high-confidence predictions.
2. **`recycling_steps`**: Affects structure refinement quality. Try 5, 10, 15 to balance speed vs accuracy.
3. **`sampling_steps`**: For critical predictions, increase to 300-500. Use 100-150 for rapid screening.

## Output Specification

```python
# Return type: Boltz2Output
Boltz2Output(
    structures: List[Structure],  # One per input complex
)

# Each Structure contains comprehensive metrics:
metrics = {
    # Primary metrics (always present)
    "confidence_score": float,   # Primary ranking score (iptm for complexes, ptm for single chains)
    "ptm": float,                # Predicted TM-score (0-1)
    "iptm": float,               # Interface pTM (0-1)

    # Per-chain metrics
    "chains_ptm": List[float],   # pTM for each chain individually
    "pair_chains_iptm": List[List[float]],  # Pairwise ipTM matrix between chains

    # Optional metrics (may be None)
    "ligand_iptm": float,        # Protein-ligand interface score
    "protein_iptm": float,       # Protein-protein interface score
    "complex_plddt": float,      # Average pLDDT across all residues (0-100)
    "complex_iplddt": float,     # Interface pLDDT (0-100)
    "complex_pde": float,        # Predicted aligned error (Angstrom), lower is better
    "complex_ipde": float,       # Interface PAE (Angstrom), lower is better
}
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `confidence_score` | `float` | `0.0 - 1.0` | Primary ranking score; >0.7 = good, >0.85 = excellent |
| `ptm` | `float` | `0.0 - 1.0` | Global fold quality; >0.7 = high quality, <0.5 = unreliable |
| `iptm` | `float` | `0.0 - 1.0` | Interface quality; >0.85 = high confidence, 0.7-0.85 = moderate, <0.7 = low |
| `ligand_iptm` | `float` | `0.0 - 1.0` | Protein-ligand binding confidence; >0.7 = reliable binding pose |
| `complex_plddt` | `float` | `0 - 100` | Per-residue average; >90 = very high, 70-90 = high, <70 = uncertain |
| `complex_pde` | `float` | `>0 Angstrom` | Spatial error estimate; <10A = good relative positions, >20A = uncertain |

## Interpreting Results

**Thresholds & decision boundaries:**
- **Excellent:** `confidence_score > 0.85` and `complex_plddt > 80` — High confidence; structure suitable for detailed analysis
- **Acceptable:** `0.7 < confidence_score <= 0.85` — Moderate confidence; verify key interactions
- **Poor:** `confidence_score <= 0.7` — Low confidence; consider more samples or alternative approaches

**Tips for interpreting output:**
- Use `ligand_iptm` for drug binding assessment, `protein_iptm` for protein-protein interfaces — not just the overall `confidence_score`
- Check `pair_chains_iptm` to identify which chain pairs have confident vs uncertain interfaces
- Compare `complex_plddt` vs `complex_iplddt` — if interface pLDDT is much lower, the binding mode may be uncertain
- Visualize PAE matrices to understand confidence in relative domain/chain positions
- `complex_pde < 10A` indicates good relative positioning between chains

## Quick Start Examples

```python
from bio_programming_tools.tools.structure_prediction.boltz2 import (
    run_boltz2_prediction,
    Boltz2PredictionInput,
    Boltz2PredictionConfig,
)
from bio_programming_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    ProteinChain,
    DNAChain,
)

# Protein-DNA complex prediction
inputs = Boltz2PredictionInput(
    complexes=[
        StructurePredictionComplex(
            protein_chains=[ProteinChain(sequence="MKTVRQ...")],
            dna_chains=[DNAChain(sequence="ATCGATCGATCG")],
        )
    ]
)
config = Boltz2PredictionConfig(
    recycling_steps=10,
    diffusion_samples=25,
)
result = run_boltz2_prediction(inputs, config)

# Check results
for structure in result.structures:
    print(f"Confidence: {structure.metrics['confidence_score']:.3f}")
    print(f"ipTM: {structure.metrics['iptm']:.3f}")
    print(f"pLDDT: {structure.metrics.get('complex_plddt', 'N/A')}")
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`recycling_steps`**:
  - Low values (3-5): Faster but less refined; use for initial screening
  - High values (10-20): More refined structures; use for final predictions
- **`sampling_steps`**:
  - Low values (100-150): Fast; good for screening when combined with high diffusion_samples
  - High values (300-500): More accurate; use for publication-quality structures
- **`diffusion_samples`**:
  - Low values (5-10): Faster; may miss optimal conformations
  - High values (25-50): Better exploration; recommended for protein-ligand or flexible complexes

**Common mistakes:**
1. **Disabling MSA:** `use_msa=False` reduces prediction accuracy significantly. MSAs provide critical evolutionary signals.
3. **Including glycans:** Boltz2 doesn't support glycans. Use Chai1 for glycoprotein structures.
4. **Insufficient samples for flexible systems:** Protein-nucleic acid complexes often need 25+ samples to find good conformations.
5. **Ignoring specialized metrics:** Use `ligand_iptm` for drug binding, `protein_iptm` for protein-protein, not just overall `confidence_score`.
6. **Network timeouts:** Large MSA queries can timeout when using remote mode. Retry with smaller sequences or check network connectivity.
7. **Wrong sequence format:** DNA should use A/T/C/G (not U), RNA should use A/U/C/G. Ligands must be SMILES strings.

**Edge cases to watch for:**
- **Very long DNA/RNA (>500 nt):** May require significant memory; consider trimming to binding region
- **Unusual base modifications:** Non-standard nucleotides may not be recognized; use standard bases
- **Highly flexible linkers:** Regions between structured domains may show low pLDDT (biologically realistic)
- **Symmetric complexes:** May predict asymmetric arrangements; verify symmetry if biologically expected
- **Orphan proteins:** Proteins without homologs (low MSA depth) may have reduced accuracy

## References

**Primary publication:**
- Wohlwend et al. (2025). "Boltz-2: Towards accurate and efficient biomolecular structure prediction". *bioRxiv*. [DOI: 10.1101/2025.06.14.659707](https://www.biorxiv.org/content/10.1101/2025.06.14.659707)
- Summary: Introduces Boltz2 as an open-source model achieving state-of-the-art accuracy on protein, nucleic acid, and ligand structure prediction benchmarks with efficient diffusion-based generation.

**Implementation:**
- GitHub: [https://github.com/jwohlwend/boltz](https://github.com/jwohlwend/boltz)
- Website: [https://boltz.bio/boltz2](https://boltz.bio/boltz2)
- Documentation: [https://github.com/jwohlwend/boltz/blob/main/README.md](https://github.com/jwohlwend/boltz/blob/main/README.md)

**Additional resources:**
- ColabFold MSA Server: [https://colabfold.com](https://colabfold.com) - MSA generation backend
- Model weights: Automatically downloaded from Hugging Face on first run

## Related Tools

**Tools often used together:**
- **`esmfold`**: Quick protein-only structure validation before expensive multi-modal predictions
- **`esm2-embedding`**: Analyze sequence features; Boltz2 uses related embedding approaches internally

**Alternative tools:**
- **`chai1-prediction`**: Better for glycan-containing complexes; Chai1 doesn't require MSA server
- **`esmfold`**: Much faster for protein-only structures when nucleic acids/ligands not needed
- **`alphafold2`**: Higher accuracy for pure protein structures; use for final validation
