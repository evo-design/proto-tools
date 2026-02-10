# Protenix

## Overview
Protenix is an open-source biomolecular structure prediction model developed by ByteDance that predicts 3D structures of proteins, nucleic acids (DNA/RNA), and ligands using a diffusion-based architecture. Protenix is the first fully open-source model that outperforms AlphaFold3 across diverse benchmarks, providing state-of-the-art accuracy with comprehensive molecular type support and MSA integration.

## Background

**What does this tool measure/predict?**
Protenix predicts the 3D atomic coordinates of biomolecular complexes from sequences. It outputs full-atom structures for proteins, nucleic acids (DNA/RNA), and ligands with support for post-translational modifications, ions, and other chemical modifications. Confidence scores include pTM, ipTM, pLDDT, and global predicted distance error (GPDE) metrics.

Confidence metrics include:
- **pTM** (predicted TM-score): Global structure accuracy (0-1), primary metric for single chains.
- **ipTM** (interface pTM): Confidence in inter-chain interfaces (0-1), primary metric for complexes.
- **pLDDT** (predicted LDDT): Per-residue confidence scores (0-100).
- **GPDE** (Global Predicted Distance Error): Spatial error estimates in Angstroms.

## Important Parameters (for param sweeps)

**Input parameters:**

| Parameter | Type | Default | Sweep Range | Description |
|-----------|------|---------|-------------|-------------|
| `complexes` | `List[StructurePredictionComplex]` | *required* | N/A | List of complexes; each can have protein, DNA, RNA, and/or ligand chains with optional modifications |
| `use_msa` | `bool` | `True` | `True, False` | Whether to generate and use MSAs for protein chains |
| `num_pairformer_cycles` | `int` | `10` | `5 - 20` | Pairformer recycling iterations; more = better quality but slower |
| `num_diffusion_steps` | `int` | `200` | `100 - 500` | Diffusion denoising steps; more = finer structures but slower |
| `num_diffusion_samples` | `int` | `5` | `1 - 10` | Independent structure samples; best is returned by confidence |
| `seeds` | `List[int]` | `[0]` | N/A | Random seeds for sampling; each seed produces num_diffusion_samples |
| `model_name` | `str` | `"protenix_base_default_v1.0.0"` | N/A | Model checkpoint to use |

**Parameters to prioritize for sweeps:**
1. **`num_diffusion_samples`**: Most impactful for conformational diversity and finding optimal binding poses. Use 1-3 for screening, 5-10 for high-confidence predictions.
2. **`num_pairformer_cycles`**: Affects structure refinement quality. Try 5, 10, 20 to balance speed vs accuracy.
3. **`num_diffusion_steps`**: For critical predictions, increase to 300-500. Use 100-200 for rapid screening.
4. **`use_msa`**: Critical for accuracy but adds runtime. Try `False` for orphan proteins or ultra-fast screening.

---

**Output specification:**

```python
# Return type: ProtenixOutput
ProtenixOutput(
    structures: List[Structure],  # One per input complex
)

# Each Structure contains comprehensive metrics:
metrics = {
    # Primary metrics (always present)
    "confidence_score": float,   # Primary ranking score (weighted combination of metrics)
    "ptm": float,                # Predicted TM-score (0-1)
    "iptm": float,               # Interface pTM (0-1), None for single chains

    # Per-residue and per-chain metrics
    "avg_plddt": float,          # Average pLDDT across all residues (0-1, normalized from 0-100)
    "chain_ptm": List[float],    # Per-chain PTM scores
    "chain_plddt": List[float],  # Per-chain average pLDDT scores

    # Spatial error metrics
    "gpde": float,               # Global Predicted Distance Error (Å)
}
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `confidence_score` | `float` | `0.0 - 1.0` | Primary ranking score; >0.8 = excellent, >0.7 = good, <0.5 = poor |
| `ptm` | `float` | `0.0 - 1.0` | Global fold quality; >0.8 = high confidence, <0.5 = unreliable |
| `iptm` | `float` | `0.0 - 1.0` | Interface quality; >0.8 = confident, 0.6-0.8 = moderate, <0.6 = uncertain |
| `avg_plddt` | `float` | `0.0 - 1.0` | Per-residue average (normalized); >0.9 = very high, 0.7-0.9 = good, <0.7 = uncertain |
| `gpde` | `float` | `>0 Å` | Global spatial error; <10Å = good, 10-20Å = moderate, >20Å = poor |

**Thresholds & decision boundaries:**
- **Excellent:** `confidence_score > 0.8` and `avg_plddt > 0.8` — High confidence; structure suitable for detailed analysis
- **Acceptable:** `0.6 < confidence_score ≤ 0.8` — Moderate confidence; verify key interactions manually
- **Poor:** `confidence_score ≤ 0.6` — Low confidence; consider more samples or alternative approaches

## Best Practices & Gotchas

**Parameter tuning:**
- **`num_pairformer_cycles`**:
  - Low values (5): Faster but less refined; use for initial screening
  - Medium values (10): Good balance for production use
  - High values (15-20): Maximum refinement; use for publication-quality structures
- **`num_diffusion_steps`**:
  - Low values (100-150): Fast; acceptable for screening
  - Medium values (200): Default; good balance
  - High values (300-500): Maximum quality; use for critical predictions
- **`num_diffusion_samples`**:
  - Low values (1-3): Faster; may miss optimal conformations
  - Medium values (5): Good default; balances speed and diversity
  - High values (7-10): Maximum exploration; recommended for flexible complexes

## References
**Protenix-specific:**
- ByteDance Protenix Team. "Protenix: Toward High-Accuracy Open-Source Biomolecular Structure Prediction" [Technical Report](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf)

**Implementation:**
- GitHub: [https://github.com/bytedance/Protenix](https://github.com/bytedance/Protenix)
- Performance Benchmarks: [https://github.com/bytedance/Protenix/blob/main/docs/model_1.0.0_benchmark.md](https://github.com/bytedance/Protenix/blob/main/docs/model_1.0.0_benchmark.md)
- Model weights: Automatically downloaded from Hugging Face on first run
- License: Apache 2.0

## Implementation Notes

**CUDA Isolation:**
> [!WARNING]
> Protenix has a fragile CUDA environment setup that may not work on all systems. See the troubleshooting section for more details.

Protenix uses a fully isolated CUDA environment managed via micromamba. The setup process:
1. Installs CUDA toolkit (12.1) locally within the venv at `$VENV_PATH/cuda_env`
2. Creates symlinks for CUDA C++ Standard Library headers (cuda/std, thrust, cub)
3. Fixes version-specific library symlinks (libcudart.so) to match installed versions
4. Generates `sitecustomize.py` to set CUDA environment variables at Python startup

This ensures Protenix does not depend on system CUDA installations and works in containerized/isolated environments.

**Troubleshooting:**
- **CUDA compilation errors:** Check `$VENV_PATH/cuda_env` exists and contains CUDA toolkit
- **Missing headers:** Verify symlinks in `$VENV_PATH/cuda_env/include/` for cuda, thrust, cub
- **Linker errors:** Check `libcudart.so` symlink is not broken in `$VENV_PATH/cuda_env/lib/`
- **Runtime errors:** Ensure `sitecustomize.py` exists in venv site-packages and sets CUDA_HOME
