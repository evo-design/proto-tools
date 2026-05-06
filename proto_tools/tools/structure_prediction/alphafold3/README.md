<a href="https://bio-pro.mintlify.app/tools/structure-prediction/alphafold3"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# AlphaFold3

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

> [!IMPORTANT]
> **AlphaFold3 model parameters are gated by DeepMind's Terms of Use.** They are NOT automatically downloaded by `setup.sh` and cannot be redistributed. To run this tool you must:
>
> 1. Request access via the form linked in DeepMind's repository: <https://github.com/google-deepmind/alphafold3#obtaining-model-parameters>
> 2. After DeepMind approves (typically 2–3 business days), download the weights archive from the link they email you.
> 3. Place the weights file (e.g. `af3.bin.zst`) at `$PROTO_HOME/proto_model_cache/alphafold3/`, or override with `export PROTO_ALPHAFOLD3_WEIGHTS_DIR=/path/to/weights`.
>
> `setup.sh` fails fast with a pointer to this flow if the weights directory is empty. See the "Weights setup" section below for full details.

## Overview

AlphaFold3 is the latest generation structure prediction model from Google DeepMind that predicts 3D structures of biomolecular complexes including proteins, DNA, RNA, ligands, ions, and chemical modifications. It uses a [diffusion](https://en.wikipedia.org/wiki/Diffusion_model)-based architecture to generate joint 3D structures revealing how molecules fit together.

**Important**: For greater accuracy, use SMILES [defined here](https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi) when possible to allow for automatic conversion to CCD inputs, which are greatly preferred by AlphaFold3.

## Background

**What does this tool measure/predict?**
AlphaFold3 predicts the 3D atomic coordinates of biomolecular complexes from sequences. It outputs full-atom structures with comprehensive confidence scores including per-atom pLDDT, predicted aligned error (PAE), pTM, ipTM, and a composite ranking score. Unlike its predecessors, AlphaFold3 generates predictions for proteins alongside DNA, RNA, small molecule ligands, ions, and post-translational modifications.

**Why is this important?**
Understanding how biomolecules interact is fundamental to biology and drug discovery:
- Drug design: Predicting how small molecule drugs bind to protein targets
- Gene regulation: Modeling protein-DNA interactions that control gene expression
- RNA biology: Understanding protein-RNA complexes involved in cellular processes
- Enzyme mechanisms: Visualizing cofactor and ion binding sites
- Glycobiology: Modeling glycan modifications on proteins
- Antibody engineering: Predicting antibody-antigen binding interfaces

**Scientific foundation:**
AlphaFold3 uses a next-generation architecture combining:
1. **Pairformer module**: An improved version of the Evoformer architecture from AlphaFold2, using triangular attention to process sequence and structural features across all molecule types.
2. **Diffusion-based structure generation**: Starting from random atomic coordinates, a diffusion network iteratively refines positions to generate physically realistic 3D structures;similar to AI image generators but for molecular structures.
3. **[Multiple sequence alignments](https://en.wikipedia.org/wiki/Multiple_sequence_alignment) (MSAs)**: Evolutionary information from homologous sequences improves prediction accuracy for proteins and RNA.

Confidence metrics include:
- **pLDDT** (predicted Local Distance Difference Test): Per-atom confidence score (0-100), where >90 indicates high confidence, 70-90 is moderate, and <50 suggests the region is probably wrong.
- **pTM** (predicted Template Modeling score): Overall structure accuracy (0-1), where >0.5 indicates the global fold might be correct.
- **ipTM** (interface pTM): Confidence in inter-chain interfaces (0-1), where >0.8 indicates high-confidence interactions and <0.6 suggests likely failed prediction.
- **PAE** (Predicted Aligned Error): Estimates error in relative positions between residue pairs; lower values indicate higher confidence.
- **ranking_score**: Composite metric combining ipTM, pTM, disorder penalty, and clash penalty for ranking multiple predictions.

## Tools

### AlphaFold3 Structure Prediction (`alphafold3-prediction`)

Predict protein 3D structures using AlphaFold3.

## Execution Modes

AlphaFold3 requires GPU with >=40GB VRAM (A100-40GB minimum; H100/A100-80GB recommended). DeepMind-licensed model weights (~1GB) are required regardless of install path.

`setup.sh` supports two install paths, chosen by autodetect on first run:

1. **Sif path (preferred when available)**: uses the bundled `standalone/Singularity.def` recipe to build a portable Apptainer `.sif` locally via user-namespace build (no root, no Docker daemon required). Output is a single ~4 GB file at `$VENV_PATH/alphafold3.sif` that caches across env rebuilds and can be `scp`'d to other machines. First build takes ~20-30 min. Chosen automatically when the host kernel supports unprivileged user namespaces **and** the user has `/etc/subuid` / `/etc/subgid` mappings (both needed for rootless `apt-get` inside the build rootfs). Managed HPC clusters commonly satisfy (1) but not (2); in that case autodetect picks the env path.
2. **Env path (fallback)**: builds AlphaFold3 from source into the tool's micromamba venv (CUDA toolkit + JAX 0.9.1 + HMMER + zlib/zstd + AF3 clone + `pip install`). Used when user namespaces aren't available, when apptainer can't be installed, or when the sif build fails. Works everywhere micromamba runs.

At runtime, `inference.py` prefers whichever path is provisioned: if `$VENV_PATH/alphafold3.sif` exists (or the config sets `sif_path`), it runs `apptainer run <sif> ...` and the sif's `%runscript` execs `run_alphafold.py`. Otherwise it runs AF3 directly from the in-env Python install.

Override knobs via env vars / config:
- `PROTO_ALPHAFOLD3_SIF_PATH` — point at an already-built sif (e.g. on shared lab storage). `setup.sh` symlinks it into the venv and skips the env build entirely (~1 min setup instead of ~30 min). The recommended pattern for HPC users sharing one lab-built sif.
- `PROTO_ALPHAFOLD3_WEIGHTS_DIR` — point at the directory containing your DeepMind-licensed weights file. Falls back to `$PROTO_MODEL_CACHE/alphafold3/` and then `$PROTO_HOME/proto_model_cache/alphafold3/`.
- `ALPHAFOLD3_BUILD_SIF={auto,1,0}` — `auto` (default) autodetects whether to attempt a local sif build; `1` forces the sif build (errors if subuid missing); `0` forces the env path. Ignored when `PROTO_ALPHAFOLD3_SIF_PATH` is set.
- `ALPHAFOLD3_VERSION` — target a different AF3 release tag (default `v3.0.2`).
- `ALPHAFOLD3_JAX_SPEC` — pin a different JAX/jaxlib combination for the env path.
- `AlphaFold3Config.sif_path` — per-call BYO-sif override at the Python API layer.
- `AlphaFold3Config.model_dir` — per-call weights directory override at the Python API layer.

**Lab "build once, share many" workflow** (recommended for HPC):
```bash
# On a machine with subuid mappings (e.g. a personal Linux workstation):
ALPHAFOLD3_BUILD_SIF=1 ./setup.sh   # builds alphafold3.sif locally
cp $VENV_PATH/alphafold3.sif /shared/lab/storage/alphafold3.sif

# On the cluster (no subuid required, just user namespaces):
export PROTO_ALPHAFOLD3_SIF_PATH=/shared/lab/storage/alphafold3.sif
# `setup.sh` then symlinks the sif into the venv, installs apptainer,
# and skips the env install entirely.
```

**Two-stage workflow**: MSA generation (when enabled) is delegated to the `colabfold-search` tool and runs separately from inference (GPU-intensive). Runtime ranges from ~30 minutes to several hours depending on complex size and MSA generation settings.

**Model weights**: AlphaFold3 weights are gated by DeepMind's [Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) and cannot be auto-downloaded. Before running this tool:

1. Follow DeepMind's instructions to request weight access (link to the current request form is maintained in their README): <https://github.com/google-deepmind/alphafold3#obtaining-model-parameters>
2. After DeepMind approves (typically 2-3 business days), download the weights archive from the link they email you.
3. Place the weights file (e.g. `af3.bin.zst`) in the proto-tools weights cache:
   - Default location: `$PROTO_HOME/proto_model_cache/alphafold3/`
   - Or override per-tool: `export PROTO_ALPHAFOLD3_WEIGHTS_DIR=/path/to/weights`

`setup.sh` fails fast with a pointer to the request form if no `.bin` / `.bin.zst` file is found. See `notes/storage.md` for the full precedence rules for `PROTO_HOME`, `PROTO_MODEL_CACHE`, and per-tool overrides.

## How It Works

**Method overview:**
AlphaFold3 employs a multi-stage architecture:
1. **MSA generation (optional):** When `use_msa=True`, protein sequences are processed using ColabFold search to retrieve multiple sequence alignments, providing evolutionary information. MSA search can use remote ColabFold API or local databases.
2. **Input encoding**: Protein, RNA, and DNA sequences are encoded with MSA features when available. Ligands are specified using SMILES strings or CCD codes. All molecules receive positional and chemical encodings.
3. **Pairformer processing**: The encoded features pass through a transformer-based Pairformer module with triangular attention, enabling the model to reason about pairwise relationships across the entire molecular system.
4. **Diffusion-based structure prediction**: Starting from a cloud of random atomic coordinates, a diffusion model iteratively denoises positions over multiple timesteps to generate physically realistic 3D structures. By default, 5 samples are generated per seed, and the best is selected by ranking score.

**Key assumptions:**
- The complex forms a stable, defined 3D structure
- Input sequences and molecule types are correctly specified
- For proteins: MSA provides useful evolutionary signal
- For ligands: SMILES correctly represents the intended molecule. **Importantly**, AlphaFold3 has much greater accuracy when providing ligands as CCD codes over SMILES strings. This implementation converts input SMILES to CCD automatically using the SMILES representations found here: https://files.wwpdb.org/pub/pdb/data/monomers/Components-smiles-stereo-oe.smi.

**Limitations:**
- **Chain limit**: 26 chains (A-Z) in the current PDB-based implementation
- **Single conformation**: Predicts one structure per sample, not conformational ensembles
- **No explicit solvent**: Does not model water molecules or membrane environment
- **Ligand accuracy varies**: Performance may drop for ligands dissimilar to training data
- **Disordered regions**: May produce spurious alpha helices in disordered regions with low confidence

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seeds` | `List[int]` | `[0]` | Random seeds for structure prediction; AlphaFold3 generates 5 diffusion samples per seed |
| `use_msa` | `bool` | `True` | Whether to generate and use MSAs for protein chains |
| `colabfold_search_config.search_mode` | `str` | `"remote"` | MSA search mode; `"remote"` uses ColabFold API, `"local"` uses local databases |
| `colabfold_search_config.use_metagenomic` | `bool` | `False` | Include metagenomic sequences in MSA search for improved coverage |
| `modifications_map` | `Dict` | `None` | Specify chemical modifications to residues (phosphorylation, methylation, etc.) |

## Configuration

### Parameter Guides

| Parameter | Sweep Range | Notes |
|-----------|-------------|-------|
| `seeds` | `[0], [0,1,2], [0,1,2,3,4]` | More seeds = more samples to choose from; linear increase in runtime |
| `use_msa` | `True, False` | `False` skips MSA generation entirely (faster but lower accuracy) |
| `colabfold_search_config.search_mode` | `"remote", "local"` | `"remote"` uses ColabFold API (requires internet); `"local"` requires local databases |
| `colabfold_search_config.use_metagenomic` | `True, False` | Can improve accuracy for sparse MSAs but adds significant runtime |

### Sweep Priorities

1. **`seeds`**: Most impactful for exploring conformational diversity. Use multiple seeds (e.g., `[0,1,2,3,4]`) for complex docking tasks like antibody-antigen interactions. Single seed often sufficient for simpler predictions.
2. **`colabfold_search_config.use_metagenomic`**: Can improve accuracy for proteins with sparse MSAs but adds significant runtime. Try with and without for important targets.
3. **`use_msa`**: Set to `False` to skip MSA generation entirely (faster but lower accuracy for most proteins).

## Output Specification

```python
# Return type: AlphaFold3Output
AlphaFold3Output(
    structures: List[Structure],  # One per input complex
    metadata: dict,
)

# Each Structure contains metrics from AlphaFold3:
metrics = {
    "ranking_score": float,     # Composite ranking score (-100 to 1.5)
    "ptm": float,               # Predicted TM-score (0-1)
    "iptm": float,              # Interface pTM (0-1)
    "fraction_disordered": float,  # Fraction of disordered residues
    "has_clash": float,         # Clash indicator (0 or 1)
    "chain_ptm": List[float],   # Per-chain pTM scores
    "chain_iptm": List[float],  # Per-chain interface scores
    "chain_pair_iptm": List[List[float]],  # Pairwise chain interface scores
}
```

**Key output fields:**

| Field | Type | Range | Interpretation |
|-------|------|-------|----------------|
| `ranking_score` | `float` | `-100 - 1.5` | Composite score for ranking; higher is better; formula: `0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash` |
| `ptm` | `float` | `0.0 - 1.0` | Global fold confidence; >0.5 suggests correct topology |
| `iptm` | `float` | `0.0 - 1.0` | Interface confidence; >0.8 = high confidence, <0.6 = likely failed |
| `plddt` | `float` | `0 - 100` | Per-atom confidence; >90 = excellent, 70-90 = good, <50 = unreliable |
| `fraction_disordered` | `float` | `0.0 - 1.0` | Proportion of low-confidence regions |
| `chain_pair_iptm` | `array` | `0.0 - 1.0` | Interface quality between specific chain pairs |

## Interpreting Results

**Thresholds & decision boundaries:**
- **Excellent:** `iptm > 0.8` and `plddt > 90`: High confidence; structure suitable for detailed analysis and drug design
- **Acceptable:** `0.6 < iptm <= 0.8` and `70 < plddt <= 90`: Moderate confidence; verify key interactions manually
- **Poor:** `iptm <= 0.6` or `plddt <= 50`: Low confidence; interaction may be incorrect; consider redesigning or using alternative methods

**Tips for interpreting output:**
- For multi-chain predictions, `iptm` is more informative than `ptm` about interface quality
- Use `chain_pair_iptm` to identify which specific interfaces are well-predicted vs uncertain
- Visualize structures colored by pLDDT (stored in B-factor column) to identify uncertain regions
- Check PAE plots for inter-domain/inter-chain relationships: low PAE between domains indicates confident relative positioning
- For ligands, examine `chain_iptm` for the ligand chain specifically to assess binding confidence
- Very small proteins (<20 residues): pTM becomes unreliable due to TM-score definition; rely on pLDDT and PAE instead

## Quick Start Examples

```python
from proto_tools.tools.structure_prediction.alphafold3 import (
    run_alphafold3_prediction,
    AlphaFold3PredictionInput,
    AlphaFold3PredictionConfig,
)
from proto_tools.tools.structure_prediction.shared_data_models import (
    StructurePredictionComplex,
    ProteinChain,
    LigandChain,
)

# Basic protein-ligand complex
inputs = AlphaFold3PredictionInput(
    complexes=[
        StructurePredictionComplex(
            protein_chains=[ProteinChain(sequence="MKTVRQ...")],
            ligand_chains=[LigandChain(sequence="CC(=O)Oc1ccccc1C(O)=O")],  # Aspirin SMILES
        )
    ]
)
config = AlphaFold3PredictionConfig(seeds=[0, 1, 2])
result = run_alphafold3_prediction(inputs, config)

# Check results
for structure in result.structures:
    print(f"Ranking score: {structure.metrics['ranking_score']:.3f}")
    print(f"ipTM: {structure.metrics['iptm']:.3f}")
    print(f"pTM: {structure.metrics['ptm']:.3f}")
```

## Best Practices & Gotchas

**Parameter tuning:**
- **`seeds`**:
  - Single seed (default): Often sufficient for well-defined complexes
  - Multiple seeds (3-5): Recommended for protein-ligand docking, antibody-antigen, and flexible targets
  - More seeds = more samples to choose from, but linear increase in runtime
- **`use_msa`**:
  - On (default): Best performance for most proteins; MSAs provide crucial evolutionary signals
  - Off: Fastest but generally lower accuracy; use only for speed-critical screening
- **`colabfold_search_config.use_metagenomic`**:
  - Off (default): Faster, usually adequate for well-characterized proteins
  - On: Use for orphan proteins, viral sequences, or when initial predictions have low confidence
- **`colabfold_search_config.search_mode`**:
  - `"remote"` (default): Uses ColabFold API; convenient but requires internet
  - `"local"`: Best performance if databases are available locally; no network dependency

**Common mistakes:**
1. **Wrong SMILES format**: Ensure ligand SMILES are valid and represent the correct protonation state. SMILE strings will be validated when passed to input classes.
2. **Ignoring ipTM for complexes**: For multi-chain predictions, `iptm` is more informative than `ptm` about interface quality.
3. **Not checking per-chain metrics**: Use `chain_pair_iptm` to identify which specific interfaces are well-predicted vs uncertain.
4. **Single seed for difficult docking**: Protein-ligand and antibody-antigen predictions benefit significantly from multiple seeds.
5. **Expecting dynamics**: AlphaFold3 predicts single static structures; for conformational diversity, run with multiple seeds and compare.

**Edge cases to watch for:**
- **Very small proteins (<20 residues)**: pTM becomes unreliable due to TM-score definition; rely on pLDDT and PAE instead
- **Highly disordered proteins**: Expect low pTM/ipTM even for correct predictions; check structured domain regions separately
- **Multiple ligand copies**: Ensure each ligand gets a unique chain ID
- **Metal ions**: Specified as ligands using CCD codes (e.g., "ZN" for zinc, "MG" for magnesium)
- **Glycans**: Requires special CCD-based specification; not all glycan types supported
- **DNA/RNA**: Does not support MSA for DNA; RNA can use MSA but coverage may be limited

## References

**Primary publication:**
- Abramson et al. (2024). "Accurate structure prediction of biomolecular interactions with AlphaFold 3". *Nature*. [DOI: 10.1038/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w)
- Summary: Introduces AlphaFold3 as a unified framework for predicting structures of proteins, nucleic acids, small molecules, ions, and their interactions with unprecedented accuracy.

**Implementation:**
- GitHub: [https://github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)
- Input documentation: [https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)
- Output documentation: [https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md)

## Related Tools

**Tools often used together:**
- **`esmfold`**: Quick protein-only structure validation before running expensive AlphaFold3 predictions
- **`esm2-embedding`**: Analyze sequence features before structure prediction

**Alternative tools:**
- **`chai1`**: Supports protein-ligand-glycan complexes with open-source, commercially-friendly license
- **`boltz2-prediction`**: MIT-licensed alternative supporting similar molecule types
- **`esmfold`**: 10-60x faster for protein-only predictions; use when no ligands/DNA/RNA needed
- **`alphafold2`**: Protein-only predecessor; may be preferred for pure protein complexes with rich MSA data
