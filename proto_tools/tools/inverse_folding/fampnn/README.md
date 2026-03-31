<a href="https://bio-pro.mintlify.app/tools/inverse-folding/fampnn"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# FAMPNN

## Overview
FAMPNN (Full-Atom MPNN) is a deep learning model for protein sequence design that jointly models discrete amino acid identity and continuous sidechain conformation. Unlike backbone-only inverse folding models, FAMPNN generates both sequences and full-atom sidechain coordinates simultaneously using combined cross-entropy and diffusion loss. This module provides interfaces for *Sequence Sampling*, *Sidechain Packing*, *Mutation Scoring*, and *Exhaustive Mutation Scanning*.

## When to Use This Tool

**Primary use cases:**
- Full-atom protein design: generating sequences with predicted sidechain coordinates
- Sidechain packing: predicting sidechain conformations for fixed backbone + sequence
- Mutation fitness prediction: scoring mutations with full-atom structural context
- Mutational landscape generation: exhaustive single-mutation scanning
- Protein stabilization through sequence optimization with sidechain awareness

**When NOT to use this tool:**
- No structure available: FAMPNN requires a 3D backbone structure. Use structure prediction first (ESMFold, AlphaFold, Boltz2).
- De novo backbone design: FAMPNN designs sequences, not structures. Use RFdiffusion or Chroma for backbone generation.
- Backbone-only design: If sidechains aren't needed, ProteinMPNN or LigandMPNN are faster.
- Ligand-aware design: FAMPNN doesn't model ligand interactions. Use LigandMPNN for ligand-conditioned design.

## Biological Background

**What does this tool do?**
FAMPNN solves the full-atom [inverse folding](https://en.wikipedia.org/wiki/Protein_design#Inverse_folding) problem: given a protein backbone structure, predict both the amino acid sequence and the [sidechain](https://en.wikipedia.org/wiki/Side_chain) conformations that are most compatible with the backbone. This goes beyond traditional inverse folding by also predicting atomic-level sidechain geometry.

**Why is this important?**
Full-atom design enables:
- More accurate protein engineering by modeling sidechain packing interactions
- Sidechain confidence scores (pSCE) for identifying uncertain regions
- Better mutation effect prediction using full structural context
- Direct generation of atomic models suitable for [molecular dynamics](https://en.wikipedia.org/wiki/Molecular_dynamics) or [docking](https://en.wikipedia.org/wiki/Molecular_docking)

**Scientific foundation:**
FAMPNN uses a two-component architecture:
1. **Iterative masked language modeling**: Sequence design via progressive unmasking (similar to MaskGIT), starting fully masked and revealing tokens iteratively.
2. **Per-token Euclidean [diffusion](https://en.wikipedia.org/wiki/Diffusion_model)**: Sidechain coordinates generated via variance-exploding EDM in local backbone reference frames.
3. **Predicted Sidechain Error (pSCE)**: A learned confidence metric predicting per-atom sidechain packing error in Angstroms.

## Tool Catalog

| Tool | Key | Input | Output | Use Case |
|------|-----|-------|--------|----------|
| Sampling | `fampnn-sample` | Structure(s) | Sequences + PDB strings + pSCE | Design sequences with sidechain co-generation |
| Packing | `fampnn-pack` | Structure(s) | Packed PDB strings + pSCE | Predict sidechain conformations for fixed sequences |
| Scoring | `fampnn-score` | Structure + mutations | Log-likelihood ratio scores | Score specific mutations |
| All Mutations | `fampnn-score-all-mutations` | Structure(s) | Position x residue score matrix | Exhaustive single-mutation scanning |

## Execution Modes

FAMPNN runs on GPU (required):
- **GPU**: ~2-4GB GPU memory for typical single-chain proteins (<500 residues).
- **Model weights**: Three checkpoints (~150MB each), downloaded automatically by `setup.sh`:
  - `fampnn_0_3.pt` -- Sequence design (PDB-trained, 0.3A noise)
  - `fampnn_0_0.pt` -- Sidechain packing (PDB-trained, 0.0A noise)
  - `fampnn_0_3_cath.pt` -- Mutation scoring ([CATH](https://www.cathdb.info/)-trained)

## How It Works

**Sampling (`fampnn-sample`):** Iteratively unmasks sequence positions while simultaneously denoising sidechain coordinates. Outputs full-atom PDB structures with per-residue pSCE confidence scores.

**Packing (`fampnn-pack`):** Given a fixed backbone and sequence, predicts sidechain coordinates using per-token Euclidean diffusion. B-factor column contains per-atom pSCE.

**Scoring (`fampnn-score`):** Evaluates mutation fitness by masking the mutated position and computing the conditional log-likelihood ratio of mutant vs. wild-type. Supports full-atom context or sequence-only (`seq_only=True`).

**Score All Mutations (`fampnn-score-all-mutations`):** For each position, masks that position and computes log-likelihood ratios for all 20 amino acids, generating a comprehensive mutational landscape.

## Input Parameters

### Sampling (`fampnn-sample`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `structure` | `Structure` | Protein structure (file path, PDB string, or Structure object) |
| `chain_ids` | `Optional[List[str]]` | Chains to design. If `None`, designs all chains |
| `fixed_positions` | `Optional[Dict[str, List[int]]]` | Residue positions to keep fixed (1-indexed) |
| `fixed_sidechain_positions` | `Optional[Dict[str, List[int]]]` | Positions with known sidechain coordinates to condition on (1-indexed) |

### Packing (`fampnn-pack`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `structure` | `Structure` | Protein structure with sequence to pack sidechains for |
| `fixed_sidechain_positions` | `Optional[Dict[str, List[int]]]` | Positions with known sidechain coordinates (1-indexed) |

### Scoring (`fampnn-score`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `structure` | `Structure` | Protein structure to evaluate mutations against |
| `mutations` | `List[str]` | Mutation strings: `'A0V'` (0-indexed), `'A0V:G5L'` for multi-site, `'wt'` for wild-type |

### Score All Mutations (`fampnn-score-all-mutations`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `List[Structure]` | Structures to score all possible single mutations |

## Configuration

### Sampling Configuration (`FAMPNNSampleConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_sequences_per_structure` | `int` | `1` | Total sequences to generate per structure |
| `batch_size` | `Optional[int]` | `None` | Max sequences per GPU forward pass |
| `temperature` | `float` | `0.1` | Sampling temperature (0.0-1.0) |
| `model_variant` | `str` | `"0.3"` | Model checkpoint variant |
| `num_steps` | `int` | `100` | Number of iterative unmasking steps |
| `seq_only` | `bool` | `False` | Skip sidechain generation |
| `repack_last` | `bool` | `True` | Repack sidechains after final sequence |
| `psce_threshold` | `float` | `0.3` | Sidechain error threshold during design |
| `seed` | `int` | `42` | Random seed |

### Packing Configuration (`FAMPNNPackConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_variant` | `str` | `"0.0"` | Model checkpoint (use `"0.0"` for packing) |
| `num_samples_per_structure` | `int` | `1` | Number of packing samples |
| `batch_size` | `int` | `1` | Samples per GPU forward pass |
| `scn_diffusion_steps` | `int` | `50` | Sidechain diffusion denoising steps |
| `seed` | `int` | `42` | Random seed |

### Scoring Configuration (`FAMPNNScoreConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_variant` | `str` | `"0.3_cath"` | Model checkpoint (use `"0.3_cath"` for scoring) |
| `batch_size` | `int` | `16` | Mutations per GPU forward pass |
| `seq_only` | `bool` | `False` | Score without sidechain context |
| `seed` | `int` | `42` | Random seed |

## Output Specification

### Sampling Output

Contains `FAMPNNSequences` objects (one per input structure):

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Designed amino acid sequences |
| `output_pdb_strings` | `List[str]` | PDB strings with designed sequences and sidechain coordinates |
| `psce` | `List[List[float]]` | Per-residue predicted sidechain error (Angstroms) |

### Packing Output (`FAMPNNPackingResult`)

| Field | Type | Description |
|-------|------|-------------|
| `packed_structures` | `List[List[str]]` | PDB strings (outer=structures, inner=samples) |
| `psce` | `List[List[List[float]]]` | Per-residue pSCE for each sample |

### Scoring Output (`FAMPNNScoreOutput`)

| Field | Type | Description |
|-------|------|-------------|
| `results[].mutations` | `List[str]` | Mutation strings that were scored |
| `results[].scores` | `List[float]` | Log-likelihood ratios (positive = favored over wild-type) |

### All Mutations Output (`FAMPNNScoreAllMutationsOutput`)

| Field | Type | Description |
|-------|------|-------------|
| `results[].scores` | `Dict[str, Dict[str, float]]` | Position label -> {residue: score} matrix |

## Interpreting Results

**For pSCE (sampling & packing):**
- **Excellent:** `pSCE < 0.5 A` (high confidence sidechain placement)
- **Good:** `pSCE < 1.0 A` (reasonable sidechain prediction)
- **Marginal:** `pSCE < 2.0 A` (uncertain sidechain)
- **Poor:** `pSCE > 2.0 A` (likely incorrect sidechain)

**For mutation scores (scoring):**
- **Positive scores**: Mutation is more likely than wild-type (potentially beneficial)
- **Zero**: Wild-type (reference)
- **Negative scores**: Mutation is less likely than wild-type (potentially deleterious)

**Important caveats:**
- Scores are log-likelihood ratios, not absolute fitness measures.
- FAMPNN's advantage over backbone-only models is strongest for mutations that affect sidechain packing (buried residues, hydrophobic core).
- Validate predictions experimentally or with orthogonal computational methods.

## Quick Start Examples

**Example 1: Design sequences with sidechain co-generation**
```python
from proto_tools.tools.inverse_folding.fampnn import (
    run_fampnn_sample,
    FAMPNNSampleInput,
    FAMPNNStructureInput,
    FAMPNNSampleConfig,
)

inputs = FAMPNNSampleInput(
    inputs=[FAMPNNStructureInput(structure="my_protein.pdb")]
)
config = FAMPNNSampleConfig(
    num_sequences_per_structure=5,
    temperature=0.1,
    num_steps=100,
)
result = run_fampnn_sample(inputs, config)

for seq in result.designed_sequences[0].sequences:
    print(f"Designed: {seq[:50]}...")
```

**Example 2: Score mutations**
```python
from proto_tools.tools.inverse_folding.fampnn import (
    run_fampnn_score,
    FAMPNNScoreInput,
    FAMPNNScoreConfig,
    MutationInput,
)
from proto_tools.entities.structures import Structure

structure = Structure(structure_filepath_or_content="my_protein.pdb")
inputs = FAMPNNScoreInput(
    inputs=[MutationInput(
        structure=structure,
        mutations=["A0V", "G5L", "wt"],  # 0-indexed positions
    )]
)
result = run_fampnn_score(inputs, FAMPNNScoreConfig())

for mut, score in zip(result.results[0].mutations, result.results[0].scores):
    print(f"{mut}: {score:+.4f}")
```

**Example 3: Sidechain packing**
```python
from proto_tools.tools.inverse_folding.fampnn import (
    run_fampnn_pack,
    FAMPNNPackInput,
    FAMPNNPackStructureInput,
    FAMPNNPackConfig,
)

inputs = FAMPNNPackInput(
    inputs=[FAMPNNPackStructureInput(structure="my_protein.pdb")]
)
config = FAMPNNPackConfig(num_samples_per_structure=3)
result = run_fampnn_pack(inputs, config)

# Save best packing
with open("packed.pdb", "w") as f:
    f.write(result.packed_structures[0][0])
```

## Best Practices & Gotchas

**Parameter tuning:**
1. `num_steps`: Use 100 for best quality, 10 for fast iteration. Self-consistency is high even at 10 steps.
2. `temperature`: Start at 0.1 (conservative), increase for diversity. FAMPNN is less sensitive to temperature than ProteinMPNN.
3. `psce_threshold`: Default 0.3A is good for most cases. Increase to 0.5-1.0 if designs are too constrained.
4. `model_variant`: Use `"0.3"` for design, `"0.0"` for packing, `"0.3_cath"` for scoring.

**Common mistakes:**
1. **Mutation indexing**: FAMPNN uses **0-indexed** positions in mutation strings (e.g., `"A0V"` for position 0). This differs from PDB residue numbering.
2. **`excluded_amino_acids`**: Not supported by FAMPNN. Will raise `ValueError` if set.
3. **Missing sidechains in input PDB**: FAMPNN can handle missing sidechains (they become ghost atoms), but providing complete structures improves packing quality.

## References

**Primary citation:**
- Widatalla, T., Shuai, R.W., Hie, B.L., & Huang, P.S. (2025). "Sidechain conditioning and modeling for full-atom protein sequence design with FAMPNN." *Proceedings of the 42nd International Conference on Machine Learning (ICML)*.

**Documentation:**
- GitHub: [https://github.com/richardshuai/fampnn](https://github.com/richardshuai/fampnn)

## Related Tools

**Tools often used together:**
- `esmfold` / `boltz2` / `chai1`: Validate that designed sequences fold correctly
- `proteinmpnn-sample`: Faster backbone-only inverse folding (when sidechains aren't needed)
- `ligandmpnn-sample`: Ligand-aware sequence design
- `rfdiffusion3`: Generate novel backbone structures as input to FAMPNN

**Alternative tools:**
- `proteinmpnn-sample`: Backbone-only inverse folding (faster, no sidechain prediction)
- `ligandmpnn-sample`: Supports ligand conditioning
