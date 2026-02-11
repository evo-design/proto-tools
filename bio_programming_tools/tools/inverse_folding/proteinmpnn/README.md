# ProteinMPNN

## Overview
ProteinMPNN is a deep learning model for protein sequence design given a protein backbone structure ("inverse folding"). It uses message passing neural networks to predict amino acid sequences that will fold into a target 3D structure. This module provides interfaces for *Sequence Sampling* (generating new sequences for a given backbone) and *Sequence Scoring* (evaluating how well a sequence fits a structure).

## When to Use This Tool

**Primary use cases:**
- Inverse folding: designing sequences that fold into target structures
- Protein stabilization through sequence optimization
- Scaffold-based protein design (e.g., grafting binding loops onto stable scaffolds)
- Generating diverse sequence libraries for experimental screening
- Evaluating sequence-structure compatibility

**When to use ProteinMPNN vs other tools:**
- ProteinMPNN: Best for designing sequences given a known or predicted backbone structure. State-of-the-art for inverse folding.
- ESM-IF1: Alternative inverse folding model with different architecture.
- RFdiffusion + ProteinMPNN: Use RFdiffusion to generate novel backbones, then ProteinMPNN to design sequences.
- ESM2/ESM3: Use for sequence-only analysis when you don't have a structure.

**When NOT to use this tool:**
- No Structure Available: ProteinMPNN requires a 3D backbone structure. If you only have a sequence, use structure prediction first (ESMFold, AlphaFold, Boltz2).
- De Novo Backbone Design: ProteinMPNN designs sequences, not structures. Use RFdiffusion or Chroma for backbone generation.
- Small Molecules/Ligands: ProteinMPNN is trained on protein backbones only. It doesn't model ligand interactions directly.
- Membrane Proteins: Performance may be reduced for membrane proteins due to training data bias.

## Biological Background

**What does this tool do?**
ProteinMPNN solves the "inverse folding" problem: given a protein backbone structure (the 3D coordinates of N, Cα, C, O atoms), predict which amino acid sequence will fold into that structure. This is the inverse of structure prediction.

**Why is this important?**
Inverse folding enables:
- Protein engineering: Redesign natural proteins with improved stability, solubility, or expression.
- Scaffold design: Create proteins with desired shapes for binding, catalysis, or assembly.
- Therapeutic development: Design antibodies, enzymes, and binding proteins.
- Experimental validation: Generate multiple sequence candidates for a single structure to find the best experimental hits.

**Scientific foundation:**
ProteinMPNN uses a message passing neural network architecture:

1. **Graph Representation**: The protein backbone is represented as a graph where each residue is a node and edges connect spatially close residues (within ~30Å).
2. **Message Passing**: Information flows between connected residues over multiple rounds, allowing the model to learn long-range dependencies.
3. **Autoregressive Decoding**: Sequences are generated one residue at a time, conditioned on previously generated residues and the full structural context.
4. **Training**: The model was trained on ~19,000 protein structures from the PDB to maximize the probability of native sequences given their structures.

## How It Works

**Method overview:**
The module exposes two distinct functions:

`proteinmpnn-sample`: Generates new protein sequences for a given backbone structure. Outputs multiple sequence candidates with diversity controlled by temperature. Returns perplexity scores and sequence identity to the original structure's sequence.

`proteinmpnn-score`: Evaluates how well a given sequence fits a structure. Returns perplexity (lower is better) and per-position logits for detailed analysis.

**The Tool Matrix:**
| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `proteinmpnn-sample` | Structure(s) | Designed sequences + metrics | Design new sequences for a target fold |
| `proteinmpnn-score` | Sequence + Structure pairs | Perplexity + logits | Evaluate sequence-structure compatibility |

**Computational Requirements:**

GPU: Required. ProteinMPNN runs on CUDA-enabled GPUs. CPU inference is possible but very slow.

Speed: ~1-2 seconds per structure for sampling 10 sequences on modern GPUs.

RAM: ~2-4GB GPU memory for typical single-chain proteins (<500 residues).

Model Size: ~150MB model weights (downloaded automatically).

## Important Parameters

### Input (What You're Designing)

**Sampling (`proteinmpnn-sample`)**
| Parameter | Type | Description |
|-----------|------|-------------|
| `structures` | `List[Structure]` or `List[str]` | Backbone structures as Structure objects or PDB file paths/content. |
| `all_chain_ids` | `Optional[List[List[str]]]` | Which chains to design for each structure. If `None`, designs all chains. |

**Scoring (`proteinmpnn-score`)**
| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence_structure_pairs` | `List[SequenceStructurePair]` | List of (sequence, structure) pairs to score. |

### Configuration (How to Design)

**Sampling Configuration (`InverseFoldingConfig`)**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | `1` | Number of sequences to generate per structure. |
| `temperature` | `float` | `0.1` | Sampling temperature (0.0-1.0). Lower = more conservative, higher = more diverse. |
| `fixed_positions` | `Optional[Dict[str, List[int]]]` | `None` | Residues to keep fixed (not redesigned). Maps chain IDs to position lists. |
| `excluded_amino_acids` | `Optional[List[str]]` | `None` | Amino acids to exclude from design (e.g., `["C"]` to avoid cysteines). |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `device` | `str` | `"cuda"` | Device for inference (`"cuda"` or `"cpu"`). |

**Scoring Configuration (`ProteinMPNNScoringConfig`)**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fixed_positions` | `Optional[Dict[str, List[int]]]` | `None` | Positions to exclude from perplexity calculation. |
| `seed` | `int` | `42` | Random seed. |
| `device` | `str` | `"cuda"` | Device for inference. |

### Temperature Guide
| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| `0.0` | Deterministic (argmax) | Most likely sequence, no diversity |
| `0.1` | Low diversity (default) | Conservative designs, high confidence |
| `0.3-0.5` | Moderate diversity | Balanced exploration |
| `0.7-1.0` | High diversity | Maximum sequence variation |

## Output Specification

**Sampling Output (`InverseFoldingOutput`)**

Contains a list of `ProteinMPNNSequences` objects, one per input structure:

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Designed amino acid sequences. Multi-chain sequences are "/"-delimited (e.g., `"MASCQT/EVQLVE"`). |
| `perplexity` | `List[float]` | Perplexity for each sequence. **Lower is better.** Typical range: 1.5-8.0. |
| `sequence_identity` | `List[float]` | Identity to the original PDB sequence (0.0-1.0). |

**Scoring Output (`ProteinMPNNScoringOutput`)**

Contains a list of `SequenceScores` objects, one per input pair. Metrics can be accessed via attribute-style (`score.perplexity`) or dict-style (`score.metrics["perplexity"]`):

| Field | Type | Description |
|-------|------|-------------|
| `scores` | `List[SequenceScores]` | List of scores, one per input sequence-structure pair. |
| `vocab` | `Optional[List[str]]` | Token ordering for logits: `logits[:, j]` corresponds to `vocab[j]`. |

Each `SequenceScores` contains:

| Field | Type | Description |
|-------|------|-------------|
| `metrics` | `Dict[str, float]` | Dictionary containing `log_likelihood`, `avg_log_likelihood`, and `perplexity`. |
| `logits` | `Optional[List[List[float]]]` | Per-position logits array of shape `(seq_length, vocab_size)`. |

Common metrics:
- `perplexity`: Perplexity of the sequence given the structure. **Lower is better.**
- `log_likelihood`: Sum of log probabilities for the sequence.
- `avg_log_likelihood`: Average log probability per position.

## Thresholds & Decision Boundaries

**For Perplexity (Sampling & Scoring):**
- **Excellent:** `perplexity < 2.0` (highly compatible sequence-structure pair)
- **Good:** `perplexity < 4.0` (reasonable designs)
- **Marginal:** `perplexity < 6.0` (may need optimization)
- **Poor:** `perplexity > 8.0` (likely incompatible)

**For Sequence Identity:**
- `>0.9`: Nearly identical to input (low temperature or constrained design)
- `0.5-0.9`: Moderate redesign
- `<0.5`: Significant redesign (high temperature or few constraints)

**Interpreting Results:**
- Perplexity is exponential—a perplexity of 2.0 means the model is "uncertain" between ~2 amino acids per position on average.
- Low perplexity ≠ guaranteed folding. Always validate designs with structure prediction (ESMFold, AlphaFold) or experimental characterization.
- Multiple low-perplexity sequences can fold into the same structure—sample many and filter.

## Best Practices & Gotchas

**Parameter Tuning:**

1. `temperature`:
   - Start with `0.1` for initial designs.
   - Increase to `0.3-0.5` if you need more diversity.
   - Use `0.0` only when you want a single deterministic answer.

2. `batch_size`:
   - Generate 10-100 sequences and filter by perplexity.
   - More sequences = more chances of finding good designs.

3. `fixed_positions`:
   - Use to preserve catalytic residues, binding sites, or experimentally validated positions.
   - Positions are **1-indexed** and must match the PDB residue numbering.
   - Format: `{"A": [1, 2, 3], "B": [10, 11]}` for chains A and B.

4. `excluded_amino_acids`:
   - Common: `["C"]` to avoid disulfide complications.
   - Use `["M"]` to avoid oxidation-sensitive methionines.

**Common Mistakes:**

1. **Wrong Position Indexing:** Fixed positions must match PDB residue numbering (usually 1-indexed), not 0-indexed Python arrays.

2. **Ignoring Multi-Chain Format:** Multi-chain outputs are "/"-delimited: `"CHAINASEQUENCE/CHAINBSEQUENCE"`. Parse accordingly.

3. **Over-Trusting Perplexity:** Low perplexity suggests compatibility, not guaranteed folding. Always validate with structure prediction.

4. **No Structure Validation:** After designing sequences, predict their structures with ESMFold/AlphaFold and check RMSD to the target backbone.

5. **CPU Inference:** Running on CPU is possible but extremely slow. Always use GPU when available.

6. **Mismatched Chain IDs:** Ensure `fixed_positions` chain IDs match the chains in your structure. Check with `structure.get_chain_ids()`.

## References

**Primary Citation:**
- Dauparas, J. et al. (2022). "Robust deep learning-based protein sequence design using ProteinMPNN." *Science* 378(6615): 49-56. [DOI: 10.1126/science.add2187](https://doi.org/10.1126/science.add2187)

**Documentation:**
- GitHub: [https://github.com/dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- ColabDesign: [https://github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)

## Quick Start Examples

**Example 1: Design sequences for a single-chain protein**
```python
from bio_programming_tools.tools.inverse_folding import (
    run_proteinmpnn_sample,
    InverseFoldingInput,
    InverseFoldingConfig
)

# Load your structure
inputs = InverseFoldingInput(
    structures=["my_protein.pdb"]
)

# Configure sampling
config = InverseFoldingConfig(
    batch_size=10,
    temperature=0.1
)

# Run ProteinMPNN
result = run_proteinmpnn_sample(inputs, config)

# Get designed sequences
for i, seq in enumerate(result[0].sequences):
    perplexity = result[0].perplexity[i]
    print(f"Sequence {i+1}: {seq[:50]}... (perplexity: {perplexity:.2f})")
```

**Example 2: Design with fixed positions (preserve active site)**
```python
from bio_programming_tools.tools.inverse_folding import (
    run_proteinmpnn_sample,
    InverseFoldingInput,
    InverseFoldingConfig
)

inputs = InverseFoldingInput(
    structures=["enzyme.pdb"],
    all_chain_ids=[["A"]]  # Only design chain A
)

# Fix catalytic triad positions
config = InverseFoldingConfig(
    batch_size=20,
    temperature=0.2,
    fixed_positions={"A": [57, 102, 195]},  # Catalytic residues
    excluded_amino_acids=["C"]  # No cysteines
)

result = run_proteinmpnn_sample(inputs, config)

# Filter for best designs
best_designs = sorted(
    zip(result[0].sequences, result[0].perplexity),
    key=lambda x: x[1]
)[:5]

for seq, ppl in best_designs:
    print(f"Perplexity {ppl:.2f}: {seq}")
```

**Example 3: Score sequence-structure compatibility**
```python
from bio_programming_tools.tools.inverse_folding.proteinmpnn import (
    run_proteinmpnn_score,
    ProteinMPNNScoringInput,
    ProteinMPNNScoringConfig,
    SequenceStructurePair,
)
from bio_programming_tools.entities.structures import Structure

# Load structure
structure = Structure(structure_filepath_or_content="my_protein.pdb")

# Create sequence-structure pairs to score
pairs = [
    SequenceStructurePair(sequence="MVLSPADKTNVKAAWGK", structure=structure),
    SequenceStructurePair(sequence="MVLSAADKTNVKAAWGK", structure=structure),  # S->A mutation
]

inputs = ProteinMPNNScoringInput(sequence_structure_pairs=pairs)
config = ProteinMPNNScoringConfig()

result = run_proteinmpnn_score(inputs, config)

# Compare perplexities
for i, score in enumerate(result.scores):
    print(f"Sequence {i+1}: perplexity = {score.perplexity:.2f}")
```

**Example 4: Multi-chain complex design**
```python
from bio_programming_tools.tools.inverse_folding import (
    run_proteinmpnn_sample,
    InverseFoldingInput,
    InverseFoldingConfig
)

# Design both chains of a heterodimer
inputs = InverseFoldingInput(
    structures=["complex.pdb"],
    all_chain_ids=[["A", "B"]]  # Design both chains together
)

config = InverseFoldingConfig(
    batch_size=10,
    temperature=0.1
)

result = run_proteinmpnn_sample(inputs, config)

# Parse multi-chain output
for seq in result[0].sequences:
    chain_a, chain_b = seq.split("/")
    print(f"Chain A: {chain_a[:30]}...")
    print(f"Chain B: {chain_b[:30]}...")
    print()
```

**Example 5: Analyze per-position logits**
```python
import numpy as np
from bio_programming_tools.tools.inverse_folding.proteinmpnn import (
    run_proteinmpnn_score,
    ProteinMPNNScoringInput,
    ProteinMPNNScoringConfig,
    SequenceStructurePair,
    ALPHAFOLD_VOCAB,
)

# Score a sequence
# ... (setup as in Example 3)

result = run_proteinmpnn_score(inputs, config)

# Get per-position probabilities
logits = result[0].logits
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

# Find positions with low confidence
for pos in range(len(probs)):
    max_prob = probs[pos].max()
    best_aa = ALPHAFOLD_VOCAB[probs[pos].argmax()]
    if max_prob < 0.5:
        print(f"Position {pos+1}: best={best_aa} (prob={max_prob:.2f}) - consider redesign")
```

## Related Tools

- `esmfold` / `boltz2` / `chai1`: Use after ProteinMPNN to validate that designed sequences fold correctly.
- `esm2-embeddings`: Use for sequence-level analysis of designed proteins.
- `mmseqs-clustering`: Use to cluster designed sequences and select diverse representatives for experimental testing.
