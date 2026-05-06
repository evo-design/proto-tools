<a href="https://bio-pro.mintlify.app/tools/inverse-folding/esm-if1"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# ESM-IF1 / ProteinDPO

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

ESM-IF1 is a language-model-based inverse folding model that designs protein sequences conditioned on backbone structure. It uses a GVP-GNN encoder to represent 3D structure and a Transformer decoder to autoregressively generate sequences. ProteinDPO is a fine-tuned variant aligned to experimental fitness data via Direct Preference Optimization, optimized for designing stable proteins.

- **Tool keys**: `esm-if1-sample`, `esm-if1-score`
- **Input**: Protein structures (PDB/CIF) with optional chain and position constraints
- **Output**: Designed amino acid sequences with log-likelihoods (sampling) or perplexity scores (scoring)
- **Execution**: GPU required

## Background

[Inverse folding](https://en.wikipedia.org/wiki/Protein_design#Inverse_folding) solves the "reverse" protein design problem: given a desired 3D backbone structure, what amino acid sequence will fold into that structure?

ESM-IF1 (Hsu et al., 2022) takes a different architectural approach from message-passing models like ProteinMPNN. It uses:

1. **GVP-GNN encoder**: A [Geometric Vector Perceptron](https://arxiv.org/abs/2009.01411) graph neural network encodes the backbone structure into per-residue representations that capture both scalar and geometric features.
2. **Transformer decoder**: A standard autoregressive Transformer generates the amino acid sequence one token at a time, attending to the structural encoding. This leverages the same architecture behind large language models.
3. **Training on predicted structures**: Unlike ProteinMPNN (trained on ~19K experimental structures), ESM-IF1 was trained on 12M predicted structures from ESM, giving it broader coverage of protein fold space.

[ProteinDPO](https://doi.org/10.1101/2024.05.20.595026) (Widatalla, Rafailov & Hie, 2024) fine-tunes ESM-IF1 using [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) on experimental fitness data, aligning the model to prefer sequences with higher measured stability. This is analogous to RLHF in language models -- the model learns to generate sequences that score well on real experimental assays.

## Tools

### ESM-IF1 Sampling (`esm-if1-sample`)

Sample protein sequences using ESM-IF1/ProteinDPO.

### ESM-IF1 Scoring (`esm-if1-score`)

Score protein sequences using ESM-IF1/ProteinDPO.

Scores each sequence against its paired structure using the full complex
structural context (score_sequence_in_complex). Returns average
log-likelihood and perplexity.

## Tool Catalog

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `esm-if1-sample` | Structure(s) | Designed sequences + log-likelihoods | Design new sequences for a target fold |
| `esm-if1-score` | Sequence + Structure pairs | Avg log-likelihood + perplexity | Evaluate sequence-structure compatibility |

## Execution Modes

| Mode | Backend | Device |
|------|---------|--------|
| Local venv | `ToolInstance("esm_if1")` running standalone subprocess | Local GPU (`cuda`) |

## How It Works

### Sampling (`esm-if1-sample`)

1. **Structure encoding**: The input PDB is parsed and the backbone coordinates (N, CA, C, O) are encoded by the GVP-GNN into per-residue structural features
2. **Context aggregation**: For multi-chain complexes, all chains contribute structural context even if only a subset is being designed
3. **Autoregressive decoding**: The Transformer decoder generates amino acids one position at a time, conditioned on the structural encoding and previously generated tokens
4. **Batched sampling**: Multiple sequences are generated in parallel batches for efficiency
5. **Log-likelihood computation**: Each designed sequence is scored by the model's own log-likelihood, providing a built-in confidence metric

### Scoring (`esm-if1-score`)

1. **Structure + sequence input**: A sequence-structure pair is provided
2. **Forward pass**: The model computes the conditional log-likelihood of the sequence given the structure
3. **Metrics**: Returns average log-likelihood and perplexity (lower perplexity = better fit)

## Input Parameters

### Sampling (`esm-if1-sample`)

`InverseFoldingInput` wraps a list of `InverseFoldingStructureInput` objects:

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `List[InverseFoldingStructureInput]` | One entry per structure to design |

Each `InverseFoldingStructureInput` contains:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `structure` | `Structure \| str \| Path` | (required) | Protein structure. Accepts file path, PDB content string, or Structure object. |
| `chain_ids` | `Optional[List[str]]` | `None` (all chains) | Chains to redesign. Non-listed chains provide structural context but are not designed. |
| `fixed_positions` | `Optional[Dict[str, List[int]]]` | `None` | Residue positions to keep fixed per chain (1-indexed). Fixed residues retain their native identity. |

### Scoring (`esm-if1-score`)

| Field | Type | Description |
|-------|------|-------------|
| `sequence_structure_pairs` | `List[SequenceStructurePair]` | List of (sequence, structure) pairs to score |

## Configuration

### Sampling Configuration (`ESMIF1SampleConfig`)

Extends `InverseFoldingConfig` with a weights variant selector:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights_variant` | `Literal["esmif", "protein_dpo"]` | `"protein_dpo"` | `"esmif"` for vanilla ESM-IF1, `"protein_dpo"` for DPO-aligned stability-optimized weights |
| `num_sequences_per_structure` | `int` | `1` | Total number of sequences to generate per input structure |
| `batch_size` | `Optional[int]` | `None` | Max sequences per GPU forward pass (defaults to `num_sequences_per_structure`) |
| `temperature` | `float` | `0.1` | Sampling temperature. Lower = more conservative, higher = more diverse |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `device` | `str` | `"cuda"` | Device for inference |

### Scoring Configuration (`ESMIF1ScoringConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights_variant` | `Literal["esmif", "protein_dpo"]` | `"protein_dpo"` | Which model weights to use |
| `device` | `str` | `"cuda"` | Device for inference |

### Parameter Guides

**Weights variant:**

| Variant | Description | Use Case |
|---------|-------------|----------|
| `"protein_dpo"` | DPO-aligned for stability (default) | General-purpose design; optimized for foldable, stable sequences |
| `"esmif"` | Vanilla ESM-IF1 | When you want the original model behavior, or for benchmarking |

**Temperature guide:**

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| `0.1` | Low diversity (default) | Conservative designs, high confidence |
| `0.3-0.5` | Moderate diversity | Balanced exploration for library generation |
| `0.7-1.0` | High diversity | Maximum sequence variation |

## Output Specification

### Sampling Output (`InverseFoldingOutput`)

Contains a list of `ESMIF1Sequences` objects, one per input structure:

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Designed amino acid sequences |
| `log_likelihoods` | `List[float]` | Average log-likelihood of each designed sequence under the model. Higher (less negative) = better. |

Export formats: `fasta`, `json`

### Scoring Output (`InverseFoldingScoringOutput`)

Contains a list of `InverseFoldingScoringMetrics` objects, one per input pair. Metric values are accessed via attribute-style (`score.perplexity`) or mapping-style (`score["perplexity"]`):

| Field | Type | Description |
|-------|------|-------------|
| `avg_log_likelihood`, `perplexity` | `float` | Scalar metrics (attribute or mapping access) |

Export formats: `csv`, `json`

## Interpreting Results

**Log-likelihood (sampling):**
- Higher (less negative) values indicate the model is more confident the sequence fits the structure
- Compare across designs for the same structure to rank candidates

**Perplexity (scoring):**
- **Excellent:** `perplexity < 2.0` (highly compatible)
- **Good:** `perplexity < 4.0` (reasonable designs)
- **Marginal:** `perplexity < 6.0` (may need optimization)
- **Poor:** `perplexity > 8.0` (likely incompatible)

**ProteinDPO vs ESM-IF1:** ProteinDPO tends to produce sequences with higher experimental fitness (stability, expression) at slight cost to diversity. Use `"esmif"` if you want unbiased sampling from the base model.

## Quick Start Examples

**Sample sequences with ProteinDPO (default):**
```python
from proto_tools.tools.inverse_folding.esm_if1 import run_esm_if1_sample, ESMIF1SampleConfig
from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingInput, InverseFoldingStructureInput,
)

inputs = InverseFoldingInput(
    inputs=[InverseFoldingStructureInput(structure="/path/to/protein.pdb")]
)
config = ESMIF1SampleConfig(num_sequences_per_structure=10, temperature=0.2)
result = run_esm_if1_sample(inputs, config)

for i, seq_res in enumerate(result.designed_sequences):
    for j, seq in enumerate(seq_res.sequences):
        ll = seq_res.log_likelihoods[j]
        print(f"Design {j+1}: {seq[:50]}... (log-likelihood: {ll:.4f})")
```

**Score a designed sequence against its target structure:**
```python
from proto_tools.tools.inverse_folding.esm_if1 import (
    run_esm_if1_score, ESMIF1ScoringInput, ESMIF1ScoringConfig,
)
from proto_tools.tools.inverse_folding.shared_data_models import SequenceStructurePair
from proto_tools.entities.structures import Structure

structure = Structure.from_file("/path/to/protein.pdb")
scoring_input = ESMIF1ScoringInput(
    sequence_structure_pairs=[
        SequenceStructurePair(sequence="MKTL...", structure=structure),
    ]
)
score_result = run_esm_if1_score(scoring_input, ESMIF1ScoringConfig())

for score in score_result.scores:
    print(f"Avg log-likelihood: {score.avg_log_likelihood:.4f}")
    print(f"Perplexity: {score.perplexity:.4f}")
```

**Use vanilla ESM-IF1 weights instead of ProteinDPO:**
```python
config = ESMIF1SampleConfig(
    weights_variant="esmif",
    num_sequences_per_structure=5,
    temperature=0.1,
)
result = run_esm_if1_sample(inputs, config)
```

## Best Practices & Gotchas

- **Temperature 0.1 is a safe default.** Increase for diversity; at very low temperatures, all sampled sequences may be identical.
- **GPU is required.** The GVP-GNN encoder + Transformer decoder are compute-intensive.
- **Multi-chain support.** Both sampling and scoring use the full complex structural context. Specify `chain_ids` to control which chains are designed.
- **Positions are 1-indexed.** Fixed positions follow biological convention.
- **Log-likelihood vs perplexity.** Sampling returns log-likelihoods; scoring returns both avg log-likelihood and perplexity. They are related: `perplexity = exp(-avg_log_likelihood)`.

## References

- Hsu, C., Verkuil, R., Liu, J., Lin, Z., Hie, B., Sercu, T., Lerer, A. & Rives, A. (2022). Learning inverse folding from millions of predicted structures. *International Conference on Machine Learning (ICML)*. [DOI: 10.1101/2022.04.10.487779](https://doi.org/10.1101/2022.04.10.487779)
- Widatalla, T., Rafailov, R. & Hie, B. (2024). Aligning protein generative models with experimental fitness via Direct Preference Optimization. *bioRxiv*. [DOI: 10.1101/2024.05.20.595026](https://doi.org/10.1101/2024.05.20.595026)
- GitHub: [https://github.com/facebookresearch/esm](https://github.com/facebookresearch/esm)
- GitHub (ProteinDPO): [https://github.com/evo-design/protein-dpo](https://github.com/evo-design/protein-dpo)

## Related Tools

**Tools often used together:**
- `esmfold` / `boltz2` / `chai1`: Validate that designed sequences fold correctly

**Alternative tools:**
- `proteinmpnn-sample`: Backbone-only inverse folding; supports excluded amino acids
- `ligandmpnn-sample`: Inverse folding with ligand, metal ion, and nucleic acid context
