<a href="https://bio-pro.mintlify.app/tools/inverse-folding/ligandmpnn"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# LigandMPNN

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

LigandMPNN is an inverse folding model that designs protein sequences conditioned on a backbone structure and its molecular context -- including bound ligands, metal ions, nucleic acids, and non-standard residues. It extends ProteinMPNN's message-passing neural network architecture to incorporate atomic-level information from the non-protein environment surrounding the design target.

- **Tool keys**: `ligandmpnn-sample`, `ligandmpnn-score`
- **Input**: Protein structures (PDB/CIF) with optional chain and position constraints
- **Output**: Designed sequences and sequence-structure scores
- **Execution**: GPU required

## Background

[Inverse folding](https://en.wikipedia.org/wiki/Protein_design#Inverse_folding) solves the "reverse" protein design problem: given a desired 3D backbone structure, what amino acid sequence will fold into that structure? This is the complement of structure prediction (sequence -> structure).

ProteinMPNN (Dauparas et al., 2022) pioneered the [message-passing neural network](https://en.wikipedia.org/wiki/Graph_neural_network) approach for inverse folding, achieving recovery rates (~50% native sequence identity) far exceeding previous physics-based methods. However, ProteinMPNN only considers the protein backbone and ignores non-protein molecules.

LigandMPNN extends this by encoding the atomic coordinates and chemical identities of:
- **Small-molecule ligands** (drugs, metabolites, [cofactors](https://en.wikipedia.org/wiki/Cofactor_(biochemistry)))
- **Metal ions** (Zn, Fe, Mg, Ca, etc.)
- **Nucleic acids** (DNA and RNA)
- **Non-standard residues** (modified amino acids, [post-translational modifications](https://en.wikipedia.org/wiki/Post-translational_modification))

This context is critical for designing functional enzymes, metalloprotein binding sites, and protein-nucleic acid interfaces, where the sequence must be compatible with both the protein fold and its molecular partners.

## Tools

### LigandMPNN Sampling (`ligandmpnn-sample`)

Sample protein sequences using LigandMPNN.

### LigandMPNN Scoring (`ligandmpnn-score`)

Score sequence-structure compatibility with LigandMPNN.

## Tool Catalog

| Tool Key | Status | Description |
|----------|--------|-------------|
| `ligandmpnn-sample` | Available | Sample protein sequences conditioned on structure + ligand context |
| `ligandmpnn-score` | Available | Score sequences against structures with ligand context |

## Execution Modes

| Mode | Backend | Device |
|------|---------|--------|
| Local venv | `ToolInstance("ligandmpnn")` running `standalone/inference.py` | Local GPU (`cuda`) |


## How It Works

1. **Structure parsing**: The input PDB/CIF is parsed to extract backbone coordinates (N, CA, C, O) for all chains plus atomic coordinates and identities for any non-protein molecules
2. **Graph construction**: A k-nearest-neighbor graph is built over backbone residues, with edges encoding spatial relationships
3. **Context encoding**: Ligand atoms, metal ions, and nucleic acid bases within the interaction radius are encoded as additional node features
4. **Message passing**: The neural network performs multiple rounds of message passing along graph edges, aggregating structural and chemical context
5. **Sequence sampling**: At each position, the model outputs a probability distribution over amino acids, and sequences are sampled autoregressively at the specified temperature

## Input Parameters

### Sampling Input

`InverseFoldingInput` wraps a list of `InverseFoldingStructureInput` objects:

| Field | Type | Description |
|-------|------|-------------|
| `inputs` | `List[InverseFoldingStructureInput]` | One entry per structure to design |

Each `InverseFoldingStructureInput` contains:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `structure` | `Structure \| str \| Path` | (required) | Protein structure. Accepts file path, PDB content string, or Structure object. |
| `chains_to_redesign` | `Optional[List[str]]` | `None` (all chains) | Chains to redesign. Non-listed chains provide structural context but are not designed. |
| `fixed_positions` | `Optional[Dict[str, List[int]]]` | `None` | Residue positions to keep fixed per chain (1-indexed). Fixed residues retain their native identity. |

### Scoring Input

`LigandMPNNScoringInput` wraps `SequenceStructurePair` entries:

| Field | Type | Description |
|-------|------|-------------|
| `sequence_structure_pairs` | `List[SequenceStructurePair]` | Sequence and structure pairs to score |

Each `SequenceStructurePair` contains:

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | `str` | Amino acid sequence to score. Length must match the protein residues parsed from the structure. |
| `structure` | `Structure` | Structure context for the sequence |

## Configuration

### Sampling Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_sequences_per_structure` | `int` | `1` | >= 1 | Total number of sequences to generate per input structure |
| `batch_size` | `Optional[int]` | `None` | >= 1 | Max sequences per GPU forward pass (defaults to `num_sequences_per_structure`) |
| `temperature` | `float` | `0.1` | 0.0-1.0 | Sampling temperature controlling sequence diversity |
| `excluded_amino_acids` | `Optional[List[str]]` | `None` | Any standard AAs | Amino acids forbidden in designed positions. Common: `["C"]` to avoid disulfides. |
| `model_type` | `Literal[...]` | `"ligand_mpnn"` | three variants | `ligand_mpnn` (default) or membrane variants (`per_residue_label_membrane_mpnn`, `global_label_membrane_mpnn`) |
| `ligand_mpnn_use_atom_context` | `bool` | `True` | -- | Encode ligand atom context in graph (ligand-aware variants only) |
| `ligand_mpnn_use_side_chain_context` | `bool` | `False` | -- | Condition on fixed-residue sidechain atoms |
| `ligand_mpnn_cutoff_for_score` | `float` | `8.0` | > 0.0 | Ligand-residue distance cutoff (A) for ligand-interface recovery score |
| `seed` | `int \| None` | `None` | any int | Random seed; a random integer is generated when unset |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cpu"` | Inference device |
| `verbose` | `int` | `0` | 0-3 | Verbosity level |

### Scoring Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fixed_positions` | `Optional[Dict[str, List[int]]]` | `None` | Residues excluded from score aggregation |
| `return_logits` | `bool` | `False` | Include per-position logits |
| `model_type` | `"ligand_mpnn"` | `"ligand_mpnn"` | LigandMPNN model type to load |
| `scoring_mode` | `"single_aa" \| "autoregressive"` | `"single_aa"` | Single-position or autoregressive sequence-conditioned probabilities |
| `seed` | `int \| None` | `None` | Random seed for decoding order |
| `device` | `str` | `"cuda"` | Inference device |

`single_aa` scores each position conditioned on the backbone and all other sequence positions. `autoregressive` scores one seed-determined decoding order.

### Parameter Guides

**Temperature:**

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Deterministic (argmax at each position) | Single best-guess sequence; highest predicted recovery |
| 0.1 | Low diversity (default) | Conservative redesign; high sequence similarity across samples |
| 0.2-0.3 | Moderate diversity | Balanced exploration; good for generating small libraries |
| 0.5-0.7 | High diversity | Broad sequence search; useful for directed evolution starting points |
| 1.0 | Maximum diversity | Near-random sampling from the learned distribution |

**Batch size vs. temperature trade-offs:**

| Strategy | num_sequences_per_structure | temperature | Purpose |
|----------|-----------|-------------|---------|
| Single best | 1 | 0.1 | Quick check of designability |
| Small library | 10-50 | 0.2 | Focused sequence exploration |
| Large library | 100-500 | 0.3-0.5 | Diverse library for experimental screening |

### Sweep Priorities

When used in optimization loops:
1. **Temperature** has the largest effect on sequence diversity and recovery rate
2. **Fixed positions** control which regions are designed vs. preserved
3. **Excluded amino acids** apply hard constraints (e.g., no cysteines)
4. **`num_sequences_per_structure`** controls throughput per optimization step

## Output Specification

### Sampling Output

`InverseFoldingOutput` contains:

| Field | Type | Description |
|-------|------|-------------|
| `designed_sequences` | `List[LigandMPNNSequences]` | One per input structure, in input order |

Each `LigandMPNNSequences` contains:

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Designed amino acid sequences (length = `num_sequences_per_structure`) |
| `sequence_recovery` | `List[float]` | Per-sequence fraction of designed residues matching the reference sequence (0.0-1.0) |
| `ligand_interface_sequence_recovery` | `List[float] \| None` | Per-sequence recovery restricted to ligand-interface residues (0.0-1.0); `None` when the input structure has no ligand |

Export formats: `fasta`, `json`

### Scoring Output

`LigandMPNNScoringOutput` contains one `InverseFoldingScoringMetrics` per input pair with `log_likelihood`, `avg_log_likelihood`, `perplexity`, and optional `logits`/`vocab`.

Export formats: `csv`, `json`

## Interpreting Results

- **Sequence recovery**: Compare designed sequences to the native sequence at each position. Recovery rates of 40-55% are typical for well-designed backbones.
- **Ligand contact positions**: Positions near the ligand often show higher conservation across samples, reflecting the model's learned binding-site preferences.
- **Diversity across samples**: At low temperature (0.1), most samples will be very similar. Increased diversity at higher temperatures is expected and useful for library generation.
- **Recovery metrics**: `sequence_recovery` and `ligand_interface_sequence_recovery` are aligned with `sequences` (parallel lists). Both are floats in `[0.0, 1.0]` — fraction of designed residues matching the input structure's reference sequence (overall vs. restricted to ligand-interface residues). When the input structure has no ligand, `ligand_interface_sequence_recovery` is `None` (the whole list, not per-element) — guard with `if designs.ligand_interface_sequence_recovery is not None:` before iterating.

## Quick Start Examples

**Basic sampling from a PDB file:**
```python
from proto_tools import (
    InverseFoldingStructureInput,
    LigandMPNNSampleConfig,
    LigandMPNNSampleInput,
    run_ligandmpnn_sample,
)

inputs = LigandMPNNSampleInput(
    inputs=[
        InverseFoldingStructureInput(structure="/path/to/enzyme_with_ligand.pdb")
    ]
)
config = LigandMPNNSampleConfig(num_sequences_per_structure=10, temperature=0.2)
result = run_ligandmpnn_sample(inputs, config)

for seq in result.designed_sequences[0].sequences:
    print(seq)
```

**Fixing active-site residues:**
```python
inputs = LigandMPNNSampleInput(
    inputs=[
        InverseFoldingStructureInput(
            structure="/path/to/enzyme_with_cofactor.pdb",
            fixed_positions={"A": [45, 67, 89, 112, 134]},
        )
    ]
)
config = LigandMPNNSampleConfig(
    num_sequences_per_structure=50,
    temperature=0.3,
    excluded_amino_acids=["C"],
)
result = run_ligandmpnn_sample(inputs, config)

print(f"Generated {len(result.designed_sequences[0].sequences)} sequences")
print(f"First sequence: {result.designed_sequences[0].sequences[0][:50]}...")
```

**Multiple structures in one call:**
```python
inputs = LigandMPNNSampleInput(
    inputs=[
        InverseFoldingStructureInput(structure="/path/to/structure1.pdb"),
        InverseFoldingStructureInput(structure="/path/to/structure2.pdb"),
        InverseFoldingStructureInput(structure="/path/to/structure3.pdb"),
    ]
)
config = LigandMPNNSampleConfig(num_sequences_per_structure=5, temperature=0.1)
result = run_ligandmpnn_sample(inputs, config)

for i, designs in enumerate(result.designed_sequences):
    print(f"Structure {i}: {len(designs.sequences)} sequences designed")
```

**Scoring a sequence against a structure:**
```python
from proto_tools import (
    LigandMPNNScoringConfig,
    LigandMPNNScoringInput,
    SequenceStructurePair,
    run_ligandmpnn_score,
)
from proto_tools.entities.structures import Structure

structure = Structure.from_file("/path/to/enzyme_with_ligand.pdb")
sequence = "".join(
    structure.get_chain_sequence(chain)
    for chain in structure.get_chain_ids()
)
score_inputs = LigandMPNNScoringInput(
    sequence_structure_pairs=[
        SequenceStructurePair(sequence=sequence, structure=structure)
    ]
)
score_result = run_ligandmpnn_score(
    score_inputs,
    LigandMPNNScoringConfig(scoring_mode="single_aa", seed=42),
)
print(score_result.scores[0].perplexity)
```

**Accessing per-sequence metrics:**
```python
result = run_ligandmpnn_sample(inputs, config)
designs = result.designed_sequences[0]
interface = designs.ligand_interface_sequence_recovery  # None when input has no ligand
for i, seq in enumerate(designs.sequences):
    iface_str = f"{interface[i]:.3f}" if interface is not None else "N/A (no ligand)"
    print(f"Seq {i}: {seq[:30]}... | recovery={designs.sequence_recovery[i]:.3f} | interface={iface_str}")
```

**Export results:**
```python
result.export("/path/to/output_dir", file_format="fasta")
result.export("/path/to/output_dir", file_format="json")
```

## Best Practices & Gotchas

- **Include ligands in your PDB.** LigandMPNN's advantage over ProteinMPNN is ligand-aware design. If your PDB does not contain HETATM records for ligands/ions, LigandMPNN reduces to ProteinMPNN and you should use ProteinMPNN directly (it is faster).
- **Fix catalytic residues.** For enzymes, always fix known catalytic residues using `fixed_positions`. LigandMPNN respects ligand context but does not guarantee catalytic geometry without explicit constraints.
- **Use one design constraint mode.** LigandMPNN accepts chain-based design (`chains_to_redesign`) or residue-based fixed positions (`fixed_positions`) in a single request, not both.
- **Temperature 0.1 is a safe default.** Start with the default and increase only if you need more diversity. Very high temperatures (>0.5) produce sequences that may not fold.
- **GPU is required.** LigandMPNN runs on GPU via CUDA. CPU execution is technically possible but impractically slow for typical batch sizes.
- **Chain IDs must exist in the structure.** If you specify `chains_to_redesign` values that are not present in the PDB, validation will raise an error with the available chains listed.
- **Positions are 1-indexed.** Fixed positions follow biological convention (first residue = position 1), not 0-indexed.

## References

- Dauparas, J., Lee, G.R., Pecoraro, R., et al. (2023). Atomic context-conditioned protein sequence design using LigandMPNN. *bioRxiv*. DOI: [10.1101/2023.12.22.573103](https://doi.org/10.1101/2023.12.22.573103)
- Dauparas, J., Anishchenko, I., Bennett, N., et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56. DOI: [10.1126/science.add2187](https://doi.org/10.1126/science.add2187)
- GitHub: https://github.com/dauparas/LigandMPNN

## Related Tools

**Often used together:**
- **ESMFold** (`esmfold-prediction`) -- Validate that designed sequences fold into the target structure
- **AlphaFold** (`alphafold3-prediction`) -- Higher-accuracy structure validation, especially for complexes
- **ProteinMPNN scoring** (`proteinmpnn-score`) -- Faster protein-only sequence-structure scoring
- **Segmasker** (`segmasker-score`) -- Filter designs with excessive low-complexity content

**Alternatives:**
- **ProteinMPNN** (`proteinmpnn-sample`) -- Faster inverse folding for protein-only structures without ligand context
- **ESM-IF** -- Language-model-based inverse folding (not currently wrapped in this toolkit)
