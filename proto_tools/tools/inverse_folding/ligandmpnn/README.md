<a href="https://bio-pro.mintlify.app/tools/inverse-folding/ligandmpnn"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# LigandMPNN

## Overview

LigandMPNN is an inverse folding model that designs protein sequences conditioned on a backbone structure and its molecular context -- including bound ligands, metal ions, nucleic acids, and non-standard residues. It extends ProteinMPNN's message-passing neural network architecture to incorporate atomic-level information from the non-protein environment surrounding the design target.

- **Tool key**: `ligandmpnn-sample`
- **Input**: Protein structures (PDB/CIF) with optional chain and position constraints
- **Output**: Designed amino acid sequences with per-sequence metrics
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

## Tool Catalog

| Tool Key | Status | Description |
|----------|--------|-------------|
| `ligandmpnn-sample` | Available | Sample protein sequences conditioned on structure + ligand context |
| `ligandmpnn-score` | Not implemented | Score sequence-structure compatibility (stub only) |

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

## Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_sequences_per_structure` | `int` | `1` | >= 1 | Total number of sequences to generate per input structure |
| `batch_size` | `Optional[int]` | `None` | >= 1 | Max sequences per GPU forward pass (defaults to `num_sequences_per_structure`) |
| `temperature` | `float` | `0.1` | 0.0-1.0 | Sampling temperature controlling sequence diversity |
| `excluded_amino_acids` | `Optional[List[str]]` | `None` | Any standard AAs | Amino acids forbidden in designed positions. Common: `["C"]` to avoid disulfides. |
| `seed` | `int` | `42` | any int | Random seed for reproducibility |
| `device` | `str` | `"cuda"` | `"cuda"`, `"cpu"` | Inference device |
| `verbose` | `bool` | `False` | -- | Print status messages during execution |

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

`InverseFoldingOutput` contains:

| Field | Type | Description |
|-------|------|-------------|
| `designed_sequences` | `List[LigandMPNNSequences]` | One per input structure, in input order |

Each `LigandMPNNSequences` contains:

| Field | Type | Description |
|-------|------|-------------|
| `sequences` | `List[str]` | Designed amino acid sequences (length = `num_sequences_per_structure`) |
| `ligandmpnn_metrics` | `List[Dict[str, Any]]` | Per-sequence metrics from LigandMPNN |

Export formats: `fasta`, `json`

## Interpreting Results

- **Sequence recovery**: Compare designed sequences to the native sequence at each position. Recovery rates of 40-55% are typical for well-designed backbones.
- **Ligand contact positions**: Positions near the ligand often show higher conservation across samples, reflecting the model's learned binding-site preferences.
- **Diversity across samples**: At low temperature (0.1), most samples will be very similar. Increased diversity at higher temperatures is expected and useful for library generation.
- **Metrics dict**: The `ligandmpnn_metrics` field contains model-internal metrics that vary by LigandMPNN version. Inspect the keys for available metrics.

## Quick Start Examples

**Basic sampling from a PDB file:**
```python
from proto_tools.tools.inverse_folding.ligandmpnn import run_ligandmpnn_sample
from proto_tools.tools.inverse_folding.shared_data_models import (
    InverseFoldingConfig, InverseFoldingInput, InverseFoldingStructureInput,
)

inputs = InverseFoldingInput(
    inputs=[
        InverseFoldingStructureInput(structure="/path/to/enzyme_with_ligand.pdb")
    ]
)
config = InverseFoldingConfig(num_sequences_per_structure=10, temperature=0.2)
result = run_ligandmpnn_sample(inputs, config)

for seq in result.designed_sequences[0].sequences:
    print(seq)
```

**Designing specific chains with fixed active-site residues:**
```python
inputs = InverseFoldingInput(
    inputs=[
        InverseFoldingStructureInput(
            structure="/path/to/homodimer_with_cofactor.pdb",
            chain_ids=["A"],
            fixed_positions={"A": [45, 67, 89, 112, 134]},
        )
    ]
)
config = InverseFoldingConfig(
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
inputs = InverseFoldingInput(
    inputs=[
        InverseFoldingStructureInput(structure="/path/to/structure1.pdb"),
        InverseFoldingStructureInput(structure="/path/to/structure2.pdb"),
        InverseFoldingStructureInput(structure="/path/to/structure3.pdb"),
    ]
)
config = InverseFoldingConfig(num_sequences_per_structure=5, temperature=0.1)
result = run_ligandmpnn_sample(inputs, config)

for i, designs in enumerate(result.designed_sequences):
    print(f"Structure {i}: {len(designs.sequences)} sequences designed")
```

**Accessing per-sequence metrics:**
```python
result = run_ligandmpnn_sample(inputs, config)
for i, seq in enumerate(result.designed_sequences[0].sequences):
    metrics = result.designed_sequences[0].ligandmpnn_metrics[i]
    print(f"Seq {i}: {seq[:30]}... | Metrics: {metrics}")
```

**Export results:**
```python
result.export("/path/to/output_dir", file_format="fasta")
result.export("/path/to/output_dir", file_format="json")
```

## Best Practices & Gotchas

- **Include ligands in your PDB.** LigandMPNN's advantage over ProteinMPNN is ligand-aware design. If your PDB does not contain HETATM records for ligands/ions, LigandMPNN reduces to ProteinMPNN and you should use ProteinMPNN directly (it is faster).
- **Fix catalytic residues.** For enzymes, always fix known catalytic residues using `fixed_positions`. LigandMPNN respects ligand context but does not guarantee catalytic geometry without explicit constraints.
- **Temperature 0.1 is a safe default.** Start with the default and increase only if you need more diversity. Very high temperatures (>0.5) produce sequences that may not fold.
- **GPU is required.** LigandMPNN runs on GPU via CUDA. CPU execution is technically possible but impractically slow for typical batch sizes.
- **Chain IDs must exist in the structure.** If you specify `chain_ids` that are not present in the PDB, validation will raise an error with the available chains listed.
- **Positions are 1-indexed.** Fixed positions follow biological convention (first residue = position 1), not 0-indexed.
- **Scoring is not yet implemented.** The `ligandmpnn-score` tool key exists as a stub but is not functional. Use `proteinmpnn-score` for scoring protein-only contexts.

## References

- Dauparas, J., Lee, G.R., Pecoraro, R., et al. (2023). Atomic context-conditioned protein sequence design using LigandMPNN. *bioRxiv*. DOI: [10.1101/2023.12.22.573103](https://doi.org/10.1101/2023.12.22.573103)
- Dauparas, J., Anishchenko, I., Bennett, N., et al. (2022). Robust deep learning-based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56. DOI: [10.1126/science.add2187](https://doi.org/10.1126/science.add2187)
- GitHub: https://github.com/dauparas/LigandMPNN

## Related Tools

**Often used together:**
- **ESMFold** (`esmfold-prediction`) -- Validate that designed sequences fold into the target structure
- **AlphaFold** (`alphafold3-prediction`) -- Higher-accuracy structure validation, especially for complexes
- **ProteinMPNN scoring** (`proteinmpnn-score`) -- Score designed sequence-structure compatibility
- **Segmasker** (`segmasker-score`) -- Filter designs with excessive low-complexity content

**Alternatives:**
- **ProteinMPNN** (`proteinmpnn-sample`) -- Faster inverse folding for protein-only structures without ligand context
- **ESM-IF** -- Language-model-based inverse folding (not currently wrapped in this toolkit)
