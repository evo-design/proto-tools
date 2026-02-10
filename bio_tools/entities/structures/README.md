# Structures

## Overview
This module provides the `Structure` class - a unified representation for protein 3D structures. It handles loading from PDB/CIF files, format conversions, sequence extraction, chain manipulation, and visualization. The class serves as the standard structure representation throughout the bio-programming toolkit, bridging the gap between structure prediction outputs and downstream analysis.

## When to Use This Tool

**Primary use cases:**
- Loading and parsing protein structures (PDB/CIF files or strings)
- Extracting sequences from structural data
- Converting between PDB and mmCIF formats
- Visualizing structures with confidence coloring (pLDDT)
- Preparing structures for inverse folding or scoring tools

**When NOT to use this tool:**
- For sequence-only analysis: Use sequence tools directly
- For molecular dynamics: Use MDAnalysis, MDTraj, or OpenMM
- For small molecule structures: Use the `ligands` module
- For nucleic acid structures: This module is protein-focused

## Biological Background

**What is a protein structure?**
A protein's 3D structure determines its function. Structures are represented as atomic coordinates, typically including:
- **Backbone atoms**: N, Cα, C, O for each residue
- **Side chain atoms**: Variable per amino acid type
- **B-factors**: Temperature factors or prediction confidence (pLDDT)

**Structure file formats:**

| Format | Extension | Description |
|--------|-----------|-------------|
| **PDB** | `.pdb` | Legacy format, human-readable, limited to 99,999 atoms |
| **mmCIF** | `.cif` | Modern format, no size limits, machine-readable |

**What is pLDDT?**
Predicted Local Distance Difference Test (pLDDT) is a per-residue confidence metric from structure prediction tools (AlphaFold, ESMFold, Boltz2, etc.):
- **90-100**: Very high confidence (well-ordered domains)
- **70-89**: Confident prediction (most structured regions)
- **50-69**: Low confidence (flexible loops, disordered)
- **<50**: Very low confidence (likely unstructured)

pLDDT is often stored in the B-factor column of structure files.

## How It Works

**Key components:**

| Component | Description |
|-----------|-------------|
| `Structure` | Main class representing a protein structure |
| `BFactorType` | Enum indicating what B-factor column contains |
| Format detection | Automatically detects PDB vs CIF format |
| Gemmi backend | Uses Gemmi library for fast parsing |

**Supported B-factor types:**

| Type | Description |
|------|-------------|
| `TEMPERATURE_FACTOR` | Traditional crystallographic B-factor |
| `PLDDT` | 0-100 scale prediction confidence |
| `NORMALIZED_PLDDT` | 0-1 scale prediction confidence |
| `CONFIDENCE` | Generic confidence score |
| `UNKNOWN` | Unspecified |
| `UNSPECIFIED` | Default (not set) |

## Important Parameters

### Initialization

| Parameter | Type | Description |
|-----------|------|-------------|
| `structure_filepath_or_content` | `Path \| str` | Path to PDB/CIF file OR structure content as string |
| `b_factor_type` | `BFactorType` | What the B-factor column represents (default: `UNSPECIFIED`) |
| `metrics` | `Dict[str, float] \| None` | Optional metrics dictionary (e.g., scores, confidence) |
| `source` | `str \| None` | Optional source identifier (auto-set from filepath) |

### Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `style` | `str` | `"cartoon"` | Visualization style: `"cartoon"`, `"stick"`, `"sphere"` |

## Output Specification

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.structure_pdb` | `str` | Structure as PDB format string |
| `.structure_cif` | `str` | Structure as mmCIF format string |
| `.structure_format` | `str` | Detected format: `"pdb"` or `"cif"` |
| `.b_factor_type` | `BFactorType` | What B-factor column contains |
| `.source` | `str \| None` | Source filepath or identifier |
| `.num_chains` | `int` | Number of chains in structure |
| `.num_residues` | `int` | Total number of residues |
| `.metrics` | `Dict[str, float]` | User-defined metrics |
| `.gemmi_struct` | `gemmi.Structure` | Underlying Gemmi structure (lazy-loaded) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_chain_sequence(chain_id)` | `str` | Amino acid sequence of specified chain |
| `get_chain_sequences()` | `Dict[str, str]` | Dict mapping chain IDs to sequences |
| `get_chain_ids()` | `List[str]` | List of all chain identifiers |
| `get_residue_position_map()` | `Dict` | Mapping of chains to (residue, position) tuples |
| `write_pdb(filepath)` | `None` | Save structure as PDB file |
| `write_cif(filepath)` | `None` | Save structure as CIF file |
| `visualize(style)` | `py3Dmol view` | Interactive 3D visualization |
| `add_metric(name, value)` | `None` | Add a metric to the structure |

## Best Practices & Gotchas

**Loading structures:**

1. **Format auto-detection**: The class automatically detects PDB vs CIF format from content - no need to specify.

2. **String vs file**: You can pass either a filepath OR the actual file contents as a string.

3. **Validation**: Invalid structures raise `ValueError` during initialization.

**Format conversions:**

1. **PDB limitations**: PDB format has limitations (99,999 atom limit, chain ID restrictions). Prefer CIF for large structures.

2. **Conversion fidelity**: Some metadata may be lost when converting between formats.

**B-factor handling:**

1. **Set correctly for visualization**: If your structure has pLDDT scores in the B-factor column, set `b_factor_type=BFactorType.PLDDT` for correct color scaling.

2. **Normalized vs raw pLDDT**: Some tools output 0-1 scale (normalized), others 0-100. Set the type accordingly.

**Common mistakes:**

1. **Forgetting b_factor_type**: Visualization colors will be wrong if you don't specify what the B-factors represent.

2. **Modifying gemmi_struct directly**: Changes to the Gemmi structure won't be reflected in `.structure_pdb`/`.structure_cif` unless you update the internal representation.

3. **Chain ID assumptions**: Not all structures use 'A', 'B', 'C' - always use `get_chain_ids()` to discover available chains.

## Quick Start Examples

**Example 1: Load and inspect a structure**
```python
from bio_programming.bio_tools.entities.structures import Structure, BFactorType

# Load from PDB file
structure = Structure(
    "protein.pdb",
    b_factor_type=BFactorType.PLDDT  # If from AlphaFold/ESMFold
)

# Basic info
print(f"Chains: {structure.get_chain_ids()}")
print(f"Total residues: {structure.num_residues}")

# Get sequences
for chain_id, sequence in structure.get_chain_sequences().items():
    print(f"Chain {chain_id}: {len(sequence)} residues")
    print(f"  Sequence: {sequence[:50]}...")
```

**Example 2: Load from structure prediction output**
```python
from bio_programming.bio_tools.entities.structures import Structure, BFactorType

# Load from CIF string (e.g., from Boltz2 output)
cif_content = """data_predicted
...
"""
structure = Structure(
    cif_content,
    b_factor_type=BFactorType.PLDDT,
    source="boltz2-prediction"
)

# Add custom metrics
structure.add_metric("avg_plddt", 85.3)
structure.add_metric("ptm", 0.91)

# Visualize with confidence coloring
structure.visualize(style="cartoon")
```

**Example 3: Convert formats and save**
```python
from bio_programming.bio_tools.entities.structures import Structure

# Load PDB, save as CIF
structure = Structure("input.pdb")

# Get both format representations
pdb_str = structure.structure_pdb
cif_str = structure.structure_cif

# Save
structure.write_cif("output.cif")
structure.write_pdb("output.pdb")
```

**Example 4: Use with inverse folding**
```python
from bio_programming.bio_tools.entities.structures import Structure, BFactorType
from bio_programming.bio_tools.tools.inverse_folding import run_proteinmpnn_sample

# Load structure
structure = Structure("design_target.pdb")

# Extract sequence for reference
original_seq = structure.get_chain_sequence("A")
print(f"Original: {original_seq}")

# Use structure content for inverse folding
# (See inverse_folding module for full example)
```

**Example 5: Analyze multi-chain complex**
```python
from bio_programming.bio_tools.entities.structures import Structure

# Load a complex
complex_struct = Structure("antibody_antigen.cif")

# Iterate over chains
sequences = complex_struct.get_chain_sequences()
for chain_id, seq in sequences.items():
    print(f"Chain {chain_id}:")
    print(f"  Length: {len(seq)} residues")
    print(f"  Sequence: {seq[:30]}...")
```

## Pydantic Serialization

The `Structure` class supports Pydantic serialization for use in tool inputs/outputs:

```python
from pydantic import BaseModel
from bio_programming.bio_tools.entities.structures import Structure

class MyToolOutput(BaseModel):
    structure: Structure
    score: float

# Structures can be serialized/deserialized automatically
output = MyToolOutput(
    structure=Structure("protein.pdb"),
    score=0.95
)

# JSON serialization works out of the box
output.model_dump_json()
```

## Dependencies

- **Gemmi**: Fast structure parsing and manipulation
- **py3Dmol**: 3D visualization in Jupyter notebooks

## Related Modules

- `structure_prediction`: Predict structures (ESMFold, Boltz2, Chai1, AF3)
- `inverse_folding`: Design sequences for target structures (ProteinMPNN)
- `ligands`: Represent small molecule ligands
