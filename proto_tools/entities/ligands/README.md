# Ligands

## Overview
This module provides classes and utilities for representing **small molecule ligands** - molecules that bind to proteins through non-covalent interactions (hydrogen bonds, van der Waals, hydrophobic, ionic). It provides a unified interface for loading ligands from SMILES strings, SDF files, or RDKit Mol objects, generating 3D conformers, and visualizing molecular structures.

## When to Use This Tool

**Primary use cases:**
- Representing small molecules for structure prediction inputs (e.g., Boltz2, Chai1, AlphaFold3)
- Loading ligand libraries from SDF or SMILES files
- Generating 3D conformers for docking or visualization
- Converting between molecular representations (SMILES ↔ SDF ↔ RDKit Mol)

**When NOT to use this tool:**
- For peptide ligands > ~50 residues: Use protein sequence tools instead
- For covalent binders: This module assumes non-covalent interactions
- For virtual screening campaigns: Use specialized docking software (AutoDock, Glide)
- For ADMET prediction: Use dedicated cheminformatics pipelines (RDKit descriptors, ML models)

## Biological Background

**What are ligands?**
In structural biology, a **ligand** is any molecule that binds to a protein target. This includes:
- **Drug molecules**: Small organic compounds (~150-900 Da)
- **Cofactors**: NAD+, FAD, ATP, heme
- **Metal ions**: Zn²⁺, Mg²⁺, Ca²⁺
- **Natural substrates**: Metabolites, signaling molecules

**Why is this important?**
Understanding protein-ligand interactions is central to:
- **Drug discovery**: Designing molecules that bind to disease targets
- **Enzymology**: Understanding how enzymes bind and transform substrates
- **Signaling**: How hormones and neurotransmitters trigger cellular responses
- **Structure prediction**: Modern tools like AlphaFold3, Boltz2, and Chai1 can predict protein-ligand complexes

**SMILES Notation**
SMILES (Simplified Molecular-Input Line-Entry System) is a line notation for describing molecular structures:
- `CCO` → Ethanol
- `c1ccccc1` → Benzene (aromatic ring)
- `CC(=O)O` → Acetic acid
- `.` → Separates disconnected fragments (e.g., salts: `[Na+].[Cl-]`)

## How It Works

**Class hierarchy:**

| Class | Description | Use Case |
|-------|-------------|----------|
| `Fragment` | Single connected molecule | One drug, one cofactor |
| `Ligands` | Collection of fragments | Multi-ligand systems, libraries |

**Key operations:**

1. **Loading**: From SMILES strings, `.smi` files, `.sdf` files, or RDKit Mol objects
2. **Validation**: Ensures SMILES are valid, molecules have atoms
3. **Conformer Generation**: Uses RDKit's ETKDG algorithm with UFF minimization
4. **Export**: Write to `.smi` or `.sdf` files
5. **Visualization**: 3D rendering with py3Dmol

## Important Parameters

### Fragment Class

At least one of `smiles` or `ccd_code` must be provided.

| Parameter | Type | Description |
|-----------|------|-------------|
| `smiles` | `str \| None` | SMILES string for a single connected molecule |
| `ccd_code` | `str \| None` | CCD code (e.g. `"ATP"`); resolved to SMILES |
| `id` | `str \| None` | Optional free-form identifier (e.g. a chain letter) |
| `name` | `str \| None` | Optional human-readable molecule name (defaults to `None`) |
| `metrics` | `dict[str, float]` | Computed metrics for this fragment (defaults to `{}`) |

### Ligands Class

| Parameter | Type | Description |
|-----------|------|-------------|
| `fragments` | `list[Fragment]` | Explicit list of fragments |
| `smiles` | `str` | Shorthand kwarg: dot-separated SMILES, expanded into fragments |
| `ccd_codes` | `list[str]` | Shorthand kwarg: CCD codes, expanded into fragments |

Use the factories `Ligands.from_smiles(...)`, `Ligands.from_ccd_codes([...])`, `Ligands.from_file(path)`, or `Ligands.from_mols([...])` to load from common sources.

### Conformer Generation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_conformers` | `int` | 1 | Number of 3D conformers to generate |
| `random_seed` | `int \| None` | 42 | Random seed for reproducibility |
| `prune_rms_threshold` | `float` | 0.5 | RMS threshold (Å) for pruning similar conformers |

### Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `int` | 400 | Viewer width in pixels |
| `height` | `int` | 400 | Viewer height in pixels |
| `style` | `str` | `'stick'` | Visualization style: `'stick'`, `'sphere'`, etc. |

## Output Specification

### Fragment Properties

| Property | Type | Description |
|----------|------|-------------|
| `.smiles` | `str` | Canonical SMILES representation |
| `.mol` | `Chem.Mol` | Underlying RDKit molecule (with hydrogens) |
| `.name` | `str` | Molecule name |
| `.conformers` | `List[Conformer]` | Generated 3D conformers |
| `.metrics` | `Dict[str, float]` | User-defined metrics (e.g., scores) |

### Ligands Properties

| Property | Type | Description |
|----------|------|-------------|
| `.fragments` | `List[Fragment]` | All fragments in the collection |
| `.smiles` | `str` | Dot-separated SMILES of all fragments |
| `len(ligands)` | `int` | Number of fragments |

## Best Practices & Gotchas

**Loading molecules:**

1. **SMILES validation**: Invalid SMILES will raise a `ValueError`. Always catch exceptions when loading user input.

2. **Hydrogens**: RDKit `AddHs()` is called automatically - molecules include explicit hydrogens for conformer generation.

3. **Multi-fragment SMILES**: SMILES like `[Na+].[Cl-]` (salts) will be split into separate fragments when loaded into `Ligands`.

**Conformer generation:**

1. **Required for visualization**: `visualize()` will auto-generate 1 conformer if none exist.

2. **Embedding failures**: Some complex molecules may fail conformer generation. Wrap calls in try/except.

3. **Pruning**: Higher `prune_rms_threshold` = fewer, more diverse conformers. Lower = more conformers, potentially redundant.

**Common mistakes:**

1. **Forgetting conformers for SDF export**: `to_sdf()` requires at least one conformer per fragment. Call `generate_conformers()` first.

2. **Confusing Fragment vs Ligands**: Use `Fragment` for single molecules, `Ligands` for collections or mixed inputs.

3. **Name collisions**: Auto-generated names from SMILES can be cryptic. Provide meaningful names when possible.

## Quick Start

See [`examples/example.ipynb`](examples/example.ipynb) for a runnable walkthrough: building a ligand from SMILES, generating 3D conformers, exporting to SDF, loading cofactors by CCD code, and 3D visualization.

## Dependencies

- **RDKit**: Core cheminformatics functionality
- **py3Dmol**: 3D visualization in Jupyter notebooks

## Related Modules

- `structure_prediction`: Predict protein-ligand complex structures (Boltz2, Chai1, AF3)
- `structures`: Represent and manipulate protein structures
