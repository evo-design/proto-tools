<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ccd-lookup"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# CCD Lookup

## Overview

CCD Lookup wraps [`pdbeccdutils`](https://github.com/PDBeurope/ccdutils) — the official PDBe Python library for the wwPDB Chemical Component Dictionary (CCD) — to provide rich enrichment for CCD entries. Given a CCD code (e.g. `ATP`) or a SMILES string, it returns a `Ligands` collection of standard `Fragment` objects (ready to feed into downstream tools that take ligands as input) plus a parallel list of `CcdEnrichment` records carrying the rest of the CCD metadata: formula, descriptors (InChI / InChIKey / SMILES), parent component, RDKit physicochemical properties, optional UniChem cross-references (DrugBank / ChEMBL / PubChem), and an optional list of PDB structures using the ligand.

Atom- and bond-level detail is intentionally not re-serialized into custom Pydantic models — once you have a `Fragment`, it lazy-loads an RDKit `Mol` (`fragment.mol`) and you can iterate `mol.GetAtoms()` / `mol.GetBonds()` directly. For 2D / 3D visualization, call `fragment.visualize()` (uses py3Dmol) on the resolved fragment.

This tool is for **user-facing enrichment workflows** — notebooks, scripts, dashboards, ligand reports. For the microsecond SMILES → CCD lookup used by Fragment validation and other internal hot paths, use the in-process module `proto_tools.entities.ligands.ccd_utils`. That module is intentionally lightweight (canonical-SMILES + InChIKey lookup) and lives on the validation hot path. `ccd-lookup` runs in a standalone micromamba env because `pdbeccdutils` pulls in heavier dependencies (Pillow, scipy, networkx, gemmi) that don't belong in-process.

## Background

The [wwPDB Chemical Component Dictionary (CCD)](https://www.wwpdb.org/data/ccd) is the canonical dictionary of every chemical component ever observed in the Protein Data Bank — small-molecule ligands, modified amino acids, ions, cofactors, nucleotides, sugars, and more. Each component has a 1- to 5-character identifier (e.g. `ATP` for adenosine triphosphate, `HEM` for heme, `MG` for magnesium ion, `SEP` for phosphoserine). Each entry stores atoms, bonds, formula, IUPAC name, descriptors (SMILES / InChI / InChIKey), release status, and — for modified residues — a parent component (e.g. `SEP` → `SER`).

Cross-references via [UniChem](https://www.ebi.ac.uk/unichem/) link CCD entries to external chemistry databases (DrugBank, ChEMBL, PubChem, ChEBI, …), so the same molecule can be looked up across resources. This makes the CCD a natural pivot point between structural biology (PDB) and cheminformatics (drug databases).

## Tools

### CCD Lookup (`ccd-lookup`)

User-facing enrichment of wwPDB Chemical Component Dictionary entries. Accepts CCD codes (e.g. `"ATP"`) or SMILES strings (mixed batches allowed) and returns a `CcdLookupOutput` containing a `Ligands` collection plus parallel `CcdEnrichment` records (formula, descriptors, parent component, RDKit physchem properties, optional UniChem cross-references, optional PDB usage).

## Execution Modes

- **CPU only.** No GPU is used.
- **Disk:** ~70 MB one-time download of the bundled wwPDB CCD `components.cif` to `$PROTO_MODEL_CACHE/ccd_lookup/`.
- **Cold start:** the first call after env setup builds an in-memory SMILES / InChIKey index over the full CCD (~10–60 s depending on hardware). Subsequent calls in the same persistent worker are fast.
- **Network:** fully offline by default. Only `include_cross_references` (UniChem) and `include_pdb_usage` (RCSB) require network access.

## How It Works

**Identifier auto-detection.** Each entry in `identifiers` is classified at runtime: a 1–5 character alphanumeric string is treated as a CCD code; anything else is treated as a SMILES string.

**SMILES path.** The input is canonicalized via RDKit and looked up against a pre-built index over the bundled CCD by canonical SMILES and InChIKey. The index is constructed lazily on the first call and cached for the lifetime of the worker. If no match is found, the corresponding `CcdEnrichment.ccd_code` is `None`; the `Fragment` in `result.ligands` still holds the original SMILES.

**CCD path.** The entry is loaded directly from the bundled `components.cif` via `pdbeccdutils.core.ccd_reader`, used to populate the `Fragment` (smiles, ccd_code, name) and the parallel `CcdEnrichment` (formula, descriptors, physchem properties, …).

**Optional UniChem cross-references** (`include_cross_references=True`). The tool calls UniChem's REST API to map the CCD entry to external databases. Results populate `enrichment.cross_references`.

**Optional PDB usage** (`include_pdb_usage=True`). The tool calls the RCSB search API to find PDB structures containing the ligand. Results populate `enrichment.pdb_structures`.

## Input Parameters

| Parameter | Type | Description |
|---|---|---|
| `identifiers` | `list[str]` | CCD codes (e.g. `"ATP"`, `"HEM"`) or SMILES strings (e.g. `"CC(=O)NC1=CC=C(C=C1)O"`). Mixed batches allowed. |

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_cross_references` | `bool` | `False` | Fetch external database cross-references via UniChem. **Requires network.** |
| `include_pdb_usage` | `bool` | `False` | Fetch the list of PDB structures containing the ligand via RCSB. **Requires network.** |
| `sanitize` | `bool` | `True` | Sanitize the parsed RDKit molecule. Disable only when working with intentionally non-standard valences. |

The inherited `verbose` and `timeout` fields from `BaseConfig` are also available; they don't affect computation and are excluded from the cache key.

## Output Specification

**`CcdLookupOutput`**

| Field | Type | Description |
|---|---|---|
| `ligands` | `Ligands` | A `Ligands` collection of `Fragment` objects (one per input identifier, in input order). Reuses `proto_tools.entities.ligands.Fragment` so the output drops directly into any tool that takes ligands as input. |
| `enrichments` | `list[CcdEnrichment]` | Parallel to `ligands.fragments`. Holds the CCD-specific metadata that doesn't fit on `Fragment`. |
| `num_resolved` | `int` | Number of identifiers with a CCD match (computed). |
| `num_unresolved` | `int` | Number of identifiers without a CCD match (computed). |

**`Fragment`** (from `proto_tools.entities.ligands`)

| Field | Type | Description |
|---|---|---|
| `smiles` | `str \| None` | Canonical SMILES (auto-resolved from `ccd_code` when only the code was supplied). |
| `ccd_code` | `str \| None` | CCD identifier (e.g. `"ATP"`). `None` when the input SMILES had no CCD match. |
| `name` | `str \| None` | CCD component name (populated from pdbeccdutils). |
| `metrics` | `dict[str, float]` | Empty by default; available for downstream annotation. |
| `mol` | `Chem.Mol` | Lazy-loaded RDKit molecule (use `mol.GetAtoms()` / `mol.GetBonds()` for atom/bond iteration). |

**`CcdEnrichment`**

| Field | Type | Description |
|---|---|---|
| `ccd_code` | `str \| None` | Mirrors `Fragment.ccd_code` for convenience. `None` when a SMILES input has no match. |
| `formula` | `str \| None` | Molecular formula (e.g. `"C10 H16 N5 O13 P3"`). |
| `formula_weight` | `float \| None` | Molecular weight in Daltons (from the CCD record). |
| `inchi` | `str \| None` | InChI string. |
| `inchikey` | `str \| None` | InChIKey. |
| `released` | `bool` | Whether the component is publicly released by wwPDB. |
| `release_status` | `str \| None` | Release status string from the CCD (e.g. `"REL"`, `"OBS"`). |
| `parent_ccd_code` | `str \| None` | Parent component for derivatives (e.g. `"SER"` for `"SEP"`). Best-effort. |
| `physchem_properties` | `dict[str, float]` | RDKit physicochemical descriptors. |
| `cross_references` | `dict[str, list[str]] \| None` | UniChem cross-refs by source name. Populated only when `include_cross_references=True`. |
| `pdb_structures` | `list[str] \| None` | PDB IDs containing the ligand. Populated only when `include_pdb_usage=True`. |
| `warnings` | `list[str]` | Non-fatal issues encountered while parsing the entry. |
| `errors` | `list[str]` | Per-record errors (the overall tool call still succeeds; check this list per record). |

## Interpreting Results

**`ccd_code is None`.** A SMILES input had no match in the bundled CCD. The corresponding `Fragment` in `result.ligands` still holds the original SMILES, and you can decide downstream whether to register a new ligand, fall back to a substructure search, or skip the entry. CCD inputs that are not present in the dictionary populate `enrichment.errors` and also have `ccd_code=None`.

**`formula_weight`.** Molecular weight in Daltons, taken directly from the CCD record. For unmatched SMILES this is computed by RDKit when possible, otherwise `None`.

**`parent_ccd_code`.** For modified residues and derivatives, this points to the parent (e.g. `SEP` (phosphoserine) → `SER`, `MSE` (selenomethionine) → `MET`). Parsing is best-effort from the underlying CIF; expect `None` for many small-molecule ligands.

**`physchem_properties`.** Canonical snake_case RDKit descriptors: `molecular_weight`, `exact_molecular_weight`, `logp`, `tpsa`, `num_h_donors`, `num_h_acceptors`, `num_rotatable_bonds`, `num_heavy_atoms`, plus `num_aromatic_rings` and `fraction_csp3` when pdbeccdutils provides them. Values are floats.

**`pdb_structures`.** Common ligands (e.g. `HEM`, `ATP`, `NAG`, `MG`, `ZN`) appear in many thousands of PDB entries; the returned list can be very large. The RCSB search API can also be rate-limited under heavy use — keep this opt-in.

**`warnings` vs `errors`.** Warnings flag non-fatal issues (e.g. RDKit could not compute a particular descriptor); the rest of the enrichment is still valid. Errors flag record-level failures (e.g. the CCD code does not exist); the surrounding tool call still returns success and processes the other identifiers.

**Atom / bond access and visualization.** The `Fragment` lazy-loads an RDKit `Mol` on first access — use `fragment.mol.GetAtoms()` / `fragment.mol.GetBonds()` for atom-level detail and `fragment.visualize()` (or `result.ligands.visualize()`) for 2D / 3D rendering via py3Dmol.

## Quick Start Examples

**Example 1: Enrich a single CCD code (offline, default config)**

```python
from proto_tools.tools.database_retrieval.ccd_lookup import (
    run_ccd_lookup, CcdLookupInput,
)

result = run_ccd_lookup(CcdLookupInput(identifiers=["ATP"]))  # Config is optional

frag = result.ligands.fragments[0]
enr = result.enrichments[0]

print(f"{frag.ccd_code}: {frag.name}")
print(f"Formula: {enr.formula} ({enr.formula_weight:.2f} Da)")
print(f"InChIKey: {enr.inchikey}")
print(f"{frag.mol.GetNumAtoms()} atoms, {frag.mol.GetNumBonds()} bonds")
```

**Example 2: Enrich a SMILES (offline)**

```python
from proto_tools.tools.database_retrieval.ccd_lookup import (
    run_ccd_lookup, CcdLookupInput,
)

# Acetaminophen / paracetamol — CCD code: TYL
result = run_ccd_lookup(CcdLookupInput(identifiers=["CC(=O)NC1=CC=C(C=C1)O"]))

frag = result.ligands.fragments[0]
enr = result.enrichments[0]
if enr.ccd_code is not None:
    print(f"SMILES matched CCD: {enr.ccd_code} ({frag.name})")
    print(f"Formula: {enr.formula}")
else:
    print("No CCD match for this SMILES.")
```

**Example 3: Mixed batch (CCD codes + SMILES)**

```python
from proto_tools.tools.database_retrieval.ccd_lookup import (
    run_ccd_lookup, CcdLookupInput,
)

inputs = CcdLookupInput(identifiers=[
    "ATP",                              # CCD code — known
    "HEM",                              # CCD code — known
    "CC(=O)NC1=CC=C(C=C1)O",            # SMILES — acetaminophen (TYL)
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCN",  # SMILES — likely no CCD match
])
result = run_ccd_lookup(inputs)

print(f"Resolved: {result.num_resolved} / {len(result.enrichments)}")

for frag, enr in zip(result.ligands.fragments, result.enrichments):
    if enr.ccd_code is None:
        print(f"  (no match) input SMILES: {frag.smiles}")
    else:
        print(f"  {enr.ccd_code}: {frag.name} [{enr.formula}]")
```

**Example 4: Network features (cross-references + PDB usage) and downstream chaining**

```python
from proto_tools.tools.database_retrieval.ccd_lookup import (
    run_ccd_lookup, CcdLookupInput, CcdLookupConfig,
)

result = run_ccd_lookup(
    CcdLookupInput(identifiers=["ATP"]),
    CcdLookupConfig(
        include_cross_references=True,  # UniChem (network)
        include_pdb_usage=True,         # RCSB (network)
    ),
)

enr = result.enrichments[0]
if enr.cross_references:
    for source, ids in enr.cross_references.items():
        print(f"{source}: {', '.join(ids[:3])}")
if enr.pdb_structures:
    print(f"Found in {len(enr.pdb_structures)} PDB entries")

# `result.ligands` is a `Ligands` collection ready to feed downstream tools
# that take ligands as input (e.g. AlphaFold3, Boltz2, Protenix), and
# `result.ligands.fragments[0].visualize()` renders it via py3Dmol.
```

> Note: Example 4's network features (`include_cross_references`, `include_pdb_usage`) require internet access and are opt-in. Default behavior is fully offline.

## Best Practices & Gotchas

1. **Cold start on first call.** The SMILES / InChIKey index over the full CCD is built lazily on the first call after env setup (~10–60 s). Subsequent calls in the same persistent worker are fast — use `ToolInstance.persist()` if you have many batches.

2. **Don't use this on the validation hot path.** For per-fragment SMILES → CCD lookups during scoring or validation, use `proto_tools.entities.ligands.ccd_utils.map_smiles_to_ccd_code` directly. It is the in-process equivalent and is orders of magnitude faster.

3. **SMILES with no CCD match silently returns `ccd_code=None`.** Always check `enrichment.ccd_code is not None` before treating an entry as "found." `result.num_unresolved` gives you the batch-level count.

4. **Network features are opt-in.** `include_cross_references` and `include_pdb_usage` require internet access. Both default to `False` so the tool is fully offline by default. RCSB and UniChem can be rate-limited; avoid hammering them with very large batches.

5. **`parent_ccd_code` is best-effort.** Parsed from the CIF when available. Expect `None` for many small-molecule ligands; expect a value for modified residues like `SEP`, `MSE`, `PTR`, etc.

6. **`pdb_structures` can be huge.** For common cofactors and ions (`HEM`, `ATP`, `NAG`, `MG`, `ZN`) the list runs to many thousands of PDB IDs. Stream or paginate downstream consumers accordingly.

7. **Visualization lives on `Fragment`.** Call `fragment.visualize()` on a resolved `result.ligands.fragments[i]` for 2D / 3D rendering via py3Dmol; this tool intentionally doesn't return SVGs.

## References

**Primary publication:**
- Kunnakkattu, I. R. *et al.* (2023). "PDBe CCDUtils: an RDKit-based toolkit for handling and analysing small molecules in the Protein Data Bank." *Journal of Cheminformatics* 15(1):117. DOI: [10.1186/s13321-023-00786-w](https://doi.org/10.1186/s13321-023-00786-w)

**Resources:**
- GitHub: https://github.com/PDBeurope/ccdutils
- wwPDB Chemical Component Dictionary: https://www.wwpdb.org/data/ccd
- UniChem: https://www.ebi.ac.uk/unichem/
- RCSB PDB: https://www.rcsb.org/

## Related Tools

**In-process equivalent (hot path):**
- `proto_tools.entities.ligands.ccd_utils.map_smiles_to_ccd_code`: microsecond SMILES → CCD code lookup using canonical SMILES + InChIKey. Use this on validation / scoring hot paths.

**Tools often used together:**
- `pdb-fetch-entry`: Fetch entry-level metadata (title, resolution, chains) for a specific PDB structure — pairs naturally with `pdb_structures` from `ccd-lookup`.
- `sequence-fetch`: Orchestrator for multiple sequence/structure databases (UniProt, NCBI, PDB) when you need entity-level metadata alongside chemical components.
