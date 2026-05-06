<a href="https://bio-pro.mintlify.app/tools/database-retrieval/pubchem"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# PubChem

> [!NOTE]
> **TODO:** This README still needs to be reviewed and quality checked

## Overview

`pubchem-fetch` resolves a small-molecule identifier (CID, name, SMILES, or InChIKey) against the PubChem PUG REST API
and returns canonical structure data plus optional synonyms. Exactly one identifier is supplied per call; the wrapper
takes care of CID resolution, property fetching, and (optionally) synonym retrieval. This is a CPU-only,
network-bound tool with no model weights or GPU dependencies.

## Background

**What does this tool measure/predict?**
[PubChem](https://pubchem.ncbi.nlm.nih.gov/) is the world's largest open chemistry database, hosted at the National
Center for Biotechnology Information (NCBI). It aggregates compound records (well-defined chemical structures),
substance records (depositor-supplied entries, possibly redundant), and bioassay results. `pubchem-fetch` queries the
*Compound* domain via PUG REST and returns canonical structure descriptors (SMILES, InChI, InChIKey, IUPAC name) along
with computed physicochemical properties (molecular weight, exact mass, TPSA, hydrogen-bond counts, rotatable bond
count, complexity, charge, heavy-atom count) and optionally a list of synonyms.

**Why is this important?**
- Compound resolution: convert a user-supplied name or SMILES into a canonical CID for downstream pipelines
- Cheminformatics: retrieve standardized SMILES/InChI/InChIKey for deduplication and database joins
- Drug discovery: pull descriptor counts (HBD, HBA, TPSA, rotatable bonds) for rule-of-five style filtering
- Annotation: enrich a list of CIDs with molecular formulas, masses, and human-readable names
- Cross-referencing: PubChem CIDs are the canonical anchor for cross-linking to ChEMBL, NCBI, and many other
  chemistry resources

**Scientific foundation:**
PubChem ingests structures from hundreds of depositors and runs them through a standardization pipeline that
canonicalizes structure representations, computes 2D and 3D descriptors, generates standard InChI/InChIKey hashes via
the [IUPAC InChI](https://www.inchi-trust.org/) library, and assigns each unique compound a stable CID. Computed
properties are derived directly from the standardized structure (TPSA via the Ertl algorithm, complexity via
Bertz/Hendrickson/Ihlenfeldt). The PUG REST endpoint exposes both the resolution graph (name/SMILES/InChIKey to CID)
and the property table; the database is updated continuously as new depositions arrive.

## Tools

### PubChem Fetch (`pubchem-fetch`)

Fetch a compound record from PubChem.

Resolves the input identifier to a PubChem CID (skipped when `inputs.cid`
is given directly), then fetches the configured property bundle and
optionally synonyms.

## How It Works

**Method overview:**
The tool wraps the PubChem PUG REST API in two stages:
1. **Identifier resolution:** If `cid` is supplied, the resolver is skipped. Otherwise the wrapper URL-encodes the
   provided `name`, `smiles`, or `inchikey` and calls the matching `/compound/{domain}/{value}/cids/JSON` endpoint to
   obtain a list of matching CIDs. The first CID is used; all matches are returned in `all_matched_cids`.
2. **Property fetch:** The wrapper then calls
   `/compound/cid/{cid}/property/{comma-separated-properties}/JSON` with the configured property bundle and parses the
   response into typed fields. PubChem returns `MolecularWeight` and `ExactMass` as JSON strings (preserving the
   server-formatted precision) and most other numeric descriptors as native JSON numbers; the wrapper coerces both to
   `float` or `int` as appropriate. If `include_synonyms=True`, a second call to `/compound/cid/{cid}/synonyms/JSON`
   retrieves the synonym list (truncated to 50).

A shared `requests` session with retry/backoff handles transient HTTP failures.

**Key assumptions:**
- Network access to `pubchem.ncbi.nlm.nih.gov` is available
- Exactly one of `cid`, `name`, `smiles`, `inchikey` is provided (a model_validator enforces this)
- Standard PubChem identifier formats (CIDs are positive integers; InChIKeys are 27-character hashes with two hyphens)

**Limitations:**
- Compound domain only (no Substance or BioAssay queries; no structure search beyond exact identity)
- Ambiguous names (e.g. "vitamin D") may resolve to many CIDs; the wrapper picks the first and logs a warning
- Property bundle is limited to PubChem's computed property table; bioactivity, vendor, and patent data are out of
  scope
- Rate-limited by PubChem: no more than 5 requests per second, 400 requests per minute, and 300 seconds of running time
  per minute per IP (see [PubChem usage policies](https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access))

**Computational requirements:**
- **Hardware:** CPU only, network access required
- **Runtime:** ~0.5-3 seconds per query (depends on network latency and whether `include_synonyms` is set)
- **Scalability:** Sequential queries; for batch retrieval, loop over identifiers and throttle to stay under 5 req/s

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cid` | `int \| None` | `None` | PubChem Compound Identifier (e.g. `2244` for aspirin). Must be >= 1. |
| `name` | `str \| None` | `None` | Common or systematic name (e.g. `"aspirin"`). URL-encoded automatically. |
| `smiles` | `str \| None` | `None` | SMILES string (e.g. `"CC(=O)Oc1ccccc1C(=O)O"`). URL-encoded automatically. |
| `inchi` | `str \| None` | `None` | Standard InChI string (e.g. `"InChI=1S/C9H8O4/..."`). Sent via POST. |
| `inchikey` | `str \| None` | `None` | Standard InChIKey (e.g. `"BSYNRYMUTXBXSQ-UHFFFAOYSA-N"`). |

**Validation rules:** Exactly one of `cid`, `name`, `smiles`, `inchi`, `inchikey` must be provided. The
model_validator rejects calls supplying zero or more than one identifier.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `properties` | `list[PubChemProperty]` | 16 defaults (see below) | PubChem property names to request. |
| `include_synonyms` | `bool` | `False` | If `True`, fetch the compound's synonyms (one extra HTTP call to `/synonyms/JSON`; up to 50 returned). |
| `include_description` | `bool` | `False` | If `True`, fetch the compound's textual descriptions (one extra HTTP call to `/description/JSON`). |
| `include_aids` | `bool` | `False` | If `True`, fetch the list of BioAssay IDs that tested this compound (one extra HTTP call to `/aids/JSON`; can return thousands of IDs). |

**Default `properties` bundle (16 entries):** `Title`, `MolecularFormula`, `MolecularWeight`, `SMILES`,
`ConnectivitySMILES`, `InChI`, `InChIKey`, `IUPACName`, `ExactMass`, `TPSA`, `Complexity`, `Charge`,
`HBondDonorCount`, `HBondAcceptorCount`, `RotatableBondCount`, `HeavyAtomCount`.

`PubChemProperty` is a `Literal` type covering the full PubChem property table (mass, descriptors, 2D/3D features,
fingerprints). Override `properties` to request a different subset; properties that are not in the default bundle
will appear in `raw_property_record` even if no typed output field exists for them.

## Output Specification

```python
# Return type: PubChemFetchOutput
PubChemFetchOutput(
    cid: int,                              # Resolved PubChem CID
    all_matched_cids: list[int],           # All CIDs returned by the resolver
    title: str | None,                     # Common compound name (e.g. "Aspirin")
    molecular_formula: str | None,         # Hill-system molecular formula
    molecular_weight: float | None,        # Average molecular weight (g/mol)
    smiles: str | None,                    # PubChem canonical SMILES (new "SMILES" property)
    connectivity_smiles: str | None,       # Connectivity-only SMILES (new "ConnectivitySMILES" property)
    inchi: str | None,                     # Standard InChI
    inchikey: str | None,                  # Standard InChIKey
    iupac_name: str | None,                # IUPAC systematic name
    exact_mass: float | None,              # Exact (monoisotopic) mass in Da
    tpsa: float | None,                    # Topological polar surface area
    complexity: int | None,                # Bertz/Hendrickson/Ihlenfeldt complexity
    charge: int | None,                    # Net formal charge
    hbond_donor_count: int | None,         # Number of hydrogen-bond donors
    hbond_acceptor_count: int | None,      # Number of hydrogen-bond acceptors
    rotatable_bond_count: int | None,      # Number of rotatable bonds
    heavy_atom_count: int | None,          # Number of non-hydrogen atoms
    synonyms: list[str],                   # Compound synonyms (empty if include_synonyms=False)
    descriptions: list[str],               # Textual descriptions (empty if include_description=False)
    bioassay_ids: list[int],               # BioAssay IDs that tested this compound (empty if include_aids=False)
    source_url: str,                       # URL of the property request
    raw_property_record: dict[str, Any],   # Complete property record from PubChem
)
```

**Key output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `cid` | `int` | Resolved PubChem CID (>= 1). |
| `all_matched_cids` | `list[int]` | All CIDs returned by the resolver; length 1 for unambiguous queries, longer for ambiguous names. |
| `title` | `str \| None` | Common compound name (e.g. `"Aspirin"`), distinct from the IUPAC systematic name in `iupac_name`. |
| `molecular_formula` | `str \| None` | Molecular formula in Hill order (e.g. `"C9H8O4"` for aspirin). |
| `molecular_weight` | `float \| None` | Average molecular weight in g/mol. PubChem returns this as a string; the wrapper parses it to `float`. |
| `smiles` | `str \| None` | PubChem canonical SMILES with stereochemistry. Pulled from the new `SMILES` property (replaces the retired `IsomericSMILES`). |
| `connectivity_smiles` | `str \| None` | Connectivity-only SMILES, with stereochemistry stripped. Pulled from the new `ConnectivitySMILES` property (replaces the retired `CanonicalSMILES`). |
| `inchi` | `str \| None` | Standard InChI string. |
| `inchikey` | `str \| None` | Standard InChIKey hash (27 characters, two hyphens). |
| `iupac_name` | `str \| None` | IUPAC systematic name. |
| `exact_mass` | `float \| None` | Exact (monoisotopic) mass in Da. |
| `tpsa` | `float \| None` | Topological polar surface area in angstroms-squared. |
| `complexity` | `int \| None` | Bertz/Hendrickson/Ihlenfeldt complexity score. |
| `charge` | `int \| None` | Net formal charge. |
| `hbond_donor_count` | `int \| None` | Number of hydrogen-bond donors. |
| `hbond_acceptor_count` | `int \| None` | Number of hydrogen-bond acceptors. |
| `rotatable_bond_count` | `int \| None` | Number of rotatable bonds. |
| `heavy_atom_count` | `int \| None` | Number of non-hydrogen atoms. |
| `synonyms` | `list[str]` | Up to 50 synonyms; empty list when `include_synonyms=False`. |
| `source_url` | `str` | URL of the PubChem property request (useful for debugging and citation). |
| `raw_property_record` | `dict[str, Any]` | Complete property record from PubChem for advanced access (includes any properties requested beyond the 16 default fields). |

**Supported export formats:** `json`, `csv`

## Interpreting Results

**SMILES naming change:**
PubChem retired the property names `CanonicalSMILES` and `IsomericSMILES` in favor of `SMILES` and
`ConnectivitySMILES`. The wrapper requests the new names and exposes them as `smiles` (the canonical, stereochemistry-
preserving SMILES) and `connectivity_smiles` (the connectivity-only SMILES, with stereochemistry stripped). If you
have legacy code referring to canonical/isomeric SMILES from PubChem, the closest mappings are:
- old `IsomericSMILES` -> new `SMILES` (stereochemistry preserved)
- old `CanonicalSMILES` -> new `ConnectivitySMILES` (stereochemistry stripped)

**Mass interpretation:**
- `molecular_weight` is the *average* molecular weight in g/mol, computed using natural-abundance atomic weights. Use
  this for stoichiometry and bulk-property calculations.
- `exact_mass` is the *monoisotopic* mass in Da, computed using the most abundant isotope of each element. Use this
  when comparing against high-resolution mass-spectrometry peaks.

**Drug-likeness heuristics (Lipinski's Rule of Five):**
Lipinski's rule of five flags compounds likely to have poor oral bioavailability:
- `molecular_weight <= 500` g/mol
- `hbond_donor_count <= 5`
- `hbond_acceptor_count <= 10`
- LogP <= 5 (not provided directly; request `XLogP` via `properties` if needed)

A compound that violates two or more rules is typically flagged as likely to have absorption or permeation problems.
`rotatable_bond_count <= 10` and `tpsa <= 140` (Veber's rules) are commonly applied alongside the rule of five for
oral-bioavailability filtering.

**TPSA interpretation:**
- `tpsa < 90`: typically associated with good blood-brain-barrier penetration
- `tpsa <= 140`: oral bioavailability is plausible
- `tpsa > 140`: poor passive membrane permeability is likely

**Interpreting edge cases:**
- An empty `synonyms` list with `include_synonyms=True` indicates PubChem has no curated synonyms for this CID, not
  an HTTP failure
- Multiple entries in `all_matched_cids` mean the supplied identifier was ambiguous; the wrapper deterministically
  returns the first CID. Pass `cid=` directly when ambiguity matters.
- `complexity` is a relative numeric score, not a unitless quality metric; compare values within a series rather
  than absolute thresholds
- `raw_property_record` contains the unmodified PubChem dictionary; use it to access properties that have no typed
  field (e.g. `Volume3D`, `Fingerprint2D`)

## Quick Start Examples

**Example 1: Fetch by CID**
```python
from proto_tools.tools.database_retrieval import (
    PubChemFetchConfig, PubChemFetchInput, run_pubchem_fetch,
)

# Aspirin
inputs = PubChemFetchInput(cid=2244)
output = run_pubchem_fetch(inputs, PubChemFetchConfig())

print(f"CID: {output.cid}")
print(f"Formula: {output.molecular_formula}")
print(f"MW: {output.molecular_weight} g/mol")
print(f"Exact mass: {output.exact_mass} Da")
print(f"SMILES: {output.smiles}")
print(f"InChIKey: {output.inchikey}")
print(f"IUPAC name: {output.iupac_name}")
```

**Example 2: Resolve by name and pull synonyms**
```python
from proto_tools.tools.database_retrieval import (
    PubChemFetchConfig, PubChemFetchInput, run_pubchem_fetch,
)

# Caffeine, with synonyms
inputs = PubChemFetchInput(name="caffeine")
config = PubChemFetchConfig(include_synonyms=True)
output = run_pubchem_fetch(inputs, config)

print(f"Resolved to CID {output.cid} ({output.molecular_formula})")
print(f"Top synonyms: {output.synonyms[:5]}")
print(f"All matched CIDs: {output.all_matched_cids}")
```

**Example 3: Resolve by SMILES and apply Lipinski filtering**
```python
from proto_tools.tools.database_retrieval import (
    PubChemFetchConfig, PubChemFetchInput, run_pubchem_fetch,
)

# Ibuprofen
inputs = PubChemFetchInput(smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O")
output = run_pubchem_fetch(inputs, PubChemFetchConfig())

violations = 0
if output.molecular_weight is not None and output.molecular_weight > 500:
    violations += 1
if output.hbond_donor_count is not None and output.hbond_donor_count > 5:
    violations += 1
if output.hbond_acceptor_count is not None and output.hbond_acceptor_count > 10:
    violations += 1

print(f"CID {output.cid} ({output.iupac_name})")
print(f"MW={output.molecular_weight}, HBD={output.hbond_donor_count}, HBA={output.hbond_acceptor_count}")
print(f"TPSA={output.tpsa}, rotatable bonds={output.rotatable_bond_count}")
print(f"Lipinski violations: {violations}")
```

**Example 4: Fetch by InChIKey with a custom property bundle**
```python
from proto_tools.tools.database_retrieval import (
    PubChemFetchConfig, PubChemFetchInput, run_pubchem_fetch,
)

# Aspirin via InChIKey, requesting only the descriptors we need
inputs = PubChemFetchInput(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
config = PubChemFetchConfig(
    properties=["MolecularFormula", "MolecularWeight", "SMILES", "XLogP"],
)
output = run_pubchem_fetch(inputs, config)

# XLogP has no typed field; access it via raw_property_record
xlogp = output.raw_property_record.get("XLogP")
print(f"CID {output.cid}: {output.molecular_formula}, MW {output.molecular_weight}, XLogP {xlogp}")
```

**Example 5: Chained workflow -- name -> CID -> SMILES -> CID round-trip (identifier coherence)**

Use this whenever a downstream tool will hand back a SMILES string that you need to map to a canonical compound; the round-trip proves PubChem accepts what we just emitted.

```python
from proto_tools.tools.database_retrieval import (
    PubChemFetchConfig, PubChemFetchInput, run_pubchem_fetch,
)

# 1. Resolve a natural-language name to canonical structure data.
by_name = run_pubchem_fetch(PubChemFetchInput(name="caffeine"), PubChemFetchConfig())
canonical_smiles = by_name.smiles  # "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

# 2. Round-trip: feed PubChem's canonical SMILES back as the identifier.
by_smiles = run_pubchem_fetch(PubChemFetchInput(smiles=canonical_smiles), PubChemFetchConfig())

# 3. Both calls must converge on the same compound.
assert by_smiles.cid == by_name.cid == 2519  # caffeine
assert by_smiles.molecular_formula == by_name.molecular_formula == "C8H10N4O2"

# Same idea for InChIKey:
by_inchikey = run_pubchem_fetch(PubChemFetchInput(inchikey=by_name.inchikey), PubChemFetchConfig())
assert by_inchikey.cid == by_name.cid
```

## Best Practices & Gotchas

**Common mistakes:**
1. **Rate-limit violations:** PubChem enforces three concurrent ceilings per IP: no more than 5 requests per second, 400
   requests per minute, and 300 seconds of running time per minute. For batch lookups, throttle client-side (e.g.
   `time.sleep(0.25)` between calls) or PubChem will return HTTP 503 (`PUGREST.ServerBusy`).
2. **Ambiguous names:** Generic names like `"vitamin D"` resolve to multiple CIDs. The wrapper picks the first CID
   and logs a warning; `all_matched_cids` contains the full list. When ambiguity matters, pass `cid=` directly.
3. **InChIKey case sensitivity:** InChIKeys are 27-character strings with two hyphens (e.g.
   `BSYNRYMUTXBXSQ-UHFFFAOYSA-N`). Case matters; lowercase strings will not resolve.
4. **Confusing `smiles` with `connectivity_smiles`:** `smiles` preserves stereochemistry, `connectivity_smiles` does
   not. Use `smiles` for any application that relies on E/Z or R/S information.

**Tips for optimal results:**
- Set `include_synonyms=False` (the default) when you only need structural data; it skips an HTTP call and is
  noticeably faster for very common compounds (which can have hundreds of synonyms)
- For large lookup batches, prefer CID inputs over name/SMILES/InChIKey: CID resolution is skipped and you save one
  HTTP call per query
- Use `raw_property_record` to access PubChem properties beyond the 16 typed defaults (e.g. `Volume3D`, `PatentCount`,
  `Fingerprint2D`); request them via `properties` and read them out of the dict

**Edge cases to watch for:**
- SMILES strings often contain `/`, `\`, `+`, `#`, `=`, and `[]`; the wrapper URL-encodes the value automatically, so
  you do not need to pre-encode
- Salts and mixtures: PubChem stores some entries as multi-component compounds. Check `all_matched_cids` and
  `connectivity_smiles` to confirm you got the expected component
- Setting `include_synonyms=True` adds one extra HTTP call; for very common compounds (e.g. water, ethanol) this can
  return hundreds of synonyms and is the slowest part of the request
- The `raw_property_record` field contains the unmodified PubChem record for properties not yet exposed as typed
  fields; treat unknown keys as strings unless you know they are numeric

## References

**Primary publication:**
- Kim, S. et al. (2023). "PubChem 2023 update." *Nucleic Acids Research*, 51(D1), D1373-D1380.
  [DOI: 10.1093/nar/gkac956](https://doi.org/10.1093/nar/gkac956)
- Summary: Describes the PubChem database (compounds, substances, bioassays), the standardization pipeline that
  generates canonical structures and computed descriptors, and the PUG REST programmatic interface used by this tool.

**Implementation:**
- PubChem website: [https://pubchem.ncbi.nlm.nih.gov/](https://pubchem.ncbi.nlm.nih.gov/)
- PUG REST documentation: [https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)

## Related Tools

**Tools often used together:**
- **`ncbi-efetch`**: PubChem entries cross-link to NCBI databases (PubMed, Gene, Protein); use `ncbi-efetch` to pull
  the linked literature or biomolecule records once you have a CID.

**Alternative tools (similar function):**
- **`chembl-fetch`** (forthcoming): ChEMBL is the canonical source for curated bioactivity data (assays, targets,
  IC50/EC50 measurements). Use it instead of `pubchem-fetch` when you need binding affinities, mechanism-of-action
  annotations, or target associations rather than structural descriptors.
