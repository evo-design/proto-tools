<a href="https://bio-pro.mintlify.app/tools/database-retrieval/ccd-lookup"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# CCD Lookup

> [!NOTE]
> **License:** CCD Lookup retrieves data from the wwPDB Chemical Component Dictionary, distributed under CC0-1.0 (public domain; no attribution required). The client wrapper code is Apache-2.0-licensed. Please refer to [the data terms](https://www.wwpdb.org/about/usage-policies) for full terms.

## Overview

CCD Lookup wraps [`pdbeccdutils`](https://github.com/PDBeurope/ccdutils), the PDBe Python library for the [wwPDB Chemical Component Dictionary (CCD)](https://www.wwpdb.org/data/ccd). Given a CCD code (e.g. `ATP`) or a SMILES string, the `ccd-lookup` tool returns a `Ligands` collection of standard `Fragment` objects, plus a parallel list of `CcdEnrichment` records carrying formula, descriptors, parent component, [RDKit](https://www.rdkit.org/) physicochemical properties, optional UniChem cross-references, and PDB structures using the ligand. It runs on CPU and is fully offline by default.

## Background

The [wwPDB Chemical Component Dictionary (CCD)](https://www.wwpdb.org/data/ccd) is the dictionary of every chemical component observed in the [Protein Data Bank](https://www.rcsb.org/): small-molecule ligands, modified amino acids, ions, cofactors, nucleotides, and sugars. Each component has a 1- to 5-character identifier (for example `ATP` for adenosine triphosphate, `HEM` for heme, `MG` for magnesium ion, `SEP` for phosphoserine), and each entry stores atoms, bonds, formula, IUPAC name, descriptors (SMILES / InChI / InChIKey), release status, and, for modified residues, a parent component (for example `SEP` to `SER`).

Cross-references via [UniChem](https://www.ebi.ac.uk/unichem/) link CCD entries to external chemistry databases ([DrugBank](https://go.drugbank.com/), [ChEMBL](https://www.ebi.ac.uk/chembl/), [PubChem](https://pubchem.ncbi.nlm.nih.gov/), [ChEBI](https://www.ebi.ac.uk/chebi/)), so the same molecule can be looked up across resources. The tool reads a bundled copy of the CCD `components.cif` ([Kunnakkattu et al., 2023](https://doi.org/10.1186/s13321-023-00786-w)) loaded via [`pdbeccdutils.core.ccd_reader`](https://pdbeurope.github.io/ccdutils/). SMILES inputs are canonicalized with [RDKit](https://www.rdkit.org/) and matched against an index built over the bundled dictionary by canonical SMILES and InChIKey. Records and their provenance come directly from the wwPDB Chemical Component Dictionary, distributed by [PDBe](https://www.pdbe.org/).

### Learning Resources

- [PDBe CCDUtils documentation](https://pdbeurope.github.io/ccdutils/) (PDBe) - official documentation for the underlying library, covering the CCD reader, descriptors, and depiction.
- [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd) (wwPDB) - the reference description of the CCD, its identifiers, and what each entry stores.
- [UniChem](https://www.ebi.ac.uk/unichem/) (EMBL-EBI) - the cross-reference service used to map CCD entries to external chemistry databases.

## Tools

### CCD Lookup (`ccd-lookup`)

Enriches wwPDB Chemical Component Dictionary entries. Accepts CCD codes (such as `"ATP"`) or SMILES strings, in mixed batches, and returns a `CcdLookupOutput` containing a `Ligands` collection plus parallel `CcdEnrichment` records: formula, descriptors, parent component, RDKit physicochemical properties, optional UniChem cross-references, and optional PDB usage.

#### Applications

Use this for user-facing enrichment workflows such as notebooks, scripts, dashboards, and ligand reports. Resolve a CCD code or SMILES to a canonical `Fragment` with formula, descriptors, and physicochemical properties before structure prediction or docking, map a ligand to external chemistry databases via UniChem, or discover which experimental structures contain a ligand before structure-based work. The returned `Ligands` collection feeds directly into tools that take ligands as input, and PDB identifiers from `pdb_structures` pair naturally with the [PDB](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) tool.

#### Usage Tips

- **For per-fragment SMILES-to-CCD lookups, use `proto_tools.entities.ligands.ccd_utils.map_smiles_to_ccd_code` instead.** It runs the same lookup in the current Python process, without the subprocess startup this tool incurs, so when you are not using the persistent tool context it can be much faster.
- **A SMILES with no CCD match returns `ccd_code=None` rather than an error.** Check `enrichment.ccd_code is not None` before treating an entry as found. `result.num_unresolved` gives the batch-level count.
- **`parent_ccd_code` is populated only when the CCD entry declares a canonical parent component.** Modified residues like `SEP` (phosphoserine, parent `SER`), `MSE` (selenomethionine, parent `MET`), or `PTR` (phosphotyrosine, parent `TYR`) carry a parent code. Most small-molecule ligands have no parent and return `None`.
- **`pdb_structures` can be very large.** For common cofactors and ions (`HEM`, `ATP`, `NAG`, `MG`, `ZN`) the list runs to many thousands of PDB IDs. It may be helpful to process it in chunks rather than loading every entry at once when you pass it to a downstream step.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every CCD Lookup tool in this toolkit (`ccd-lookup`).

- **Offline by default.** The tool reads a bundled copy of the wwPDB CCD and runs fully offline. Only `include_cross_references` (UniChem) and `include_pdb_usage` (RCSB) require network access, and both default to off.
- **One-time data download.** First use downloads the roughly 70 MB compressed CCD bundle (`components.cif.gz`) to `$PROTO_MODEL_CACHE/ccd_lookup/` and decompresses it to a `components.cif` of roughly 500 MB, which grows as the dictionary grows. Subsequent runs reuse the decompressed file.
