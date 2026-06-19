<a href="https://bio-pro.mintlify.app/tools/database-retrieval/pubchem"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# PubChem

![PubChem](https://proto-bio.github.io/proto-assets/images/tool/pubchem/hero.png)

> [!NOTE]
> **License:** PubChem retrieves data from the PubChem database, in the public domain (U.S. Government public domain). The client wrapper code is MIT-licensed. Please refer to [the data terms](https://www.ncbi.nlm.nih.gov/home/about/policies/) for full terms.

## Overview

[PubChem](https://pubchem.ncbi.nlm.nih.gov/) is a public repository of chemical structures, their computed properties, and bioactivity data, maintained by the [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/). The `pubchem-fetch` tool resolves a single small-molecule identifier (CID, name, SMILES, InChI, or InChIKey) against the PubChem PUG REST API and returns the canonical structure descriptors, computed physicochemical properties, and optionally synonyms, textual descriptions, and BioAssay identifiers. It runs on CPU and requires only network access.

## Background

[PubChem](https://pubchem.ncbi.nlm.nih.gov/) ([Kim et al., 2023](https://doi.org/10.1093/nar/gkac956)) is a freely accessible chemistry resource hosted by [NCBI](https://www.ncbi.nlm.nih.gov/). It aggregates compound records with well-defined chemical structures, depositor-supplied substance records, and bioassay results contributed by hundreds of data sources. Each unique compound is assigned a stable Compound Identifier (CID), and standardized structure representations and computed descriptors are derived from a uniform processing pipeline.

Internally, the tool calls the [PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) endpoint at `https://pubchem.ncbi.nlm.nih.gov/rest/pug`. It first resolves the supplied identifier to one or more CIDs. A name, SMILES, or InChIKey is sent as a URL-encoded GET against the matching `/compound/{domain}/{value}/cids/JSON` endpoint, an InChI is submitted via POST, and a CID skips resolution entirely. It then fetches the configured property bundle from `/compound/cid/{cid}/property/{properties}/JSON`, and optionally retrieves synonyms, descriptions, and BioAssay identifiers through additional endpoints. Results reflect the live database at query time rather than a fixed release snapshot.

### Learning Resources

- [PUG REST documentation](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) (PubChem) - official reference for the request grammar, compound domains, property names, and response formats.
- [Programmatic access](https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access) (PubChem) - overview of the programmatic interfaces and the published usage policies and rate limits.

## Tools

### PubChem Fetch (`pubchem-fetch`)

Resolves a single small-molecule identifier to a PubChem CID and returns the requested property bundle, the full list of matched CIDs, and optionally synonyms, textual descriptions, BioAssay identifiers, the source URL, and the raw property record.

#### Applications

Use this to resolve a ligand to its canonical structure and properties before structure-based or chemical-constraint work: convert a user-supplied name or SMILES into a canonical CID and standardized SMILES/InChI/InChIKey before docking, deduplicate or join compound sets on canonical identifiers, or pull descriptor counts for rule-of-five style filtering. PubChem CIDs anchor cross-references into other chemistry resources. Pair this with [NCBI E-utilities](https://bio-pro.mintlify.app/tools/database-retrieval/ncbi) to pull linked literature or biomolecule records once a CID is resolved.

#### Usage Tips

- **Ambiguous names resolve to multiple CIDs.** A generic name can match many compounds. The tool deterministically selects the first CID and records the full list in `all_matched_cids`. Pass a CID directly when the identity must be exact.
- **Prefer CID inputs for large batches.** Supplying a CID skips the resolution call and reduces the request count per query, which matters under the rate limits.
- **Synonym, description, and BioAssay retrieval each add a request.** Enabling them issues an extra HTTP call, and for common compounds the BioAssay list can return thousands of identifiers.
- **Results track the live database.** The same call can return updated structures or properties as PubChem ingests new depositions. It is not pinned to a release.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PubChem tool in this toolkit (`pubchem-fetch`).

- **Requires network access.** The tool calls the live PubChem PUG REST API. It does not run offline and keeps no local copy of the database.
- **Subject to PUG REST throttling.** PubChem applies dynamic per-user throttling, with limits of no more than 5 requests per second, 400 requests per minute, and 300 seconds of running time per minute. Exceeding them returns HTTP 503.
