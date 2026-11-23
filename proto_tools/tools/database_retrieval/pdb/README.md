<a href="https://bio-pro.mintlify.app/tools/database-retrieval/pdb"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# PDB

![PDB](https://proto-bio.github.io/proto-assets/images/tool/pdb/hero.png)

> [!NOTE]
> **License:** PDB retrieves data from the RCSB Protein Data Bank, distributed under CC0-1.0 (public domain; no attribution required). The client wrapper code is MIT-licensed. Please refer to [the data terms](https://www.rcsb.org/pages/policies) for full terms.

## Overview

The RCSB Protein Data Bank is a database of experimentally determined three-dimensional structures of proteins, nucleic acids, and their complexes. This toolkit provides four CPU-only tools: `pdb-fetch-entry` retrieves structure metadata (title, experimental method, and resolution), `pdb-fetch-fasta` retrieves chain sequences, `pdb-sequence-search` finds polymer entities whose sequences resemble a query, and `pdb-structure-similarity` searches experimental PDB chains or assemblies with RCSB's learned structure embedding. They require network access.

## Background

The Protein Data Bank ([Berman et al., 2000](https://doi.org/10.1093/nar/28.1.235)) is the single worldwide archive of experimentally determined macromolecular structures, served here through the RCSB PDB. It is operated by the Research Collaboratory for Structural Bioinformatics (RCSB) at Rutgers University and the University of California San Diego, with funding from the National Science Foundation, the National Institutes of Health, and the Department of Energy. Entries are solved by X-ray crystallography, cryo-electron microscopy, nuclear magnetic resonance spectroscopy, and other experimental methods.

The tools call two RCSB HTTP endpoints directly. `pdb-fetch-entry` issues a GET request to the RCSB Data API core entry endpoint (`https://data.rcsb.org/rest/v1/core/entry/{pdb_id}`) and reads the structure title from `struct.title`, the experimental method from the first `exptl` record, and the resolution from `rcsb_entry_info.resolution_combined`, which covers both X-ray and cryo-EM entries. Entries solved by NMR have no resolution value. `pdb-fetch-fasta` requests the FASTA endpoint (`https://www.rcsb.org/fasta/entry/{pdb_id}`), parses each record, extracts the author-assigned chain identifiers from the header, and classifies a sequence as protein when it contains amino-acid letters that do not also occur in nucleotide alphabets. Both tools uppercase the supplied accession, retry transient HTTP failures with backoff, and return an empty result when the accession is not found (HTTP 404). Results reflect the live archive at query time rather than a fixed release snapshot.

`pdb-structure-similarity` calls the RCSB Search API's structure service. The service represents structures with ESM3 residue embeddings followed by a learned aggregator whose cosine similarity was trained to approximate maximum TM-score ([Segura et al., 2026](https://doi.org/10.1093/bioinformatics/btag058)). It is an independent approximate-nearest-neighbor search, not Foldseek or an exact structural alignment. Public PDB references are queried by entry and chain/assembly ID. Custom structures are converted to PDB format and sent to RCSB's public upload service.

`pdb-sequence-search` calls the same Search API's sequence service, which aligns the query against every polymer entity in the archive with MMseqs2 ([Steinegger and Söding, 2017](https://doi.org/10.1038/nbt.3988)) and returns ranked entities with their alignment statistics, typically in under a second. Hits are then annotated with one request to the RCSB Data API GraphQL endpoint (`https://data.rcsb.org/graphql`), which supplies each entity's description, source organism, UniProt accessions, author chain identifiers, and canonical sequence.

### Learning Resources

- [RCSB PDB Data API documentation](https://www.rcsb.org/docs/programmatic-access/web-apis-overview) (RCSB PDB) - reference for the REST endpoints, query syntax, and rate limits.
- [RCSB PDB sequence search](https://www.rcsb.org/docs/search-and-browse/advanced-search/sequence-similarity-search) (RCSB PDB) - MMseqs2 sequence-similarity search semantics, cutoffs, and limits.
- [RCSB PDB 3D similarity search](https://www.rcsb.org/docs/search-and-browse/advanced-search/3d-similarity-search) (RCSB PDB) - structure-embedding search semantics and query modes.
- [PDB-101 training and education](https://pdb101.rcsb.org/) (RCSB PDB) - guided material on PDB data, structure determination methods, and how to interpret entries.

## Tools

### PDB Fetch Entry (`pdb-fetch-entry`)

Retrieves structure metadata for a PDB accession from the RCSB Data API core entry endpoint, returning the structure title, the experimental method, the resolution in angstroms, and the request URL.

#### Applications

Use this to assess whether an experimental structure is suitable as a reference before structure-based design or benchmarking: check the experimental method and resolution, then decide whether to use the entry. It pairs with [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot), whose returned PDB cross-references can be ranked by resolution, and with [PDB Fetch FASTA](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) to pull the chain sequences once a suitable entry is selected.

#### Usage Tips

- **Resolution is absent for some methods.** NMR and fiber-diffraction entries have no resolution value, so `resolution` is `None`. Filter on it before sorting entries by quality.
- **This is metadata only.** The tool returns the title, method, and resolution, not atomic coordinates or a structure file.
- **An unknown accession is not an error.** A missing or obsolete accession returns an empty output rather than raising, so check the populated fields before using the result.

### PDB Fetch FASTA (`pdb-fetch-fasta`)

Retrieves the chain sequences of a PDB entry from the RCSB FASTA endpoint, returning one record per unique sequence with the author-assigned chain identifiers that share it, the FASTA header, the sequence, and a protein/nucleic-acid classification, plus the request URL.

#### Applications

Use this to extract reference sequences from an experimental structure for sequence design, alignment, or comparison against computational predictions. Filter `chains` by `is_protein` to separate protein subunits from nucleic-acid chains in a complex, and deduplicate identical sequences to recover the unique entities of a homo-oligomer. It follows [PDB Fetch Entry](https://bio-pro.mintlify.app/tools/database-retrieval/pdb) once a suitable entry is chosen and consumes PDB identifiers surfaced by [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot).

#### Usage Tips

- **One record can cover several chains.** A single `PdbChain` carries every author-assigned chain identifier that shares its sequence, so a homo-oligomer collapses to one record with multiple `chain_ids`.
- **Protein classification is heuristic.** A chain is called protein only when it contains amino-acid letters absent from nucleotide alphabets. Peptide nucleic acids and other hybrid molecules may be misclassified.
- **An unknown accession is not an error.** A missing or obsolete accession returns an empty `chains` list rather than raising.
- **Exporting to `fasta` writes the original headers.** The `fasta` export emits each record using its stored FASTA header verbatim. `json` and `csv` are also supported, with the `csv` form joining shared chain identifiers with a semicolon.

### PDB Sequence Search (`pdb-sequence-search`)

Searches the PDB's polymer entities for sequences similar to each query using RCSB's MMseqs2 sequence service. A query is a protein, DNA, or RNA sequence of at least 10 residues. Each result lists ranked entity identifiers (`1BM7_1`) with the best alignment's identity, E-value, bit score, aligned ranges, and aligned strings, and, by default, the entity's description, source organism, UniProt accessions, author chain identifiers, and canonical sequence.

#### Applications

Use this to identify which protein a fragment or design comes from, to find experimental structures of a sequence or its homologs before structure-based work, or to check a designed sequence for close experimental neighbors. Because it tolerates mismatches and returns in about a second, it is a fast first pass before a slower whole-database search. Chain identifiers feed directly into [PDB Fetch FASTA](https://bio-pro.mintlify.app/tools/database-retrieval/pdb), and UniProt accessions into [UniProt](https://bio-pro.mintlify.app/tools/database-retrieval/uniprot).

#### Usage Tips

- **Coverage is the PDB, not all of UniProt.** Only sequences with a deposited structure can match. For proteins without one, search UniProt or NCBI with `blast-search` instead.
- **Identity is over the aligned region.** A short fragment can match at 1.0 identity while covering a small part of the entity. Check `query_beg`/`query_end` and `subject_beg`/`subject_end` before treating a hit as the same protein.
- **`chain_ids` are author chain IDs.** They match what `pdb-fetch-fasta` reports, unlike the `label_asym_id` values `pdb-structure-similarity` uses.
- **No match returns an empty result.** RCSB answers HTTP 204 when nothing passes the cutoffs, which comes back as an empty `hits` list with `total_count` 0. Loosen `evalue_cutoff` or `identity_cutoff` if you expected hits.
- **Queries shorter than 10 residues are rejected.** RCSB does not search shorter sequences, so the input fails validation before any request is sent.
- **Short nucleic-acid queries rarely match, even when the sequence is in the PDB.** A 12-nt Dickerson dodecamer, or the same sequence doubled to 24 nt, returns no hits even at `evalue_cutoff=10`, although hundreds of entries contain it. Use DNA or RNA queries of roughly 50 nt or more, or look up short oligonucleotides by PDB ID with `pdb-fetch-fasta`.
- **RNA queries are converted from T to U.** RCSB rejects T in an RNA query, so a sequence copied from a DNA source is converted automatically and the conversion is logged at INFO.
- **Unsupported characters fail validation.** Gaps (`-`), stops (`*`), and letters RCSB cannot search for the chosen `sequence_type` are rejected before any request. This includes `X` in proteins and `N` in nucleic acids: RCSB accepts them, but its sequence server then fails with HTTP 500. Remove or replace them before searching.
- **Computed models are off by default.** `include_computed_models` adds RCSB's computed structure models to the search, but RCSB's sequence server often times out on that index.
- **Turn off `annotate` for large result sets.** Annotation adds one GraphQL request per query. With `annotate=False`, hits carry only identifiers and alignment statistics.
- **Annotation failures are reported, not hidden.** If the GraphQL request returns only errors, the tool raises. If some hits come back without annotation, those hits keep `None` annotation fields and the output's `warnings` lists their entity IDs.

### PDB Structure Similarity (`pdb-structure-similarity`)

Searches the experimental PDB structure index for chains or assemblies similar to each query. A query is either a public PDB chain or assembly, or a custom `Structure`. Each result lists ranked hit identifiers with RCSB's normalized ranking score and the structure service's predicted TM-score.

#### Applications

Use this to find experimental structures related to a design or a known protein: screen de novo structures for close experimental neighbors, identify remote structural analogs, or compare assemblies at whole-complex scale. Pass several queries in one call to search a shortlist. Results come back in input order.

#### Usage Tips

- **This is not Foldseek or an exact TM-align result.** `predicted_tm_score` comes from RCSB's learned embedding model, so treat it as an estimate rather than a measured alignment score.
- **For exact scores, align the retrieved hits yourself.** Fetch hit identifiers with `pdb-fetch-entry` and align them against your query with a `structure_alignment` tool such as `tmalign-alignment` or `usalign-alignment`.
- **`chain_id` is RCSB's `label_asym_id`, not the author chain ID.** The two often match but can differ: in 1EFA, author chains A to C (protein) are label chains C to E, and label chains A and B are DNA. `pdb-fetch-fasta` reports author IDs, so check the label ID on the entry's RCSB page before querying. Chain hit identifiers (`1YCR.A`) use label IDs too.
- **Use `global` for whole-fold similarity and `local` for shared substructures.** Match `target_search_space` to chains (`polymer_entity_instance`) or assemblies (`assembly`).
- **Custom structures leave the local environment.** Their coordinates and embedded sequences are sent to RCSB's public upload service. Public `pdb_id` queries upload nothing.
- **A small result limit can miss hits.** `number_of_candidates` (at most 10,000, RCSB's per-request cap) bounds the approximate search, and `predicted_tm_score_cutoff` filters by score. `total_count` reports how many matches RCSB found in all.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PDB tool in this toolkit (`pdb-fetch-entry`, `pdb-fetch-fasta`, `pdb-sequence-search`, `pdb-structure-similarity`).

- **Requires network access.** The tools call the live RCSB PDB HTTP endpoints. They do not run offline and keep no local copy of the archive.
- **Subject to RCSB rate limits.** RCSB throttles clients that exceed a few requests per second and returns HTTP 429 when the limit is exceeded. Space out high-volume requests, since no account or API key is available to raise the limit.
- **Runs on CPU.** There is no model and no GPU. Latency is dominated by the network round-trip.
