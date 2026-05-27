<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/blast"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# BLAST

> [!NOTE]
> **License:** BLAST is licensed under Custom (NCBI BLAST+ public domain). Please refer to [the license](https://www.ncbi.nlm.nih.gov/IEB/ToolBox/CPP_DOC/lxr/source/scripts/projects/blast/LICENSE) for full terms.

## Overview

[BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) (Basic Local Alignment Search Tool) is a sequence-similarity search method maintained by the [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/). It finds regions of local similarity between a query nucleotide or protein sequence and entries in a reference database, returning ranked alignments with statistical significance scores. This toolkit exposes both the public NCBI BLAST web service and the local NCBI BLAST+ command-line distribution under a single Python interface.

## Background

BLAST ([Altschul et al., 1990](https://doi.org/10.1016/S0022-2836(05)80360-2)) performs sequence-similarity search through a heuristic algorithm that approximates the exhaustive [Smith-Waterman](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) local alignment at a fraction of its computational cost. The query is first broken into short fixed-length words, exact word matches are located in the database, and each match is extended in both directions until the running alignment score drops below a threshold. The statistical significance of each surviving alignment is expressed as an E-value derived from the Karlin-Altschul statistics, which represents the number of alignments with at least the observed score that would be expected to occur by chance for a database of the given size.

BLAST supports five program variants that pair query and database types appropriately. `blastn` aligns a nucleotide query against a nucleotide database. `blastp` aligns a protein query against a protein database. `blastx` translates a nucleotide query and aligns the translations against a protein database. `tblastn` aligns a protein query against a database of translated nucleotide sequences. `tblastx` translates both query and database. The toolkit's local execution mode uses the [NCBI BLAST+](https://www.ncbi.nlm.nih.gov/books/NBK279690/) command-line distribution ([Camacho et al., 2009](https://doi.org/10.1186/1471-2105-10-421)), which provides the `blastn`, `blastp`, `blastx`, `tblastn`, `tblastx`, and `makeblastdb` command-line programs that this toolkit invokes. The remote execution mode dispatches to the public [NCBI BLAST web service](https://blast.ncbi.nlm.nih.gov/Blast.cgi) through the QBLAST API.

### Learning Resources

- [NCBI BLAST web service](https://blast.ncbi.nlm.nih.gov/Blast.cgi) (NCBI). The public hosted interface that the remote execution mode targets, useful for an interactive run before scripting against the tool.
- [NCBI BLAST+ User Manual](https://www.ncbi.nlm.nih.gov/books/NBK279690/) (NCBI Bookshelf). The reference manual for the command-line distribution that the local execution mode runs.

## Tools

### BLAST Search (`blast-search`)

Aligns a query sequence against a reference database and returns the resulting hits. The remote execution mode submits the query to the NCBI BLAST web service through the QBLAST API. The local execution mode invokes the appropriate BLAST+ program (`blastn`, `blastp`, `blastx`, `tblastn`, or `tblastx`) against a user-supplied database. The query field accepts either a raw nucleotide or protein sequence string or a path to a FASTA file, and the input form is detected automatically.

#### Applications

This tool is the standard first step in any analysis that begins with an unknown sequence and asks what it resembles. Representative applications include functional annotation of a newly assembled gene through homology to characterised proteins, taxonomic identification of an environmental DNA fragment, off-target screening of a PCR primer or CRISPR guide against a reference genome, and tracing the evolutionary distribution of a gene across species.

#### Usage Tips

- **The `program` field must match the query and database types.** Mismatched combinations return no hits and waste a search. Use `blastn` for nucleotide-against-nucleotide, `blastp` for protein-against-protein, `blastx` for a nucleotide query against a protein database, `tblastn` for a protein query against a nucleotide database, and `tblastx` for translated nucleotide against translated nucleotide.
- **Remote execution targets the NCBI BLAST web service and is limited by NCBI rate limits.** The `database` field selects from the hosted reference databases (`nt`, `nr`, `refseq_rna`, `refseq_protein`, `swissprot`, `pdb`, `pataa`, `patnt`). High-throughput or batch workloads should use local execution to avoid being throttled or blocked by NCBI.
- **Local execution requires a `local_db` value pointing at a prebuilt database.** Build one with `blast-create-db` or download a prebuilt NCBI database. The path is the database stem with no file extension. The configuration validator hard-errors when `local_db` is missing in local mode.
- **`evalue` is the primary parameter controlling sensitivity.** The BLAST+ default of `10.0` is permissive and returns spurious hits. Set it to `1e-5` or stricter to filter out alignments that would occur by chance, or use a higher value when searching for short or divergent matches.
- **`extra_args` accepts verbatim BLAST+ CLI tokens and applies only in local execution.** Pass any CLI flag not exposed as a typed field through this list (for example `["-max_hsps", "1"]`). The remote QBLAST API does not accept arbitrary CLI tokens, so `extra_args` is ignored when `search_mode="online"` and the configuration validator emits a warning in that case.

### Create BLAST Database (`blast-create-db`)

Builds a local BLAST database from a FASTA file using the BLAST+ `makeblastdb` program. The output is a set of indexed files referenced by a common stem path. The stem path is returned as `db_path` and can be passed directly as `local_db` to `blast-search`.

#### Applications

This tool is the prerequisite for any local BLAST workflow that searches against a custom reference set, such as an in-house genome assembly, a curated subset of a public database, or a panel of designed sequences. Building a local database once and reusing it across many queries avoids repeated network traffic to NCBI and gives full control over the reference content.

#### Usage Tips

- **`dbtype` must match the input FASTA type.** Use `"nucl"` for nucleotide sequences and `"prot"` for amino-acid sequences. The configuration validator hard-errors on any other value, and a mismatch against the FASTA content will be caught by `makeblastdb` at runtime.
- **`out_prefix` defaults to the input FASTA stem in the same directory.** Set it explicitly when the database should live in a different location or under a different name.
- **`parse_seqids=True` is required for FASTA identifiers to be addressable.** Enable it when downstream calls need to retrieve sequences by identifier through `blastdbcmd` or when building a taxonomy-aware database. Pair it with `hash_index=True` for faster identifier lookups.
- **`extra_args` accepts verbatim `makeblastdb` CLI tokens.** Use it for niche flags not exposed as typed fields, such as `["-mask_data", "/path/to/mask"]` for premasking input or `["-gi_mask", "..."]` for taxonomy-related options.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every BLAST tool in this toolkit (`blast-search`, `blast-create-db`).

- **Hits use the standard BLAST `-outfmt 6` tabular schema.** Each `BlastHit` carries the twelve canonical fields `qseqid`, `sseqid`, `pident`, `length`, `mismatch`, `gapopen`, `qstart`, `qend`, `sstart`, `send`, `evalue`, and `bitscore`. `pident` is reported on a 0-to-100 scale.
- **The local installation downloads the platform-specific NCBI BLAST+ distribution on first use.** The standalone setup pulls the appropriate NCBI BLAST+ tarball and extracts the `blastn`, `blastp`, `blastx`, `tblastn`, `tblastx`, and `makeblastdb` executables. No reference database is bundled, so local execution requires either a user-built database from `blast-create-db` or a separately downloaded NCBI database.
- **The two tools differ in execution mode.** `blast-search` supports both online (`search_mode="online"`, the default) and local (`search_mode="local"`) execution. `blast-create-db` runs only locally because the NCBI web service does not expose `makeblastdb`.
