<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/colabfold-search"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# ColabFold Search

![ColabFold](https://github.com/sokrypton/ColabFold/raw/main/.github/ColabFold_Marv_Logo.png)

> *Image source: [sokrypton/ColabFold](https://github.com/sokrypton/ColabFold)*

> [!NOTE]
> **License:** ColabFold Search is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/sokrypton/ColabFold/blob/main/LICENSE) for full terms.

> [!NOTE]
> **This tool will be replaced by a near-identical successor in the future.**

## Overview

[ColabFold](https://github.com/sokrypton/ColabFold) is an open-source pipeline that combines fast [MMseqs2](https://github.com/soedinglab/MMseqs2)-based homology search with AlphaFold-class structure prediction. This toolkit exposes the homology-search step of ColabFold, which generates a multiple sequence alignment (MSA) for each input protein sequence by searching reference databases for homologs. The resulting MSAs are the canonical input to structure-prediction tools and to coevolutionary or conservation analyses.

## Background

ColabFold ([Mirdita et al., 2022](https://doi.org/10.1038/s41592-022-01488-1)) is an open-source pipeline that pairs the [MMseqs2](https://github.com/soedinglab/MMseqs2) (Many-against-Many sequence searching) engine with AlphaFold-class structure prediction. The homology-search step uses a three-stage cascade. Short k-mer matches between the query and the database are located first, surviving candidates are scored with an ungapped extension, and final hits are realigned with gapped [Smith-Waterman](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) alignment. The pipeline produces per-query multiple sequence alignments that capture the evolutionary signal that AlphaFold and similar structure-prediction models rely on. Conservation patterns within an MSA reveal residues under structural or functional constraint, and covarying residue pairs identify spatial contacts.

This toolkit exposes the search step in two execution modes. Remote execution targets the public ColabFold MMseqs2 API operated by the upstream developers and requires no local database. Local execution runs the bundled `colabfold_search` command-line tool against a local MMseqs2 database, supporting much higher throughput and optional GPU acceleration. The local database is the [UniRef30](https://www.uniprot.org/help/uniref) clustered reference of UniProt, optionally augmented with a metagenomic environmental database. The local database must be provisioned once on the host machine.

### Learning Resources

- [sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) (Steinegger and Ovchinnikov labs). Official repository and the source of the `colabfold_search` command-line tool, plus the Google Colab notebooks that interactively expose the ColabFold pipeline.
- [ColabFold web service](https://colabfold.com) (Steinegger and Ovchinnikov labs). Hosted entry point to the ColabFold MSA-search and structure-prediction pipeline, useful for a quick browser-based run before scripting against the tool.

## Tools

### ColabFold MSA Search (`colabfold-search`)

Generates a multiple sequence alignment for each input protein sequence by searching reference databases for homologs. Remote execution submits the query to the public ColabFold MMseqs2 API. Local execution runs the bundled `colabfold_search` command-line tool against a local MMseqs2 database. The tool returns one result per query in input order, each carrying a list of per-chain `MSA` objects (one for an unpaired query; row-aligned per-chain MSAs for a paired group) that can be exported to A3M or FASTA. Inputs accept raw sequence strings (one unpaired query each), a nested list of sequences (one taxonomy-paired group), or `ColabfoldSearchQuery` objects.

#### Applications

The most common application is generating the MSA input to a structure-prediction tool such as AlphaFold or its open-source successors. MSAs also drive coevolutionary analyses that identify covarying residue pairs as candidate spatial contacts, conservation analyses that highlight functionally important residues, and homolog mining for protein engineering and design pipelines where the natural sequence neighbourhood of a query is informative.

#### Usage Tips

- **Sequence identifiers must be unique across the input batch.** The input validator rejects duplicate identifiers up front. Identifiers omitted from the input are auto-generated as `seq_<sha256[:10]>` and are guaranteed unique for distinct sequences.
- **Remote execution is the default and is appropriate for small batches.** The public ColabFold MMseqs2 API is rate-limited by the upstream developers. High-throughput or batch workloads should use local execution to avoid being throttled.
- **Local execution requires a `msa_db_dir` pointing at a provisioned MMseqs2 database.** The configuration validator hard-errors when the directory does not exist or does not contain the expected `*.dbtype` file for the configured `database_name`. See the local-database note in Toolkit Notes for the provisioning script.
- **`sensitivity` controls the MMseqs2 prefilter in local CPU execution.** Higher values recover more distant homologs at the cost of additional runtime. Setting `sensitivity` has no effect when `use_gpu=True`, because the GPU path forces an ungapped prefilter.
- **`use_gpu=True` requires Linux and a GPU-padded database.** The validator hard-errors on macOS or Windows, when paired with remote execution, or when the `{database_name}.idx_pad` file is missing from `msa_db_dir`. The padded database is built by the provisioning script described in Toolkit Notes.
- **`use_metagenomic_db=True` deepens the MSA by including environmental sequences but substantially increases search runtime.** Use it only when the standard reference database returns a shallow alignment. Leave it `False` (the default) for routine searches.
- **`result.msa` is `None` when no homologs are detected.** Always check `result.msa is not None` before accessing alignment properties. The `num_homologs_found` property returns `0` in that case.
- **`extra_args` accepts verbatim `colabfold_search` CLI tokens and applies only in local execution.** Pass any CLI flag not exposed as a typed field through this list (for example `["--max-accept", "500"]`). The remote API does not accept arbitrary CLI tokens, so `extra_args` is ignored when `search_mode="remote"` and the configuration validator emits a warning in that case.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every ColabFold Search tool in this toolkit (`colabfold-search`).

- **Local execution requires a one-time UniRef30 database setup on the host machine.** The bundled `setup_databases.sh` script downloads the UniRef30 MMseqs2 database, builds the standard index, and optionally builds the GPU-padded index. The fully indexed database occupies approximately 630 GB of disk space and the download alone is approximately 99 GB. The optional metagenomic environmental database adds approximately 110 GB. The wrapper does not provision the database automatically.
- **Outputs are returned as typed `MSA` objects.** Each `ColabfoldSearchResult` carries an `MSA` object (or `None` when no homologs are found) along with the query identifier. `MSA` objects expose alignment dimensions and column-level conservation statistics, and serialise to A3M or FASTA through `to_a3m_file` and `to_fasta_file`.
