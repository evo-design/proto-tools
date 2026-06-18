<a href="https://bio-pro.mintlify.app/tools/sequence-alignment/mmseqs2"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# MMseqs2

![MMseqs2](https://proto-bio.github.io/proto-assets/images/tool/mmseqs2/hero.png)

> [!NOTE]
> **License:** MMseqs2 is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/soedinglab/MMseqs2/blob/master/LICENSE.md) for full terms.

## Overview

[MMseqs2](https://github.com/soedinglab/MMseqs2) (Many-against-Many sequence searching) is an ultra-fast protein and nucleotide sequence search and clustering toolkit from the [Steinegger](https://steineggerlab.com/) and [Söding](https://www.mpinat.mpg.de/soeding) labs. It searches very large databases at speeds substantially beyond BLAST with comparable sensitivity, supports GPU acceleration on recent NVIDIA hardware, and powers the homology-search step of the ColabFold structure-prediction pipeline. The toolkit exposes protein search, nucleotide-genome search, clustering, and ColabFold-style MSA generation as four registered tools.

## Background

MMseqs2 ([Steinegger and Söding, 2017](https://doi.org/10.1038/nbt.3988)) implements sequence-similarity search and clustering through a cascaded prefilter-align approach. Short k-mer matches between the query and the database are located first, surviving candidates are scored with an ungapped extension, and final hits are realigned with gapped [Smith-Waterman](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) alignment. Clustering uses greedy [set cover](https://en.wikipedia.org/wiki/Set_cover_problem) over the alignment graph. The cascade reduces the search space by several orders of magnitude while retaining sensitivity comparable to BLAST, making analyses over databases with billions of sequences tractable on a single workstation.

The GPU build ([Kallenborn et al., 2025](https://doi.org/10.1038/s41592-025-02819-8)) accelerates the prefilter and alignment stages on NVIDIA Turing-generation or newer hardware. On top of the search engine, the ColabFold homology-search pipeline ([Mirdita et al., 2022](https://doi.org/10.1038/s41592-022-01488-1)) iterates MMseqs2 searches against clustered reference databases such as UniRef30 to produce the multiple sequence alignments that AlphaFold-class structure predictors consume. This toolkit exposes that pipeline as `mmseqs2-homology-search` in addition to the more general search and clustering operations.

### Learning Resources

- [soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) (Söding and Steinegger labs). Official repository and the source of the `mmseqs` command-line program that this toolkit invokes.
- [MMseqs2 wiki](https://github.com/soedinglab/MMseqs2/wiki) (Söding and Steinegger labs). The reference wiki for the command-line surface, including the workflow modules that the four registered tools wrap.
- [ColabFold homology-search documentation](https://github.com/sokrypton/ColabFold/wiki) (Steinegger and Ovchinnikov labs). Walks through the iterative MSA pipeline that `mmseqs2-homology-search` runs internally.

## Tools

### MMseqs2 Protein Search (`mmseqs2-search-proteins`)

Performs `mmseqs easy-search` of one or more protein query sequences against either a user-supplied target database or an inline list of target proteins. Returns the alignment hits per query, each with target identifier, percent identity, and E-value. The local execution mode runs on CPU by default and supports an opt-in GPU mode for searches against a prebuilt database with a GPU-padded index.

#### Applications

This tool is appropriate for ad-hoc protein homology search at scale, ranking hits against a custom reference database, identifying functional homologs across a sequenced library, and any analysis in which the BLAST-style hit table is the deliverable. The MMseqs2 sensitivity model finds remote homologs that fall well below the sequence-identity range where standard BLAST searches lose signal.

#### Usage Tips

- **Targets are specified on `Mmseqs2SearchProteinsConfig` via either `mmseqs_db` or `target_sequences`, but not both.** Use `mmseqs_db` for a path to a FASTA file or a prebuilt MMseqs2 database when the target set is large or reused across calls. Use `target_sequences` for short inline lists. They live on Config because every query in a run searches against the same target collection.
- **`mmseqs_db` is cached by path, not by contents.** The per-item cache key includes the path string but not the bytes at that path, so mutating the database in place (overwriting `/dbs/uniref90` with a new build at the same path) will silently return stale hits. If you swap a DB at a stable path, either clear the cache or use a versioned filename (`/dbs/uniref90_v2`) so the new path forces a fresh key.
- **`sensitivity=5.7` is the wrapper default and matches upstream `easy-search`.** Higher values recover more distant homologs at the cost of additional runtime. The accepted range is 1.0 to 7.5.
- **`only_top_hits=True` (the default) returns only the best hit per query by percent identity.** Set it to `False` to retain every hit that passes the configured thresholds.
- **`use_gpu=True` requires a GPU-padded index alongside the target database.** Build the index once with `mmseqs makepaddedseqdb <db> <db>.idx_pad`. The configuration validator hard-errors when the `.idx_pad` companion is missing or when `use_gpu=True` is combined with inline `target_sequences` (the GPU path does not accept inline targets).
- **`extra_args` accepts verbatim [`mmseqs easy-search`](https://github.com/soedinglab/MMseqs2/wiki#easy-search) CLI tokens.** Pass any flag not exposed as a typed field through this list (for example `["--alignment-mode", "3"]`). Tokens are appended after the typed flags.

### MMseqs2 Genome Search (`mmseqs2-search-genomes`)

Performs the full MMseqs2 nucleotide search pipeline against either a user-supplied target database or an inline list of target genomes. The tool builds query and target databases with `createdb`, runs `search`, and converts the result to the BLAST-style tabular schema with `convertalis`. Runs on CPU only.

#### Applications

This tool is appropriate for genome-to-genome similarity analysis, locating homologous regions between assembled genomes, comparative genomics over closely related strains, and any nucleotide analog of the protein-search workflow.

#### Usage Tips

- **Targets are specified on `Mmseqs2SearchGenomesConfig` via either `target_db` or `target_genomes`, but not both.** Use `target_db` for a FASTA file or a prebuilt MMseqs2 database; use `target_genomes` for inline nucleotide sequences. They live on Config because every query in a run scans against the same target collection.
- **`target_db` is cached by path, not by contents.** The per-item cache key includes the path string but not the bytes at that path, so mutating the database in place will silently return stale hits. If you swap a DB at a stable path, either clear the cache or use a versioned filename so the new path forces a fresh key.
- **`sensitivity=7.5` is the wrapper default for nucleotide search.** This is a wrapper bias above the upstream MMseqs2 default of 5.7, chosen because nucleotide searches typically benefit from the higher sensitivity setting. The accepted range is 1.0 to 7.5.
- **`strand=2` (both strands) is the wrapper default.** Upstream defaults to forward strand only. Set `strand=1` to restrict to the forward strand or `strand=0` for reverse only.
- **`extra_args` accepts verbatim [`mmseqs search`](https://github.com/soedinglab/MMseqs2/wiki#search-workflow) CLI tokens.** Tokens are appended after the typed flags.

### MMseqs2 Clustering (`mmseqs2-clustering`)

Performs `mmseqs cluster` over an inline list of sequences or a prebuilt MMseqs2 database and returns per-sequence cluster assignments. Each result records the cluster identifier and whether the sequence is the cluster representative. Runs on CPU only.

#### Applications

This tool is appropriate for deduplicating a sequence set before downstream analysis, partitioning a protein library into functional families, selecting representative sequences from a redundant collection, and any analysis that benefits from a similarity-based grouping of sequences.

#### Usage Tips

- **Inputs are specified via either `input_sequences` or `mmseqs_db`, but not both.** Use `input_sequences` for inline sequences; use `mmseqs_db` for a prebuilt database that may be reused across calls.
- **`mmseqs_db` is cached by path, not by contents.** The cache key includes the path string but not the bytes at that path, so mutating the database in place will silently return stale cluster assignments. If you swap a DB at a stable path, either clear the cache or use a versioned filename so the new path forces a fresh key.
- **`min_seq_id=0.6` is the wrapper default.** This is a wrapper bias above the upstream MMseqs2 default of 0.0, chosen as a reasonable starting point for grouping proteins into functional families. Set it higher (for example `0.95`) to remove near-duplicates, or lower (for example `0.3`) to group remote homologs.
- **`cluster_mode=0` (set-cover) is the default greedy algorithm.** Alternative modes are `1` (connected-component, BLASTclust-style) and `2` or `3` (greedy by length, CD-HIT-style).
- **The cluster representative is the first sequence to cover the cluster during greedy set-cover.** It is not necessarily the longest or most central sequence. Choose an alternative `cluster_mode` if a different representative-selection policy is needed.
- **`extra_args` accepts verbatim [`mmseqs cluster`](https://github.com/soedinglab/MMseqs2/wiki#clustering) CLI tokens.** Tokens are appended after the typed flags.

### MMseqs2 Homology Search (`mmseqs2-homology-search`)

Generates a multiple sequence alignment per query protein using the ColabFold homology-search pipeline. Returns one `MSA` object per query, suitable as the MSA input to AlphaFold-class structure predictors. By default (`search_mode="remote"`) it queries the hosted ColabFold MSA API over the network and needs no local database; set `search_mode="local"` to run MMseqs2 against a registry-provisioned reference database on disk (GPU-accelerated by default on supported hardware).

#### Applications

This tool is the proto-tools entry point for generating the MSA input to structure-prediction tools. It also drives coevolutionary analyses that identify covarying residue pairs as candidate spatial contacts, conservation analyses that highlight functionally important residues, and homolog mining for protein engineering and design pipelines.

#### Usage Tips

- **The `dataset` field selects one registered reference database.** The default is `uniref30-2302`. It is a scalar enum of the searchable ColabFold-style protein databases; non-searchable or non-protein datasets are rejected by validation.
- **`search_mode="remote"` is the default.** It queries the hosted ColabFold MSA API over the network; `dataset`, `use_gpu`, and `sensitivity` are ignored, and no local database or GPU is required. Set `search_mode="local"` to run MMseqs2 against a provisioned on-disk database instead.
- **Local mode (`search_mode="local"`) uses GPU by default.** The configuration validator hard-errors only when `use_gpu=True` on a local search on macOS or Windows (GPU search is Linux-only). Set `use_gpu=False` to force the CPU pipeline.
- **Local mode requires the reference database to be provisioned once before the first call.** Run `python -m proto_tools.tools.sequence_alignment.mmseqs2.setup_databases <dataset>`, where the dataset key matches the value of `Mmseqs2HomologySearchConfig.dataset`. The wrapper does not auto-download databases at call time. (Remote mode skips this entirely.)
- **Local search needs enough RAM to hold the dataset's sequence database.** When available memory (cgroup-aware) is below that file's size, the tool logs a warning and falls back to a disk-paged (mmap) search that completes but is much slower; for a responsive search, allocate more memory or use `search_mode="remote"`.
- **Each query produces an `MSA` object or `None`.** Always check `result.msas[i] is not None` before accessing alignment properties. The `num_homologs_found` list returns `0` for queries that produced no homologs. `MSA` objects serialise to A3M or FASTA through the standard export interface.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every MMseqs2 tool in this toolkit (`mmseqs2-search-proteins`, `mmseqs2-search-genomes`, `mmseqs2-clustering`, `mmseqs2-homology-search`).

- **All four tools share a single MMseqs2 installation.** The local installation downloads the GPU-capable MMseqs2 build, which is a strict superset of the CPU-only build and runs CPU subcommands without enabling GPU code paths.
