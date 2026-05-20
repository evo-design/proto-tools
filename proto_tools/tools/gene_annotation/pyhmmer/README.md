<a href="https://bio-pro.mintlify.app/tools/gene-annotation/pyhmmer"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a><img align="right" src="https://img.shields.io/badge/Use_on_Proto-coming_soon-6c5ce7?style=flat-square&labelColor=6c5ce7&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4=&logoColor=white" alt="Use on Proto (coming soon)">

# PyHMMER

> [!NOTE]
> **License:** PyHMMER is open source and free for academic and commercial use under an MIT license. Please refer to [the license](https://github.com/althonos/pyhmmer/blob/master/COPYING) for full terms.

## Overview

[PyHMMER](https://github.com/althonos/pyhmmer) is a Python library that binds [HMMER3](http://hmmer.org/) for [profile hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) sequence search and domain annotation. It exposes the five canonical HMMER programs (`hmmsearch`, `hmmscan`, `phmmer`, `nhmmer`, `jackhmmer`) as Python functions, returns structured per-hit and per-domain results, and reaches the sensitivity of HMMER while staying entirely in-process.

## Background

[PyHMMER](https://github.com/althonos/pyhmmer) ([Larralde & Zeller, 2023](https://doi.org/10.1093/bioinformatics/btad214)) is a Cython binding to the HMMER C API that ships the HMMER source itself, so a single `pip install` provides both the Python interface and the compiled search engine. The underlying [HMMER3](http://hmmer.org/) algorithm ([Eddy, 2011](https://doi.org/10.1371/journal.pcbi.1002195)) builds a profile hidden Markov model from a [multiple sequence alignment](https://en.wikipedia.org/wiki/Multiple_sequence_alignment), where each match state stores position-specific emission probabilities and the transitions between states model insertions and deletions. Search proceeds through a cascade of accelerated filters: a [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)-vectorised multiple-segment Viterbi (MSV) filter, a vectorised Viterbi filter, and a Forward/Backward filter, each tightening the candidate set before the final scored alignment. Each hit carries a database-size-independent bit score together with an [E-value](https://en.wikipedia.org/wiki/E-value) derived from extreme-value-distribution theory. The E-value calibrates the expected number of false-positive hits at that bit score for the database being searched.

Profile HMMs detect homology that pairwise methods such as [BLAST](https://en.wikipedia.org/wiki/BLAST_(biotechnology)) miss because they encode an entire family's position-specific conservation pattern rather than the similarity of two sequences alone. HMMER3 brought profile-HMM search within roughly the runtime envelope of BLAST while keeping that sensitivity advantage. PyHMMER preserves the algorithm exactly and adds Python-native multithreading, in-memory HMM and sequence handles, and structured result objects. Coordinates returned for HMM matches, target alignments, and envelopes are reported as 1-indexed, inclusive intervals to match biological residue selection conventions.

### Learning Resources

- [pyhmmer documentation](https://pyhmmer.readthedocs.io/) (Martin Larralde) - the canonical API reference, with worked examples for every binding and a guide to feeding HMM and sequence files in and out of memory.
- [HMMER User's Guide](http://hmmer.org/) (The Eddy/Rivas Laboratory, Harvard) - reference for the HMMER 3 command-line surface, the MSV/Viterbi/Forward filter cascade, E-value statistics, and the gathering/noise/trusted cutoff system used by Pfam HMMs.
- [Pfam (via InterPro)](https://www.ebi.ac.uk/interpro/) (EMBL-EBI) - the standard curated HMM library that ships gathering, noise, and trusted cutoffs, and the typical target database for `hmmscan` domain annotation.

## Tools

### PyHMMER Profile Search (`pyhmmer-hmmsearch`)

Searches one or more HMM profiles against a set of protein sequences and returns the sequences (and the per-domain alignments within them) that match each profile.

#### Applications

Use this when the question is "which proteins belong to family X." Build or download an HMM for a family of interest, then sweep a proteome, a [metagenome](https://en.wikipedia.org/wiki/Metagenomics), or a designed library to enumerate members and pull out their domain coordinates for downstream filtering or alignment.

#### Usage Tips

- **`bit_cutoffs="gathering"` activates the Pfam-curated thresholds and replaces the E-value filter.** Each Pfam HMM ships a hand-curated gathering (`--cut_ga`) cutoff that defines family membership, together with the auto-derived noise (`--cut_nc`) and trusted (`--cut_tc`) cutoffs that bracket the curated set. Use `"gathering"` for routine Pfam annotation; ad-hoc HMMs without stored cutoffs raise `MissingCutoffs`.
- **The default `evalue_threshold=10.0` is intentionally permissive.** Tighten to `0.001` for confident annotation or `1e-10` for stringent homology detection; loose thresholds are useful only when you plan to post-filter on `included` or `domain_included`.

### PyHMMER HMM Scan (`pyhmmer-hmmscan`)

Searches one or more query protein sequences against an HMM database and returns the profiles that match each query.

#### Applications

Use this when the question is "what does this protein contain." Run a query proteome against [Pfam](https://www.ebi.ac.uk/interpro/) to annotate each protein with its domain architecture, then filter on `domain_included` to keep curated hits.

#### Usage Tips

- **Pick `hmmscan` versus `hmmsearch` by what you are querying with.** `hmmscan` takes sequences as queries and a database of HMMs as the target; `hmmsearch` is the reverse. For one or a few queries against Pfam, `hmmscan` is the natural choice; for one HMM against a large sequence database, `hmmsearch` is much faster.
- **`bit_cutoffs="gathering"` applies here too and is the recommended Pfam annotation default.** As with `hmmsearch`, the cutoff is read from the HMM file and ad-hoc HMMs without stored cutoffs will fail with `MissingCutoffs`.

### PyHMMER Single-Sequence Protein Search (`pyhmmer-phmmer`)

Searches one or more protein query sequences against a target protein database by building a temporary HMM around each query.

#### Applications

Use this for HMM-grade sensitivity when no pre-built profile is available. Typical workflows include finding remote homologs of a newly characterised protein in a reference proteome and running a sequence-based homology pass when the family of interest is too narrow or too new to have a curated HMM.

#### Usage Tips

- **A single-query, single-target search will not converge.** `phmmer` builds the HMM from the query against the target database's residue statistics; a database of one sequence has no background to estimate against. Use `phmmer` with a real target proteome, not a synthetic pair.

### PyHMMER Nucleotide Search (`pyhmmer-nhmmer`)

Searches nucleotide query sequences against a nucleotide target database with the same profile-HMM machinery used for proteins.

#### Applications

Use this to find homologs of transposable elements, non-coding RNAs, regulatory elements, and other nucleotide features that diverge fast enough to slip past direct sequence alignment. Pair it with [Dfam](https://www.dfam.org/) - the curated profile-HMM library of transposable-element families that was co-designed with nhmmer - or with custom-built nucleotide HMMs when annotating genomes and metagenomic contigs.

#### Usage Tips

- **`strand` defaults to `"both"` and searches the forward and reverse-complement strands.** Set `"watson"` to restrict to forward or `"crick"` to restrict to reverse-complement when the orientation of a hit is meaningful (e.g., on annotated coding strands).

### PyHMMER Iterative Protein Search (`pyhmmer-jackhmmer`)

Performs iterative protein-sequence search against a target protein database, rebuilding the HMM from each round's included hits to extend the search outward across remote homologs.

#### Applications

Use this when you need to reach divergent family members that a single-pass `phmmer` would miss, for example when seeding a new family from one characterised representative or expanding a manually curated set to its full evolutionary breadth.

#### Usage Tips

- **`inclusion_evalue_threshold` is the lever that controls iterative drift.** Each iteration rebuilds the HMM from hits that pass the inclusion thresholds (`--incE` / `--incdomE`, defaults `0.01`). A looser inclusion threshold pulls in more sequences per round and increases the risk of pulling in unrelated families; tighten it when iterations start drifting.
- **`max_iterations` defaults to 5 and the search exits early on convergence.** Raising it rarely helps if the search has already converged on a stable set, and a higher cap multiplies runtime on long-running jobs.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every PyHMMER tool in this toolkit (`pyhmmer-hmmsearch`, `pyhmmer-hmmscan`, `pyhmmer-phmmer`, `pyhmmer-nhmmer`, `pyhmmer-jackhmmer`).

- **Runs on CPU with SIMD acceleration.** The HMMER3 filter cascade is SIMD-vectorised on x86 platforms. pyhmmer compiles HMMER from source at install time and inherits whatever instruction sets the build host exposes, with no GPU acceleration to enable.
- **Self-contained after install.** The HMMER C library is compiled into the PyHMMER wheel, so no separate HMMER install or PATH lookup is needed; HMM databases such as Pfam-A still have to be downloaded separately.
- **`num_threads` parallelises within a single search.** Default `0` auto-detects the available cores. Memory scales with HMM database size; Pfam-A (around 20,000 HMMs) needs roughly 2 GB of RAM held resident.
- **Reporting versus inclusion thresholds are independent filters.** `evalue_threshold` / `score_threshold` (and their `domain_*` siblings) control what appears in the output, while `inclusion_evalue_threshold` marks the stricter "trusted" subset via the `included` and `domain_included` flags. `jackhmmer` seeds the next iteration's HMM from the included set, so the inclusion threshold drives iterative behaviour while reporting only affects what is returned.
