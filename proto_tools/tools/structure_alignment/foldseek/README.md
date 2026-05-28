<a href="https://bio-pro.mintlify.app/tools/structure-alignment/foldseek"><img align="right" src="https://img.shields.io/badge/View_Docs-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="View Docs"></a><a href="examples/example.ipynb"><img align="right" src="https://img.shields.io/badge/Example_Notebook-2e7d32?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yIDNoNmE0IDQgMCAwIDEgNCA0djE0YTMgMyAwIDAgMC0zLTNIMnoiLz48cGF0aCBkPSJNMjIgM2gtNmE0IDQgMCAwIDAtNCA0djE0YTMgMyAwIDAgMSAzLTNoN3oiLz48L3N2Zz4=" alt="Example Notebook"></a>

# Foldseek

![Foldseek](https://raw.githubusercontent.com/steineggerlab/foldseek/master/.github/foldseek.png)

> *Image source: [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek)*

> [!NOTE]
> **License:** Foldseek has a GPL-3.0 license. Please refer to [the license](https://github.com/steineggerlab/foldseek/blob/master/LICENSE.md) for full terms.

## Overview

[Foldseek](https://github.com/steineggerlab/foldseek) is a structural search and alignment tool from the [Steinegger Lab](https://steineggerlab.com/) at Seoul National University. It encodes each residue of a protein structure as a discrete letter over a learned structural alphabet, then performs sensitive alignment over those letter sequences alongside the underlying amino acids. The result is structural homology search at sequence-search speed, together with complementary clustering, multimer-search, and reciprocal-best-hits operations.

## Background

Foldseek ([van Kempen et al., 2024](https://doi.org/10.1038/s41587-023-01773-0)) performs structural homology search, identifying distant evolutionary relatives of a query protein by structural similarity rather than sequence similarity. Each residue of a protein structure is represented as a discrete letter over a learned structural alphabet (the 3Di alphabet) that captures the tertiary interactions between that residue and its spatial neighbours. Pairs of structures are then aligned by running MMseqs2-style sensitive sequence alignment over the 3Di strings together with the underlying amino-acid sequences. The original publication reports that this approach decreases computation times by four to five orders of magnitude relative to the established structural aligners Dali, TM-align, and CE. Foldseek can also accept amino-acid sequences directly, in which case the bundled [ProstT5](https://github.com/mheinzinger/ProstT5) language model predicts a 3Di sequence before alignment.

Foldseek-Multimer ([Kim et al., 2025](https://doi.org/10.1038/s41592-025-02593-7)) extends the same machinery to multi-chain complexes. It computes pairwise chain-to-chain alignments and then clusters their superposition vectors to identify mutually compatible chain pairs. The multimer publication reports speedups of three to four orders of magnitude over the gold-standard multimer aligner while producing comparable alignments, and demonstrates that the method aligns billions of complex pairs within 11 hours of compute. The Foldseek codebase is released as open source by the [Steinegger Lab](https://steineggerlab.com/) at [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek), and the same group operates a public web service at [search.foldseek.com](https://search.foldseek.com) that the remote execution modes of this toolkit target.

### Learning Resources

- [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) (Steinegger Lab, Seoul National University). Official repository and command-line interface for `easy-search`, `easy-cluster`, `easy-multimersearch`, `easy-multimercluster`, and `easy-rbh`.
- [search.foldseek.com](https://search.foldseek.com) (Steinegger Lab). The public web service that the remote execution mode targets.

## Tools

### Foldseek Search (`foldseek-search`)

Aligns a single-chain query structure against one or more reference databases and returns a ranked list of structural hits. The remote execution mode submits the query to the Steinegger Lab web service and downloads the result archive. The local execution mode runs `foldseek easy-search` against a user-supplied target database.

#### Applications

This tool is the structural analogue of BLAST. It is the appropriate first step for detecting distant homologues that fall below the sequence-similarity twilight zone (commonly cited as below 30 percent pairwise identity), for finding structural templates against the [AlphaFold Database](https://alphafold.ebi.ac.uk/) when no experimental structures are available for a target, and for assessing whether a designed protein recapitulates a known fold or represents a novel topology.

#### Usage Tips

- **The remote service is the default execution mode and provides a hosted set of reference databases.** Selectable databases are `pdb100`, `afdb50`, `afdb-swissprot`, `afdb-proteome`, `mgnify_esm30`, `gmgcl_id`, `BFVD`, `cath50`, and `bfmd`. The remote default queries `pdb100` and `afdb50`. Override the selection through the `databases` configuration field.
- **The alignment algorithm is selected by `mode` in remote execution and by `alignment_type` in local execution.** For `mode`, the default `3diaa` performs 3Di-plus-amino-acid local alignment, `tmalign` runs the global TM-align, and `lolalign` runs the LoL-aligner local alignment. The local-mode equivalent `alignment_type` takes the integer values `0` (3Di), `1` (TM-align), `2` (3Di+AA, the default), and `3` (LoL).
- **Local execution requires a target database.** Provide either a prebuilt Foldseek database or a directory of PDB files via the `local_db` configuration field. Foldseek constructs a temporary database from a directory of files at runtime, but a prebuilt database from `foldseek createdb` is more efficient for repeated queries.
- **`sensitivity` controls the prefilter stage during local execution.** Higher values recover more distant homologues at the cost of additional runtime. The wrapper default of 9.5 matches the upstream `--sensitivity` default.
- **Local execution can be GPU-accelerated.** Set `use_gpu=True` to run with `--gpu 1` on a compatible NVIDIA GPU host (see Toolkit Notes for requirements).

### Foldseek Cluster (`foldseek-cluster`)

Groups a set of structures into clusters by 3Di structural similarity using `foldseek easy-cluster`. Inputs can be structure text (PDB or mmCIF) or amino-acid sequences (FASTA). The latter are routed through the bundled ProstT5 language model, which predicts a 3Di sequence per input before clustering proceeds.

#### Applications

This tool is appropriate for deduplicating a set of designed structures before downstream analysis, for surveying fold families across a screened library, and for partitioning a large structure collection into representative groups for further inspection. Clusters with a single member identify structurally isolated entries that share no near-neighbour in the input set.

#### Usage Tips

- **`structures` accepts either a list or a directory path.** Provide an in-memory list of structure or FASTA text strings (Structure objects and file paths are also accepted per item), or a single path to a directory of supported files, in which case filename stems become the structure identifiers.
- **A single call must use one input format.** Mixing FASTA inputs with PDB or mmCIF inputs is rejected by input validation. Format is auto-detected per input entry.
- **`min_seq_id=0.0` is intentional and lets 3Di structural similarity dominate cluster assignment.** Raising it adds a sequence-identity floor to cluster membership. Use a non-zero value only when a sequence-similarity constraint is desired alongside structural similarity.
- **There is no parameter that requests an exact cluster count.** Foldseek clusters by similarity threshold, not by a target count. To approximate a target number of clusters, sweep the `cov` field and select the run whose cluster count is closest to the target.

### Foldseek Multimer Search (`foldseek-multimer-search`)

Aligns a multi-chain query complex against multimer-aware reference databases using the same execution-mode pattern as `foldseek-search`. The remote service hosts the multimer endpoint, and the local execution mode runs `foldseek easy-multimersearch` against a user-supplied target database.

#### Applications

This tool ranks reference complexes by structural similarity to a query complex. It is appropriate for finding natural complexes that resemble a designed binder-target pose, for identifying multi-chain assemblies that share interface architecture with a query, and for mining experimentally determined complexes that match a hypothesised binding mode. Sequence-only methods cannot perform the equivalent search because chain compatibility is governed by tertiary contacts rather than sequence similarity.

#### Usage Tips

- **The default remote database is `pdb100`.** Override through the `databases` configuration field with any of the values in the database list documented under `foldseek-search`.
- **The `mode` value is sent to the remote endpoint with a `complex-` prefix internally.** Configure the field as plain `3diaa`, `tmalign`, or `lolalign`. The toolkit applies the multimer wire-format prefix during submission.
- **Local execution requires a target database via `local_db`.** As with single-chain search, either a prebuilt Foldseek database or a directory of multimer files is accepted.

### Foldseek Multimer Cluster (`foldseek-multimercluster`)

Groups a set of multi-chain assemblies into clusters using `foldseek easy-multimercluster`, which combines per-chain TM-score and interface lDDT into a multimer-level similarity score. Inputs are multi-chain PDB or mmCIF text.

#### Applications

This tool is appropriate for partitioning a candidate set of designed complexes by overall complex geometry, for selecting structurally diverse representatives from a larger pool of binder-target poses, and for analysing the structural diversity of an experimentally determined complex collection.

#### Usage Tips

- **Structure identifiers must not contain an underscore.** Foldseek emits cluster member identifiers as `{multimer_id}_{chain}`, so an underscore in the multimer identifier would silently corrupt downstream parsing. Both user-supplied and filename-derived identifiers are validated and rejected if they contain an underscore.
- **Three thresholds control cluster membership.** `multimer_tm_threshold` (default `0.65`) sets the multimer-level TM-score required for inclusion. `chain_tm_threshold` (default `0.001`) governs the per-chain TM-score required during chain-pair filtering. `interface_lddt_threshold` (default `0.5`) sets the interface quality required for a chain-pair alignment to contribute to the multimer score.

### Foldseek Reciprocal Best Hits (`foldseek-rbh`)

Performs a reciprocal-best-hits structural search between a single-chain query and a target database using `foldseek easy-rbh`. Only mutual best matches are returned, in contrast to the all-hit output of `foldseek-search`.

#### Applications

This tool produces conservative one-to-one structural correspondences. It is appropriate for structural orthology calls between species, for mapping designed proteins to their closest natural counterpart in a curated reference set, and for any analysis in which the absence of a reciprocal best match should be interpreted as no confident correspondence.

#### Usage Tips

- **This tool runs only in local execution mode.** No remote endpoint exists for reciprocal best hits, and a `local_db` value pointing at a prebuilt database or a directory of PDB files is required.
- **The output is sparse by construction.** Most queries return zero or one hit, and the absence of a reciprocal best match indicates that no target in the database satisfies the reciprocity criterion.

## Toolkit Notes

<a href="https://bio-pro.mintlify.app/tools/guides/tool-persistence"><img src="https://img.shields.io/badge/Tool_Persistence_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Tool Persistence guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/device-management"><img src="https://img.shields.io/badge/Device_Management_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Device Management guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/parallel-execution"><img src="https://img.shields.io/badge/Parallel_Execution_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Parallel Execution guide"></a> <a href="https://bio-pro.mintlify.app/tools/guides/cloud-inference"><img src="https://img.shields.io/badge/Cloud_Inference_→-046e7a?style=flat-square&logo=readthedocs&logoColor=white" alt="Cloud Inference guide"></a>

These apply to every Foldseek tool in this toolkit (`foldseek-search`, `foldseek-cluster`, `foldseek-multimer-search`, `foldseek-multimercluster`, `foldseek-rbh`).

- **Local memory consumption scales linearly with database size.** The upstream documentation gives a per-residue cost of `(6 + 1 + 1) bytes × num_residues` for Cα coordinates, 3Di letters, and amino-acid letters, and reports that the 54 million entries in AFDB50 require approximately 151 GB of RAM under default settings.
- **Hits use a 12-column M8 tabular schema with `sequence_identity` normalised to the range 0 to 1.** Filtering structural hits by sequence identity defeats the purpose of structural search, since distant homologues commonly share fold without sharing sequence. `evalue` and `bit_score` are the appropriate ranking criteria.
- **Accepted input formats differ by tool.** `foldseek-search`, `foldseek-multimer-search`, and `foldseek-rbh` currently accept only raw PDB text, `foldseek-cluster` accepts PDB, mmCIF, or FASTA, and `foldseek-multimercluster` accepts PDB or mmCIF.
- **Local execution requires a user-supplied target.** Either a prebuilt Foldseek database or a directory of structure files must be provided through the `local_db` field. No reference database is bundled with the toolkit.
- **A directory passed to `structures` caches by file content, not directory path.** Modifying files in place between calls correctly invalidates the cache, so structure-set updates do not produce stale results.
- **Local search can use an NVIDIA GPU.** Set `use_gpu=True` on any local-mode tool; the GPU build auto-installs on Linux x86_64 hosts with a compatible NVIDIA driver (`>= 525.60.13`).
