<a href="https://bio-pro.mintlify.app/tools/structure-alignment/foldseek"><img align="right" src="https://img.shields.io/badge/View_in_Proto_Docs_→-046e7a?style=for-the-badge&logo=readthedocs&logoColor=white" alt="View in Proto Docs →"></a>

# Foldseek Toolkit

## Overview

The Foldseek toolkit wraps the Steinegger Lab's structural search/alignment binary and the public Foldseek server (`search.foldseek.com`). Three sibling tools, all backed by the same Foldseek codebase:

| Tool key | Operation | Modes |
|---|---|---|
| `foldseek-search` | Single-chain query-vs-DB structural search | remote (server) + local (CLI) |
| `foldseek-cluster` | Cluster a set of structures by structural similarity | local only |
| `foldseek-multimer-search` | Multimer (complex) structural search | remote (server) + local (CLI) |

`foldseek-search` is the canonical tool for "find proteins structurally similar to my query" — the structural analog of BLAST. `foldseek-cluster` groups a user-supplied set of PDBs into structural clusters. `foldseek-multimer-search` searches with a multi-chain assembly (biological complex) against multimer-aware databases.

## Background

**What does this toolkit measure?**
Foldseek encodes each protein structure as a sequence over a structural alphabet (3Di) describing tertiary amino-acid interactions, then runs sensitive sequence alignment over those 3Di sequences combined with amino-acid sequences. The result is structural homology search at sequence-search speed — the Foldseek paper reports computation times reduced by four to five orders of magnitude relative to Dali, TM-align, and CE while retaining 86%, 88%, and 133% of their sensitivities, respectively (van Kempen et al., *Nature Biotechnology*, 2024). Foldseek-Multimer (Kim et al., *Nature Methods*, 2025) extends the pipeline to multi-chain assemblies.

**Why is this important?**
- **Scaffold mining for binder design:** sequence-only search misses many structurally similar proteins below ~30% sequence identity (the "twilight zone"). Foldseek surfaces those structural homologs.
- **Template discovery for hard targets:** when UniProt's `pdb_crossrefs` is empty, Foldseek against AFDB50 finds structurally similar predicted structures across all of UniProt.
- **Design novelty validation:** "is my designed protein a known fold or genuinely novel?" — Foldseek against PDB100 is the standard QC step.
- **Multimer / complex search:** ranking complexes by structural similarity to a target binding pose is impossible with sequence-only methods.
- **Structural deduplication:** clustering eliminates near-identical structures from a candidate set before downstream design or analysis.

**Scientific foundation:**
Per Foldseek's GitHub README, "many of Foldseek's modules (subprograms) rely on MMseqs2." Foldseek inherits MMseqs2's sequence-search infrastructure to align over the 3Di structural alphabet. The default `3diaa` mode performs local Gotoh-Smith-Waterman alignment combining 3Di and amino-acid scores; `tmalign` runs full global TMalign on hit candidates; `lolalign` is the local LoL-aligner. Multimer search internally aligns each query chain, then scores chain-pair compatibility against candidate complexes.

## How It Works

**Remote modes (`foldseek-search`, `foldseek-multimer-search` with `search_mode="remote"`):**
1. POST the query PDB + database list to `search.foldseek.com/api/ticket` (multipart/form-data). For multimer search the wrapper wraps the alignment mode as `complex-{mode}` per the server's wire protocol.
2. Poll `/api/ticket/{id}` every `poll_interval_seconds` until status reaches `COMPLETE` (or `ERROR`, or timeout).
3. Download the `.tar.gz` archive from `/api/result/download/{id}` — one `alis_{db}.m8` file per queried database.
4. Parse each M8 file (standard 12-column BLAST tabular format) into typed `FoldseekHit` objects.

**Local modes (`foldseek-search`/`foldseek-multimer-search` with `search_mode="local"`, `foldseek-cluster`):**
1. Provision the Foldseek binary via the standalone env (`standalone/setup.sh` calls `proto_tools/utils/install_binary.py foldseek`, which downloads the platform-specific tarball from `mmseqs.com/foldseek` and extracts the binary into the venv's `bin/` directory).
2. The wrapper writes inputs to a temp dir, invokes `foldseek easy-search` / `easy-cluster` / `easy-multimersearch` via `ToolInstance.dispatch`, parses the M8 (or cluster TSV) output.

**Key assumptions:**
- Query structures are PDB-format text (single-chain for search/cluster, multi-chain for multimer).
- Remote modes require network reachability of `search.foldseek.com` (anonymous access).
- Local modes require the standalone env to have been provisioned (`setup.sh`).

**Limitations:**
- The server's documented `/api/result/{id}` endpoint has a known 404 bug ([Foldseek issue #380](https://github.com/steineggerlab/foldseek/issues/380)); the wrapper uses `/api/result/download/{id}` instead.
- Local modes need a pre-built Foldseek database (use `foldseek createdb` outside this wrapper).
- No CIF input — PDB-format text only.
- No documented public-server rate limit; for large batch sweeps prefer local mode.

**Computational requirements:**
- **Remote modes:** HTTP-only on the wrapper side; CPU only. ~30-90s for a typical small protein against `pdb100`.
- **Local modes:** local CPU, scales with database size + threads. The Foldseek binary is ~50 MB; database memory cost is roughly `(6 + 1 + 1) bytes × num_residues` (e.g. ~150 GB for AFDB50's 54M entries — keep target DB sizes modest for testing).

## Input Parameters

### `FoldseekSearchInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure_text` | `str` | *required* | PDB-format text of the query structure. |

### `FoldseekClusterInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structures` | `list[str]` | *required* (≥2) | PDB-format text strings to cluster. |
| `structure_ids` | `list[str] \| None` | `None` | Optional IDs per structure (default: `'structure_0'`, `'structure_1'`, ...). |

### `FoldseekMultimerSearchInput`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure_text` | `str` | *required* | Multi-chain PDB-format text of the query complex. |

## Configuration

### `FoldseekSearchConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_mode` | `Literal["remote", "local"]` | `"remote"` | `remote` hits the public server; `local` runs the Foldseek CLI. |
| `databases` | `list[FoldseekDatabase]` | `["pdb100", "afdb50"]` | Remote-only — server-hosted reference databases. |
| `mode` | `Literal["3diaa", "tmalign", "lolalign"]` | `"3diaa"` | Remote-only — alignment mode. |
| `poll_interval_seconds` | `float` | `5.0` | (advanced, remote-only) Delay between status polls. |
| `timeout_seconds` | `float` | `600.0` | (advanced, remote-only) Max wall-clock time. |
| `local_db` | `str \| None` | `None` | Local-only (required when `search_mode="local"`) — path to a Foldseek DB. |
| `num_threads` | `int` | `4` | (advanced, local-only) CPU threads. |

`FoldseekDatabase` accepts: `pdb100`, `afdb50`, `afdb-swissprot`, `afdb-proteome`, `mgnify_esm30`, `gmgcl_id`, `BFVD`, `cath50`, `bfmd`.

### `FoldseekClusterConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_seq_id` | `float` | `0.0` | Sequence-identity threshold (0-1); 0.0 lets 3Di structural similarity dominate. |
| `cov` | `float` | `0.8` | Coverage threshold for the alignment. |
| `cov_mode` | `Literal[0, 1, 2]` | `0` | (advanced) 0 = bidirectional, 1 = target, 2 = query. |
| `num_threads` | `int` | `4` | (advanced) CPU threads. |

### `FoldseekMultimerSearchConfig`

Same shape as `FoldseekSearchConfig`. Remote mode wraps `mode` as `complex-{mode}` on the wire (e.g. `complex-3diaa`); the wrapper handles this transparently. The `databases` default is `["pdb100"]` — the multimer-aware subset.

## Output Specification

```python
# foldseek-search
FoldseekSearchOutput(
    ticket_id: str,                    # Remote ticket ID; "" in local mode
    hits: list[FoldseekHit],           # All alignment hits
    num_hits: int,                     # len(hits)
    databases_queried: list[str],      # Echoes config.databases (or [local_db])
    result_url: str,                   # Remote archive URL; "" in local mode
)

# foldseek-cluster
FoldseekClusterOutput(
    clusters: list[FoldseekCluster],   # representative_id + member_ids per cluster
    num_clusters: int,
    num_structures: int,
)

# foldseek-multimer-search — same shape as FoldseekSearchOutput
```

`FoldseekHit` (and the type-aliased `FoldseekMultimerHit`) — 12 fields per the standard BLAST M8 columns: `database`, `target_id`, `sequence_identity` (normalized to [0, 1]), `alignment_length`, `mismatches`, `gap_openings`, `query_start`, `query_end`, `target_start`, `target_end`, `evalue`, `bit_score`.

**Supported export formats:** `json`

## Interpreting Results

**E-value cutoffs (rule of thumb):**
- `evalue < 1e-10`: confidently homologous
- `1e-10 ≤ evalue < 1e-3`: likely related; inspect alignment
- `evalue ≥ 1e-3`: weak; treat as exploratory

**Sequence identity:**
High structural similarity at low sequence identity is precisely the regime where Foldseek beats sequence search. Don't filter by `sequence_identity` if you're looking for distant homologs.

**Cluster results:**
Clusters with one member are isolated structures (no structurally similar peers in the input set). Larger clusters typically share fold and may also share function. The representative is the structure Foldseek picks as the cluster centroid (chosen for connectivity to other members).

**Edge cases:**
- Server timeout / queue: the default 600s timeout covers most queries; raise `timeout_seconds` for very large queries or during server load spikes.
- Empty `pdb100` results but populated `afdb50`: the protein has no experimental structure but predicted structural homologs exist.
- Multimer mismatch: the server may return empty results when query/target chain counts differ substantially (open Foldseek issue).

## Quick Start Examples

**Example 1: Remote single-chain search — find structural neighbors for TP53 in PDB.**

```python
from proto_tools.tools.database_retrieval import (
    AlphaFoldDBFetchConfig, AlphaFoldDBFetchInput, run_alphafold_db_fetch,
)
from proto_tools.tools.structure_alignment import (
    FoldseekSearchConfig, FoldseekSearchInput, run_foldseek_search,
)

afdb = run_alphafold_db_fetch(
    AlphaFoldDBFetchInput(uniprot_id="P04637"),
    AlphaFoldDBFetchConfig(structure_format="pdb"),
)
output = run_foldseek_search(
    FoldseekSearchInput(structure_text=afdb.structure_text),
    FoldseekSearchConfig(databases=["pdb100"]),
)
for hit in sorted(output.hits, key=lambda h: h.evalue)[:5]:
    print(f"{hit.target_id}: evalue={hit.evalue:.2e}, identity={hit.sequence_identity:.1%}")
```

**Example 2: Local single-chain search against a user-built database.**

```python
output = run_foldseek_search(
    FoldseekSearchInput(structure_text=my_pdb_text),
    FoldseekSearchConfig(search_mode="local", local_db="/data/my_foldseek_db", num_threads=8),
)
print(f"{output.num_hits} hits against {output.databases_queried[0]}")
```

**Example 3: Cluster a candidate set of designed structures.**

```python
from proto_tools.tools.structure_alignment import (
    FoldseekClusterConfig, FoldseekClusterInput, run_foldseek_cluster,
)

output = run_foldseek_cluster(
    FoldseekClusterInput(structures=[design_a, design_b, design_c, design_d]),
    FoldseekClusterConfig(),
)
for cluster in output.clusters:
    print(f"  {cluster.representative_id}: {len(cluster.member_ids)} members")
```

**Example 4: Multimer search — find complexes structurally similar to a binder + target pair.**

```python
from proto_tools.tools.structure_alignment import (
    FoldseekMultimerSearchConfig, FoldseekMultimerSearchInput, run_foldseek_multimer_search,
)

output = run_foldseek_multimer_search(
    FoldseekMultimerSearchInput(structure_text=multimer_pdb_text),
    FoldseekMultimerSearchConfig(databases=["pdb100"]),
)
print(f"{output.num_hits} multimer hits")
```

## Best Practices & Gotchas

1. **Don't filter Foldseek hits by `sequence_identity`.** This defeats the purpose of structure search; use `evalue` or `bit_score`.
2. **Submit PDB, not mmCIF.** Convert upstream (e.g. via `alphafold-db-fetch` with `structure_format="pdb"`) if your structure is in CIF.
3. **Clustering uses 3Di structural similarity, not sequence identity.** The default `min_seq_id=0.0` is intentional — set it >0 only when you also want a sequence-similarity floor.
4. **Cache responsibly.** All three tools are `cacheable=True`; subsequent calls with the same inputs + config skip the work. Polling parameters and threads are correctly excluded from the cache key.
5. **Multimer wire format:** the wrapper transparently encodes `mode` as `complex-{mode}` for the multimer endpoint. Pass plain `"3diaa"` / `"tmalign"` / `"lolalign"` in the config.
6. **Public-server fairness.** No documented rate limit, but the search server is best-effort. For batch sweeps over hundreds of queries, use local mode.

## Related Tools

- [`alphafold-db-fetch`](../../database_retrieval/alphafold_db/README.md) — upstream source of query structures by UniProt accession; the canonical chain is AFDB → Foldseek → PDB-fetch.
- [`pdb-fetch-entry`](../../database_retrieval/pdb/README.md) — downstream metadata lookup for top hits' PDB IDs.
- [`tmalign`](../tmalign/README.md) — pairwise structural alignment; useful for re-scoring a small number of Foldseek hits at higher precision.
- [`usalign`](../usalign/README.md) — generalization of TMalign to complexes; pair with `foldseek-multimer-search` for multi-chain re-scoring.

## References

- [Foldseek paper](https://www.nature.com/articles/s41587-023-01773-0) — van Kempen et al., *Nature Biotechnology* 2024
- [Foldseek-Multimer paper](https://www.nature.com/articles/s41592-025-02593-7) — Kim et al., *Nature Methods* 2025
- [Foldseek GitHub](https://github.com/steineggerlab/foldseek)
- [Foldseek search server](https://search.foldseek.com)
