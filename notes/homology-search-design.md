# Homology Search — Design

Design doc for a generalized MMseqs2-based MSA generation tool (`mmseqs2-homology-search`) that replaces `colabfold-search` in structure predictors. Supersedes the WIP on branch [`colabfold-search-paired-msa`](https://github.com/proto-bio/proto-tools/tree/colabfold-search-paired-msa) (ref-only; we crib code, not commits).

Closes [#543](https://github.com/proto-bio/proto-tools/issues/543) (paired MSA support).

## Motivation

`colabfold-search` has three load-bearing limitations:

1. **Protein only.** AF3 RNA inference is blocked — no path to pull RNA MSAs from `rnacentral` / `rfam` / `nt`.
2. **Hardcoded DB assumptions.** Paths like `CHIMERA_COLABFOLD_DB_LOCATION = "/large_storage/hielab/brk/databases/colabfold"` and a single `database_name` field. No on-demand download, no platform portability, no way for a structure predictor to declare which DB(s) it wants.
3. **Unpaired only.** Multimer predictors (AF3, Boltz-2, Chai-1, Protenix) want paired MSAs for interface coevolution signal; `colabfold-search` cannot produce them. #543's WIP branch proves out the surface but leaves the local GPU path + predictor plumbing open.

All three are one design away from each other: a **dataset registry** that holds per-dataset metadata (molecule type, URLs, index recipe, pairing support, MMseqs2 flags) + a **generalized tool** that takes a registry key and produces grouped paired/unpaired MSAs.

## Scope

**In scope:**

- Dataset registry at `proto_tools/tools/sequence_alignment/databases/` (shared with `colabfold-search` from day 1 — see [Dataset Registry](#dataset-registry))
- `mmseqs2-homology-search` tool at `proto_tools/tools/sequence_alignment/mmseqs2_homology_search/`
- GPU-accelerated by default (MMseqs2-GPU, same engine as AlphaFast)
- Protein + nucleotide (RNA/DNA) support via registry-driven MMseqs2 flags
- Paired MSAs via grouped input (shape from #543)
- Migration of 6 structure predictors: AF2, AF3, Chai-1, Protenix, Boltz-2, BioEmu
- Deprecation of `colabfold-search` (last, after all callers migrate)

**Out of scope:**

- `mmseqs-search-proteins` / `mmseqs-search-genomes` / `mmseqs-clustering` — different workflow (annotation / clustering, not MSA generation). Left alone.
- Cloud API for hosted datasets — belongs in `proto-tools-api`, not here. Registry schema should be forward-compatible (URL → API endpoint swap).
- Template search (structure-level). Separate tool, separate issue.

## Dataset Registry

Lives at `proto_tools/tools/sequence_alignment/databases/` — shared between `colabfold-search` (today) and `mmseqs2-homology-search` (planned). Scoped to `sequence_alignment/` rather than the top-level `proto_tools/databases/` because all current and planned consumers live under that category; lift to top-level when a structural-database or clustering-database consumer appears.

**Status:** schema + `DatasetRegistry` + `get_dataset_dir()` helper + the `uniref30-2302` entry landed in the colabfold-search rmtree-fix PR. `DatasetManager.ensure()` / `provision_datasets()` / CLI / remaining entries ship with `mmseqs2-homology-search`.

### Schema

```python
# proto_tools/tools/sequence_alignment/databases/registry.py

class DatasetEntry(BaseModel):
    """One searchable homology database."""

    name: str                              # registry key, kebab-case; e.g. "uniref30-2302"
    molecule_type: Literal["protein", "rna", "dna"]
    display_name: str                      # human-readable
    description: str                       # one-liner for UI
    citation_doi: str | None               # paper DOI if applicable

    # Download
    urls: list[DownloadSpec]               # one or more files to fetch (tarball + newtax + …)
    total_download_bytes: int              # for UI / disk precheck
    total_disk_bytes: int                  # post-extract, post-index

    # Indexing
    index_recipe: IndexRecipe              # how to turn downloads into an MMseqs2 DB

    # MMseqs2 search-time
    mmseqs_flags: MmseqsFlags              # sensitivity, prefilter mode, max-seqs, etc.

    # Capabilities
    supports_gpu: bool                     # requires .idx_pad
    supports_pairing: bool                 # requires tax-tagged index
    min_gpu_memory_gb: float | None        # None = CPU-viable or negligible

    # A3M adapter
    a3m_adapter: Literal["colabfold", "plain", "rna"]   # how to stitch m8 hits into an A3M


class DownloadSpec(BaseModel):
    url: str
    filename: str                          # local name (for resume + checksum)
    sha256: str | None                     # verify after download
    required: bool = True                  # e.g. newtaxonomy is optional for non-pairing use


class IndexRecipe(BaseModel):
    """Commands to run after download, in order."""
    steps: list[IndexStep]                 # e.g. extract, createdb, makepaddedseqdb, createtaxdb
    output_files: list[str]                # files whose presence means "indexed"
    paired_output_files: list[str]         # additional files required for paired search


class MmseqsFlags(BaseModel):
    sensitivity: float
    prefilter_mode: int = 0
    max_seqs: int = 300
    extra_args: list[str] = []             # escape hatch; avoid if possible
```

### Registry storage

- Entries declared as module-level `DatasetEntry(…)` literals in `proto_tools/tools/sequence_alignment/databases/entries/`, one file per dataset (`uniref30_2302.py`, `rnacentral.py`, `rfam.py`, `colabfold_envdb_202108.py`, `bfd.py`, `mgnify.py`, `nt.py`, …). Each module calls `DatasetRegistry.register(ENTRY)` at import time.
- `proto_tools/tools/sequence_alignment/databases/__init__.py` exposes `DatasetRegistry.get(name)`, `.list_all()`, `.by_molecule_type(type)`, `get_dataset_dir(name)`, `get_databases_root()`. In a follow-up PR it also exposes `DatasetManager.ensure(name, …)` and `provision_datasets(names, …)`.
- No remote manifest, no auto-discovery. Adding a dataset = one PR.

### On-disk layout

Database files themselves live under `$PROTO_MODEL_CACHE` (default `$PROTO_HOME/proto_model_cache/`), in a `databases/` subdirectory — the same root as HuggingFace/PyTorch weight caches. This is intentional: setting `PROTO_MODEL_CACHE` to a shared NFS path makes datasets shareable across users on the same machine without any per-tool env var, exactly like model weights today. The on-disk layout is **dataset-scoped, not tool-scoped** (`databases/uniref30_2302/`, not `databases/colabfold/uniref30_2302/`) so any MMseqs2-based tool reads the same files.

```
$PROTO_MODEL_CACHE/                       (default: $PROTO_HOME/proto_model_cache/)
├── huggingface/, torch/                  existing weight caches (unchanged)
└── databases/                            ← new, managed by mmseqs2-homology-search registry
    ├── uniref30_2302/
    │   ├── .downloaded                   marker — all files in urls present + sha256 match
    │   ├── .indexed                      marker — index_recipe.output_files all present
    │   ├── .paired_indexed               marker — paired_output_files present (optional)
    │   ├── uniref30_2302_db*             MMseqs2 DB files
    │   ├── uniref30_2302_db.idx_pad      GPU index
    │   └── uniref30_2302_db_taxonomy     files (when paired)
    ├── rnacentral/
    └── …
```

Download/index is idempotent and resumable: check markers before re-running any step. One `DatasetManager.ensure(name, need_pairing=bool)` call at tool dispatch time — it's a no-op when everything's present.

No separate `PROTO_MMSEQS2_HOMOLOGY_SEARCH_DATABASES_DIR` env var — `PROTO_MODEL_CACHE` is already the right knob. Update `notes/storage.md` to document `databases/` when the first registry entry lands.

### Initial datasets (launch set)

| Name | Type | Size (download / indexed) | Pairing | Use |
|---|---|---|---|---|
| `uniref30-2302` | protein | ~99 GB / ~250 GB | ✓ | AF2/AF3/Chai-1/Protenix/Boltz-2 — primary protein MSA |
| `colabfold-envdb-202108` | protein | ~110 GB / ~650 GB | ✓ | metagenomic, optional per-predictor |
| `bfd` | protein | ~270 GB / ~1.8 TB | ✗ | AF2 path |
| `rnacentral` | rna | ~30 GB / ~60 GB | ✗ | AF3 RNA |
| `rfam` | rna | ~1 GB / ~3 GB | ✗ | AF3 RNA |
| `nt` | dna | ~200 GB / ~400 GB | ✗ | AF3 RNA (covers DNA→RNA fallback hits) |

Not all materialize at launch — registry entry lands + download/index kicks in on first use of that key.

## Bulk Provisioning

The primary audience is **end users** who want to pre-fetch one or several datasets instead of waiting on a ~2-hour lazy download the first time they run a structure predictor. The *same* entrypoint is what our Modal deployment (planned) will use during image build — serverless containers cannot do a 100 GB lazy download at cold start, but the volume-build step is just the user CLI invoked from a Modal image recipe. One code path, two consumers. CI cache-warmup jobs are the third natural consumer.

**Interface**: a Python CLI that wraps `DatasetManager.ensure()` for one or many entries, exposed as both a console script (`proto-tools provision …`) for users and an importable function (`proto_tools.tools.sequence_alignment.databases.provision_datasets([...])`) for Modal / CI. Not a bash script — reusing the registry directly avoids duplicating URL lists and index recipes between the user path and the deployment path, which is exactly the duplication that bit us with `setup_databases.sh` vs. hardcoded Chimera DB paths today.

```bash
# One dataset
proto-tools provision uniref30-2302

# Multiple, explicit
proto-tools provision uniref30-2302 rnacentral rfam

# All registered (CI / Modal image build)
proto-tools provision --all

# Filter by molecule type
proto-tools provision --molecule-type rna

# Include tax indexes for pairing (otherwise built lazily on first paired use)
proto-tools provision uniref30-2302 --include-paired

# Dry run — show disk + bandwidth estimate, no downloads
proto-tools provision --all --dry-run
```

(`python -m proto_tools.tools.sequence_alignment.databases.provision …` works identically; the console script is added as a `[project.scripts]` entry in `pyproject.toml` for user-facing ergonomics.)

**Behavior:**

- Exits non-zero if `sum(total_disk_bytes)` exceeds free space on the `$PROTO_MODEL_CACHE` filesystem (with 20% safety margin). `--force` to override.
- Sequential per-dataset (they're large enough that parallel downloads fight for bandwidth). Parallelism *within* a dataset comes for free via aria2c's `--max-connection-per-server=8`.
- Idempotent: rerun is a no-op when `.downloaded` / `.indexed` / `.paired_indexed` markers are all present.
- Emits a structured JSON summary at the end (entries attempted, completed, skipped, failed) for consumption by CI / Modal build logs.

**Modal deployment hook** (future, when we wire up the hosted version) — the same `provision_datasets` function the user-facing CLI wraps, invoked at image build:

```python
from proto_tools.tools.sequence_alignment.databases import provision_datasets

# In the Modal image definition
image = (
    modal.Image.debian_slim()
    .pip_install("proto_tools[mmseqs2-homology-search]")
    .run_function(
        lambda: provision_datasets(
            ["uniref30-2302", "colabfold-envdb-202108", "rnacentral", "rfam"],
            include_paired=True,
        ),
        volumes={"/cache": db_volume},
        env={"PROTO_MODEL_CACHE": "/cache"},
    )
)
```

The volume persists across container starts, so the download cost is paid once per Modal app revision, not per request. `provision_datasets()` is just `DatasetManager.ensure(name)` loops — identical code path whether invoked by the user CLI, a Modal image builder, a CI warmup job, or lazy first-use on a dev machine. **No separate "deployment mode" code, and no duplicated URL/index knowledge between a shell setup script and the Python tool.**

## `mmseqs2-homology-search` Tool Surface

`proto_tools/tools/sequence_alignment/mmseqs2_homology_search/mmseqs2_homology_search.py`.

### Input

Nested-list shape from #543 (backward-compatible with flat lists):

```python
class Mmseqs2HomologySearchQuery(BaseModel):
    sequence: str
    sequence_id: str | None = None
    molecule_type: Literal["protein", "rna", "dna"] | None = None   # inferred if None

class Mmseqs2HomologySearchInput(BaseToolInput):
    queries: list[Mmseqs2HomologySearchQuery | list[Mmseqs2HomologySearchQuery]]
    # Flat item = singleton (unpaired) group.
    # Inner list = paired group (rows cross-chain aligned by taxonomy).
```

The `before` normalizer accepts strings / tuples / Query objects like `colabfold-search` does today. Groups preserve order.

### Config

```python
class Mmseqs2HomologySearchConfig(BaseConfig):
    datasets: list[str] = ["uniref30-2302"]    # registry keys; must all share molecule_type
    use_metagenomic: bool = False              # convenience: also hits "colabfold-envdb-202108" if present for molecule_type
    use_gpu: bool = True                       # GPU default; falls back to CPU with warning if no GPU
    pairing_strategy: Literal["greedy", "complete"] = "greedy"
    sensitivity: float | None = None           # None = use registry default
    output_dir: str | None = None              # defaults to $PROTO_HOME/mmseqs2_homology_search
    search_mode: Literal["local", "remote"] = "local"   # remote reserved for future cloud API
    timeout: int = 3600
```

- `datasets` is a list, not a single value — matches how AF2/AF3 consume *multiple* DBs (UniRef30 + envdb + BFD).
- `use_gpu=True` is the default. Registry entry must have `supports_gpu=True` for every dataset in `datasets`, else validator errors up front with a fix suggestion ("re-index with `makepaddedseqdb` or set `use_gpu=False`").
- Pairing auto-triggers on any group with ≥2 queries. Validator errors if any selected dataset has `supports_pairing=False` and a paired group is present.

### Output

```python
class Mmseqs2HomologySearchResult(BaseModel):
    sequence_ids: list[str]                    # per group
    msas: list[MSA | None]                     # per-chain unpaired
    paired_msas: list[MSA | None]              # per-chain paired, row-synchronized; None for singleton groups
    datasets_searched: list[str]               # registry keys hit
    num_homologs_found: list[int]              # per chain

class Mmseqs2HomologySearchOutput(BaseToolOutput):
    results: list[Mmseqs2HomologySearchResult]  # one per input group
```

`_export_output` writes one A3M (+ paired A3M if present) per chain per group.

## Paired MSA Mechanics (closes #543)

Two paths:

- **Remote** (future, via `proto-tools-api`): call a paired-search endpoint. The existing branch's `run_mmseqs2(use_pairing=True, pairing_strategy=...)` code ports verbatim. Gated behind `search_mode="remote"` + cloud-API availability.
- **Local GPU**: new work this design unblocks.
  - Registry entry declares `supports_pairing=True` + a `createtaxdb` step in `index_recipe`.
  - Search: run the standard MMseqs2 profile pipeline per chain against the tax-tagged DB. Collect per-hit `taxID`.
  - Pair: intersect taxIDs across chains in a group (greedy: one hit per species per chain, best-by-pident; complete: all combinations).
  - Write paired A3M with row-synchronized order across chains. Adapter = same format AF2/AF3/Chai-1/Boltz-2/Protenix consume for paired MSAs today (verify per consumer during migration).
  - Reference implementation to port: `colabfold/batch.py::pair_msa`.

Blocker on the old branch ("no tax-tagged DB materialized yet") dissolves because the registry's `ensure()` builds the tax index on first paired use.

## Structure Predictor Integration

Each predictor gets a config field:

```python
preferred_datasets: list[str] = [...]          # registry keys, predictor-specific defaults
```

Example defaults:

- AF2: `["uniref30-2302", "bfd"]` (+ `colabfold-envdb-202108` when `use_metagenomic`)
- AF3 (protein chains): `["uniref30-2302"]` (+ envdb when enabled)
- AF3 (RNA chains): `["rnacentral", "rfam", "nt"]`  ← unblocks RNA
- Chai-1 / Protenix / Boltz-2: `["uniref30-2302"]` initially; tune per-predictor in migration PRs
- BioEmu: `["uniref30-2302"]`

Each predictor owns its own "grouped input → `Mmseqs2HomologySearchInput`" builder (i.e. for AF3 multimers: one group per complex, one query per chain). The output consumer (`_assign_msas_to_input_json` in AF3, analogous in others) reads `result.msas` and `result.paired_msas` and writes into the predictor's native input format.

**One-patch-per-predictor follow-ups** (per #543) stay scoped — each is a small, reviewable change against a stable `mmseqs2-homology-search` contract.

## Migration Plan

Phased so nothing breaks mid-flight:

1. **Registry + tool land, `colabfold-search` migrates its DB default.**
   PR 1 (landed): `sequence_alignment/databases/` registry schema + `uniref30-2302` entry + `get_dataset_dir()` helper; `colabfold-search` default `msa_db_dir` wired to the registry.
   PR 2: `DatasetManager.ensure()` + `provision_datasets()` + `proto-tools provision` CLI (registry gets materialization machinery).
   PR 3: `mmseqs2-homology-search` tool (protein + unpaired only initially, for smoke testing against UniRef30).
   PR 4: paired MSA local path (tax index + pair writer). Closes #543 *infrastructurally* (but no predictor consumes it yet).

2. **Predictor migrations, one PR each.** Order:
   1. AF3 — highest payoff (unblocks RNA + multimer paired MSAs simultaneously). Adds RNA datasets to registry.
   2. Boltz-2 — validates paired-MSA format compatibility in a second consumer.
   3. Chai-1, Protenix, AF2, BioEmu — mechanical.

   Each predictor PR: swap the `ColabfoldSearchInput/Output` calls for `Mmseqs2HomologySearchInput/Output`, add `preferred_datasets` to the predictor's config with sensible defaults, keep `colabfold-search` imports dead-but-not-removed.

3. **Deprecate `colabfold-search`.**
   Emit `DeprecationWarning` on import. One release later: delete the directory. Reference docs auto-regenerate.

Each phase is independently revertable.

## Open Questions

- **Tax DB build cost.** `createtaxdb` over UniRef30 is ~30–60 min one-time on a fat machine. Acceptable as first-paired-use tax. Should we pre-bake and ship the tax files as a separate registry download to skip the local build? → Lean toward **yes**, add an optional `prebuilt_taxonomy_url` to `IndexRecipe` that registry entries can populate.
- **Gated datasets.** Some RNA DBs (e.g. parts of `nt`) require NCBI API keys or throttle aggressively. Registry needs an `auth_env_var` field (like `proto_tools.utils.auth.require_hf_token` but generic). Not launch-blocking.
- **Versioning.** `uniref30-2302` has a date in the name; future versions → new entry (`uniref30-2403`). Don't mutate. Predictors pin exact versions in `preferred_datasets`.
- **GPU fallback UX.** `use_gpu=True` on a CPU-only host → hard error or warn-and-fall-back? → Lean toward **hard error** with clear suggestion (`use_gpu=False`). No silent slowness.
- **Backward-compat shim for `colabfold-search` callers outside this repo.** proto-language + MCP consumers hit it by `tool_key`. Keep the tool registered (warning on use) through one full release cycle.
- **When to lift the registry to top-level `proto_tools/databases/`.** Currently at `proto_tools/tools/sequence_alignment/databases/` since all consumers (present and planned) live under that category. Trigger for the lift: a structural-database (PDB, AlphaFold DB) consumer or a clustering-DB consumer appears. Until then, keeping it under sequence_alignment avoids overpromoting a registry that might invite non-homology datasets.

## Related

- [#543](https://github.com/proto-bio/proto-tools/issues/543) — paired MSA support (closed by this design)
- Branch `colabfold-search-paired-msa` — reference implementation for input/output shape + remote-path pairing
- MMseqs2-GPU paper: https://www.nature.com/articles/s41592-025-02819-8
- AlphaFast (MMseqs2-GPU + paired MSAs in AF3 pipeline): https://github.com/RomeroLab/alphafast
- Lightning-Boltz: https://github.com/RomeroLab/lightning-boltz
- ColabFold API pairing: `colabfold.colabfold.run_mmseqs2(use_pairing=True, pairing_strategy=...)`
- Internal: `notes/storage.md` (PROTO_HOME layout), `notes/tool-environments.md` (standalone env patterns)
