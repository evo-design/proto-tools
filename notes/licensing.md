# Licensing metadata

Every toolkit ships a `license.yaml` recording the terms that govern it. The schema is
enforced by `tests/style_consistency_tests/test_license_consistency.py`, which rejects
unknown keys, so a typo surfaces immediately rather than being silently ignored.

Most fields are descriptive: they tell a reader what the terms are, and nothing checks that
the repository obeys them. Three have teeth, in the sense that a test will fail if the
repository and the metadata disagree:

| field | what a test enforces |
|---|---|
| `code.spdx` / `weights.spdx` | an allowlisted SPDX id resolves its text from `proto_tools/tools/_licenses/{spdx}.txt`; a `Custom (...)` id must inline its text |
| `weights.access` | `hf-gated` must match exactly the toolkits calling `require_hf_token()`, and the gated table in `notes/gated-models.md` |
| `output_sharing` | `false` means no executed example may be committed (below) |

## `output_sharing`

**What it means: may the repository publish what the tool produces?**

A licence usually governs the software and its weights. Some reach further and govern the
tool's *results* as well, and those are what `output_sharing` records. It is a bool, it
defaults to `true` when absent, and it is worth setting to `false` only when upstream terms
genuinely restrict the output rather than the software.

Do not confuse it with its neighbours. They answer different questions:

- `redistribution` — may we **ship the software or weights**? Nearly always about a binary,
  a checkpoint or a source tree, which a notebook contains none of. `redistribution: false`
  is therefore *not* a reason to set `output_sharing: false`; most GPL and academic-licence
  toolkits are `redistribution: false` and publish their example output quite legitimately.
- `commercial_use` — may we **run it** for commercial purposes? About use, not publication.
- `attribution_required` — must we **credit the source** when we pass something on? Output
  may be shared; it just has to carry its credit.

So `output_sharing: false` is a narrow claim: not "these terms are strict" but "these terms
name the output, and publishing it would breach them".

### What setting it to `false` does

Two things, which together keep the repository honest without relying on anyone remembering:

1. `scripts/run_example_notebooks.py` **skips the notebook**. It is never executed, so a
   full re-run cannot quietly repopulate it, and the run reports what it skipped and why.
2. `test_forbidden_output_is_not_committed` **fails if output appears anyway** — committed
   cell outputs in any notebook under `examples/`, or tracked files in an output directory
   there (any directory whose name contains `output` or `result`).

The example therefore ships unexecuted. That is a real cost: the notebook is the toolkit's
main documentation and a reader loses the worked numbers. Accept it only where the terms
actually require it, and say why in the toolkit's `notes` field so the next reader does not
assume it was an oversight and re-run it.

### The two toolkits that set it today

- **`rna_splicing/spliceai2`** — Illumina's SpliceAI2 Model Terms of Use grant the "source
  code, model weights, **and precomputed scores**" for non-commercial use by non-commercial
  organizations, and prohibit "redistribution, and third-party access". The scores are named
  in the grant alongside the code, so committing per-position probabilities publishes
  something the licence withholds. `examples/.gitignore` also keeps `example_output/` out of
  the tree.
- **`epitope_prediction/netmhcpan`** — DTU Health Tech's academic agreement, Section 7, bars
  "publishing any results of benchmark tests run on the Product to a third party without
  HEALTH's prior written consent". The toolkit's tests were already written to assert
  structural and relational properties rather than commit score tables; the notebook follows
  the same rule.

### Adding it to a toolkit

Set `output_sharing: false` in `license.yaml`, explain the clause in `notes`, clear the
notebook (`jupyter nbconvert --clear-output --inplace`), untrack anything already committed
under the toolkit's output directory, and gitignore that directory. The style suite will
confirm the result.

Fields that merely require credit do not belong here — use `attribution_required` and name
the source in the toolkit README, as `database_retrieval/synbiohub` does with a dedicated
`examples/example_output/ATTRIBUTION.md`.
