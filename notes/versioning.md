# Toolkit Versioning

Every toolkit has a version, and every tool output records the version that produced it. A version is pinned to a hash of the toolkit's source, so a stored result can always be traced back to the exact code behind it.

## versions.yaml

Each toolkit directory holds a `versions.yaml` mapping version to source hash, newest first:

```yaml
1.1.0: 9f3c1a77d0e2
1.0.0: 4b7e02a91c3f
```

The first key is the current version. All tools in a toolkit share it. The output's `tool_id` identifies which operation ran. Dates, commits, and change descriptions live in git, not in the file (see [History](#history)).

## What the hash covers

`proto_tools/utils/tool_versions.py` computes a 12-hex SHA-256 over:

- every file in the toolkit directory except `README.md`, `cite.bib`, `links.yaml`, `license.yaml`, `versions.yaml`, `examples/`, Markdown, notebooks, dotfiles, the root `__init__.py` (it only re-exports tools), and root `example_*` data files (read only by `example_input()`).
- the shared environment named in `standalone/shared_env.txt`, if any.
- the category's top-level modules, such as `structure_prediction/dispatch.py` or `masked_models/shared_data_models.py`, except its `__init__.py`, which only re-exports toolkits.
- the first-party modules defining the tools' input, config, and output models (their MRO), wherever they live.

Python comments, trailing whitespace, and blank lines are removed before hashing,
so those edits do not change the hash. Docstrings are included in the hash.
Line endings are normalized for every file.

Framework code (`proto_tools/utils/`, `tool_registry.py`) is not hashed because
an edit there would change every toolkit's hash at once. Framework changes are
therefore not reflected in tool versions.

Inside a git checkout, only tracked and untracked-but-not-ignored files count, so git-ignored build artifacts (BLAST database files, standalone helper copies) never move the hash. A wheel install hashes the files it ships.
`test_hashed_files_ship_in_the_wheel` checks that these are the same set.

## Bumping a version

Add a new top entry whenever the source hash changes, so that each version always refers
to the same source. `tests/style_consistency_tests/test_toolkit_versions.py` fails otherwise, and prints the lines to paste:

```
esmfold source changed: versions.yaml has 1.0.0 → 4b7e02a91c3f, current hash is 9f3c1a77d0e2
Add a new top entry to proto_tools/tools/structure_prediction/esmfold/versions.yaml:
  1.0.1: 9f3c1a77d0e2   (patch: no output change)
  1.1.0: 9f3c1a77d0e2   (minor: outputs may differ)
  2.0.0: 9f3c1a77d0e2   (major: schema break)
```

Pick the level by what a caller sees:

| Bump | When |
|---|---|
| Major | Input, config, or output schema changes incompatibly |
| Minor | The same call can return different results: new weights, a changed default, an upstream release |
| Patch | Results are unchanged: refactors, logging, docstrings |

While a branch is in progress, keep one new entry and update its hash as the source changes, rather than adding an entry per commit.

Use `scripts/update_versions.py` to write the entries. You can update all
toolkits with changed source hashes or name specific toolkits.

```bash
python scripts/update_versions.py --bump patch            # add a patch entry to each changed toolkit
python scripts/update_versions.py --bump minor esmfold    # only esmfold
python scripts/update_versions.py --in-place              # update the top entry's hash, e.g. one added on this branch
```

If a toolkit's top entry already matches its current source hash, no update is needed. Two branches bumping the same toolkit conflict in `versions.yaml`, which forces the second to reconcile.

A new toolkit starts at `1.0.0` with its current hash.

## Collapsing history

Only the newest entry is checked against the source. Older entries are kept as a record and are not verified. You can rewrite the
history as long as the file stays well formed and its top hash matches.

- **Collapse a run of bumps** into one entry above the last version you keep. If a downstream copy of the repo is at `1.2.0` and development since then went through `1.2.1` to `1.5.0`, replace those entries with a single `1.3.0: <current hash>`, picking the level of the largest bump collapsed.
- **Reset a toolkit** by deleting its `versions.yaml`: the test then fails with the `1.0.0: <hash>` line to write.
- **Rehash in place** when the hashing itself changes (what `tool_versions.py` covers or how it normalizes) but no toolkit's code does: replace each top entry's hash (`--in-place`) rather than adding versions, since the code behind each version is the same.

Outputs stamped with a dropped version still carry it, but it no longer resolves through `versions.yaml` in a repo whose history never had it.

## The version field on outputs

The `@tool` wrapper stamps `tool_version` on every `BaseToolOutput`: the current version from `versions.yaml`, suffixed `+dirty` when the toolkit's source no longer matches it. The hash is computed once per toolkit per process. A `+dirty` version means the output came from unversioned code, such as an uncommitted edit, and a warning is logged the first time. Outputs from remote dispatch are stamped by the caller, like `tool_id`.

`ToolRegistry.get_version(key)` returns a tool's current version. An out-of-tree toolkit opts in by adding a `versions.yaml`. Without one, `tool_version` is `None` and nothing is hashed.

To learn which tools a block of code actually called, and their versions, wrap it in `ToolRegistry.record_calls()`. The yielded dict maps each tool key called inside the block to the `tool_version` its outputs carry, in first-call order:

```python
with ToolRegistry.record_calls() as calls:
    run_pipeline()
calls  # {"esmfold2-prediction": "<version>", "ipsae-scoring": "<version>"}
```

It records the call itself, so a cache hit, a call served by a dispatch backend (Modal), a failed call, and a tool called by another tool all count. Blocks nest, and every open block records a call. The record is a `ContextVar`, so concurrent programs on other threads or asyncio tasks keep separate records. Work that runs in a copy of the block's context is recorded, which includes the worker threads `ToolPool` starts. A thread started without one, such as a plain `threading.Thread`, records nothing. proto-language uses it to write the tools a program ran into `run_info.json`.

## History

The commit that added a version is the one that introduced its hash, which survives squash merges and carries the PR number in its message:

```bash
git log -S 9f3c1a77d0e2 --format='%H %as %s' -- proto_tools/tools/structure_prediction/esmfold/versions.yaml
```

To look at the source of a version, check out that commit.
