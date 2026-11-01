# Toolkit Versioning

Every toolkit has a version, and every tool output records the version that produced it. A version is pinned to a hash of the toolkit's source, so a stored result can always be traced back to the exact code behind it.

## versions.yaml

Each toolkit directory holds a `versions.yaml` mapping version to source hash, newest first:

```yaml
1.1.0: 9f3c1a77d0e2
1.0.0: 4b7e02a91c3f
```

The first key is the current version. All tools in a toolkit share it; `tool_id` on an output says which operation ran. Dates, commits, and change descriptions live in git, not in the file (see [History](#history)).

## What the hash covers

`proto_tools/utils/tool_versions.py` computes a 12-hex SHA-256 over:

- every file in the toolkit directory except `README.md`, `cite.bib`, `links.yaml`, `license.yaml`, `versions.yaml`, `examples/`, Markdown, notebooks, and dotfiles;
- the shared environment named in `standalone/shared_env.txt`, if any;
- the category's top-level modules, such as `structure_prediction/dispatch.py` or `masked_models/shared_data_models.py`, except its `__init__.py`, which only re-exports toolkits;
- the first-party modules defining the tools' input, config, and output models (their MRO), wherever they live.

Python comments, trailing whitespace, and blank lines are dropped before hashing, so formatting and comment edits never move it. Docstrings are code. Line endings are normalized for every file.

Framework code (`proto_tools/utils/`, `tool_registry.py`) is not hashed: an edit there would move every toolkit at once, so framework changes are not reflected in tool versions.

Inside a git checkout, only tracked and untracked-but-not-ignored files count, so git-ignored build artifacts (BLAST database files, standalone helper copies) never move the hash. A wheel install hashes the files it ships; `test_hashed_files_ship_in_the_wheel` keeps those the same set.

## Bumping a version

Any change to the hash needs a new top entry: same version, same source. `tests/style_consistency_tests/test_toolkit_versions.py` fails otherwise, and prints the lines to paste:

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

While a branch is in progress, keep one new entry and update its hash as the source changes, rather than adding an entry per commit. Two branches bumping the same toolkit conflict in `versions.yaml`, which forces the second to reconcile.

A new toolkit starts at `1.0.0` with its current hash.

## Collapsing history

Only the newest entry is checked against the source; older entries are a record, never verified. History can therefore be rewritten freely, as long as the file stays well formed and its top hash matches:

- **Collapse a run of bumps** into one entry above the last version you keep. If a downstream copy of the repo is at `1.2.0` and development since then went through `1.2.1` to `1.5.0`, replace those entries with a single `1.3.0: <current hash>`, picking the level of the largest bump collapsed.
- **Reset a toolkit** by deleting its `versions.yaml`: the test then fails with the `1.0.0: <hash>` line to write.
- **Rehash in place** when the hashing itself changes (what `tool_versions.py` covers or how it normalizes) but no toolkit's code does: replace each top entry's hash rather than adding versions, since the code behind each version is the same.

Outputs stamped with a dropped version still carry it, but it no longer resolves through `versions.yaml` in a repo whose history never had it.

## The version field on outputs

The `@tool` wrapper stamps `tool_version` on every `BaseToolOutput`: the current version from `versions.yaml`, suffixed `+dirty` when the toolkit's source no longer matches it. The hash is computed once per toolkit per process. A `+dirty` version means the output came from unversioned code, such as an uncommitted edit, and a warning is logged the first time. Outputs from remote dispatch are stamped by the caller, like `tool_id`.

`ToolRegistry.get_version(key)` returns a tool's current version. An out-of-tree toolkit opts in by adding its own `versions.yaml`; without one, `tool_version` is `None` and nothing is hashed.

## History

The commit that added a version is the one that introduced its hash, which survives squash merges and carries the PR number in its message:

```bash
git log -S 9f3c1a77d0e2 --format='%H %as %s' -- proto_tools/tools/structure_prediction/esmfold/versions.yaml
```

To look at the source of a version, check out that commit.
