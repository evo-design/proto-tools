# Vendored CodonFM (Encodon) source

All Python files under this `src/` directory are **vendored verbatim** from the upstream
CodonFM repository. They are the model, inference, tokenizer, and data modules required to load
and run the published Encodon checkpoints; they are kept byte-for-byte identical to upstream so
those checkpoints unpickle/load against the exact module topology they were trained with.

- **Source:** [NVIDIA-BioNeMo/CodonFM](https://github.com/NVIDIA-BioNeMo/CodonFM)
- **Commit (pinned):** `0561907ca25adae7ecbe685e8fc2882e634b1f91`
- **License:** Apache-2.0 (upstream code). The public Encodon model *weights* are separately
  licensed under the NVIDIA Open Model License — see the toolkit `license.yaml`.
- **Files vendored:** the 27 `.py` files under `src/` (see each file's header for its exact
  upstream path).

## What was changed

**Nothing in the file bodies.** Each file has exactly a 4-line comment header prepended:

```
# ruff: noqa
# Vendored verbatim from NVIDIA-BioNeMo/CodonFM @ <sha> (<upstream/path>); Apache-2.0.
# Only this header is prepended; the body is unchanged so the published Encodon
# checkpoints load and run exactly as upstream. Do not edit — re-vendor to update.
```

Everything after line 4 is the upstream file unchanged (including its own SPDX/Apache-2.0
header). Do not edit these files by hand — re-vendor from upstream at a new pinned SHA instead.

## Re-verify (byte-for-byte)

For each file, strip the 4-line header and diff the remainder against the pinned upstream commit:

```bash
f=inference/encodon.py
sha=0561907ca25adae7ecbe685e8fc2882e634b1f91
upath=$(sed -n '2p' "$f" | sed -E 's/.*\((.+)\); Apache.*/\1/')
diff <(tail -n +5 "$f") <(curl -sSL "https://raw.githubusercontent.com/NVIDIA-BioNeMo/CodonFM/$sha/$upath")
```

An empty diff for every file confirms the vendored copy is unchanged from upstream.
