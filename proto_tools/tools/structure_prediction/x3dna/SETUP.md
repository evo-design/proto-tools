# X3DNA Setup (gated)

The `x3dna-fiber` tool wraps the `fiber` program from **X3DNA v2.4**, which is
distributed under **CC-BY-NC-4.0** and gated behind a free registration on the
3DNA Forum. It cannot be auto-downloaded, so you provision it once by hand. The
steps below are copy-paste once you have accepted the terms.

## 1. Register and download

1. Go to [x3dna.org](https://x3dna.org/) and register / accept the terms.
2. Download the X3DNA v2.4 archive for your platform, for example
   `x3dna-v2.4-linux-x86_64.tar.gz` (Linux) or the macOS build.

## 2. Stage it into the managed cache

Placing the install in the managed cache means the tool finds it automatically,
with **no environment variable to set**. Adjust the archive name/path to what you
downloaded:

```bash
# Resolve the cache location the tool reads from (default ~/.proto/proto_model_cache):
X3DNA_CACHE="${PROTO_MODEL_CACHE:-${PROTO_HOME:-$HOME/.proto}/proto_model_cache}/x3dna"
mkdir -p "$X3DNA_CACHE"

# Extract the downloaded archive, then copy the install root (the directory that
# contains bin/ and config/) into the cache so bin/fiber lands at $X3DNA_CACHE/bin/fiber:
tar -xzf ~/Downloads/x3dna-v2.4-linux-x86_64.tar.gz -C /tmp
cp -a /tmp/x3dna-v2.4/. "$X3DNA_CACHE/"
```

## 3. Verify

```bash
X3DNA="$X3DNA_CACHE" "$X3DNA_CACHE/bin/fiber" -b -seq=ACGT /tmp/x3dna_smoke.pdb \
  && grep -c '^ATOM' /tmp/x3dna_smoke.pdb \
  && echo "X3DNA ready at $X3DNA_CACHE"
```

A non-zero atom count and the `X3DNA ready` line mean `x3dna-fiber` will run with
no further configuration.

## Alternative: point at an existing install

If you would rather keep X3DNA elsewhere, skip step 2 and set the standard
variable instead (the tool also accepts a `x3dna_dir` config value):

```bash
export X3DNA=/path/to/x3dna-v2.4   # the directory containing bin/fiber
```

> Note: if you have set `PROTO_MODEL_CACHE=IN_ENV` or `=NONE`, the per-environment
> cache cannot be pre-populated; use the `export X3DNA=...` alternative above.
